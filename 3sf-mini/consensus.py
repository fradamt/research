from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Tuple
import hashlib
import json
import copy

ZERO_HASH = '0'*64
MAX_BACKOFF_INTERVAL_JUSTIFICATION = 8
MAX_BACKOFF_INTERVAL_VOTING = 1

# Chain configuration
@dataclass
class Config:
    num_validators: int

@dataclass(frozen=True)
class Checkpoint:
    hash: str
    slot: int

# Blockchain state
@dataclass
class State:
    config: Config
    latest_justified: Checkpoint
    latest_finalized: Checkpoint
    historical_block_hashes: List[str] = field(default_factory=list)
    justified_checkpoints: List[Checkpoint] = field(default_factory=list)
    justifications: Dict[str, List[bool]] = field(default_factory=dict)

class BackoffType(Enum):
    JUSTIFICATION = "justification"
    VOTING = "voting"

# A vote. In a live implementation this would also include a signature
@dataclass
class Vote:
    validator_id: int
    head: str
    finalized_slot: int
    source: Checkpoint
    target: Checkpoint
    target_block_slot: int

# A block
@dataclass
class Block:
    slot: int
    parent: Optional[str]
    votes: List[Vote] = field(default_factory=list)
    state_root: Optional[str] = None

# Stub for computing block hash, state root...
# (in real life replace with SSZ hashing)
def compute_hash(obj: object):
    if isinstance(obj, tuple):
        serialized = json.dumps([asdict(item) if hasattr(item, '__dataclass_fields__') else item for item in obj], sort_keys=True).encode()
    else:
        serialized = json.dumps(asdict(obj), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()

def compute_backoff_interval(finalized_slot: int, slot: int, backoff_type: BackoffType):
    assert slot >= finalized_slot
    delta = slot - finalized_slot
    if backoff_type == BackoffType.VOTING:
        max_backoff_interval = MAX_BACKOFF_INTERVAL_VOTING
    else:
        max_backoff_interval = MAX_BACKOFF_INTERVAL_JUSTIFICATION
    return min(2**(delta // 4), max_backoff_interval)

# Determines if a candidate slot is justifiable based on an exponential backoff mechanism (with a cap).
# A checkpoint_interval is calculated based on the distance from the last finalized slot, and a slot
#  is justifiable only if it is a multiple. The mechanism helps finality progress under high latency.
def is_justifiable_slot(finalized_slot: int, target_slot: int):
    backoff_interval = compute_backoff_interval(finalized_slot, target_slot, BackoffType.JUSTIFICATION)
    return target_slot % backoff_interval == 0

def is_voting_slot(finalized_slot: int, slot: int):
    backoff_interval = compute_backoff_interval(finalized_slot, slot, BackoffType.VOTING)
    return slot % backoff_interval == 0


# Given a state, output the new state after processing that block
def process_block(state: State, block: Block) -> State:
    state = copy.deepcopy(state)
    # Track historical blocks in the state
    state.historical_block_hashes.append(block.parent)
    while len(state.historical_block_hashes) < block.slot:
        state.historical_block_hashes.append(None)
    # Process votes
    for vote in block.votes:

        if (
            not is_voting_slot(vote.finalized_slot, vote.target.slot)
            or vote.source.slot < state.latest_finalized.slot
            or vote.source not in state.justified_checkpoints
            or vote.target.slot <= vote.source.slot
        ):
            continue

        if vote.target.hash != ZERO_HASH and (
            not is_justifiable_slot(vote.finalized_slot, vote.target.slot)
            or vote.target.hash != state.historical_block_hashes[vote.target_block_slot]
        ):
            continue

        # Track attempts to justify new hashes
        justification_key = compute_hash((vote.finalized_slot, vote.source, vote.target))
        if justification_key not in state.justifications:
            state.justifications[justification_key] = [False] * state.config.num_validators

        if not state.justifications[justification_key][vote.validator_id]:
            state.justifications[justification_key][vote.validator_id] = True

        count = sum(state.justifications[justification_key])

        # If 2/3 voted for the same new valid hash to justify
        if count == (2 * state.config.num_validators) // 3:
            if vote.target.hash != ZERO_HASH:
                state.latest_justified = vote.target
                state.justified_checkpoints.append(vote.target)
            del state.justifications[justification_key]

            # Finalization: if the target is the next valid justifiable
            # slot after the source, wrt the finalized slot in the votes
            if not any(
                is_justifiable_slot(vote.finalized_slot, slot)
                for slot in range(vote.source.slot + 1, vote.target.slot)
            ):
                state.latest_finalized = vote.source
                # Prune old checkpoints
                state.justified_checkpoints = [
                    checkpoint for checkpoint in state.justified_checkpoints 
                    if checkpoint.slot >= state.latest_finalized.slot
                ]

    return state

# Get the highest-slot justified block that we know about
def get_latest_justified_checkpoint(post_states: Dict[str, State]) -> Checkpoint:
    latest = max(   
        post_states.values(),
        key=lambda s: s.latest_justified.slot
    )
    return latest.latest_justified

# Use LMD GHOST to get the head, given a particular root (usually the
# latest known justified block)
def get_fork_choice_head(blocks: Dict[str, Block],
                         root: str,
                         votes: List[Vote],
                         min_score: int = 0) -> str:
    # Start at genesis by default
    if root == ZERO_HASH:
        root = min(blocks.keys(), key=lambda block: blocks[block].slot)

    # Identify latest votes
    latest_votes = {}
    for vote in sorted(votes, key=lambda vote: vote.target.slot):
        latest_votes[vote.validator_id] = vote

    # For each block, count the number of votes for that block. A vote
    # for any descendant of a block also counts as a vote for that block
    vote_weights: Dict[str, int] = {}

    for vote in latest_votes.values():
        if vote.head in blocks:
            block_hash = vote.head
            while blocks[block_hash].slot > blocks[root].slot:
                vote_weights[block_hash] = vote_weights.get(block_hash, 0) + 1
                block_hash = blocks[block_hash].parent

    # Identify the children of each block
    children_map: Dict[str, List[str]] = {}
    for _hash, block in blocks.items():
        if block.parent and vote_weights.get(_hash, 0) >= min_score:
            children_map.setdefault(block.parent, []).append(_hash)

    # Start at the root (latest justified hash or genesis) and repeatedly
    # choose the child with the most latest votes, tiebreaking by slot then hash
    current = root
    while True:
        children = children_map.get(current, [])
        if not children:
            return current
        current = max(children,
                      key=lambda x: (vote_weights.get(x, 0), blocks[x].slot, x))
