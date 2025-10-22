from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
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
    chain_slot: int
    checkpoint_slot: int

# Blockchain state
@dataclass
class State:
    config: Config
    latest_justified: Checkpoint
    latest_finalized: Checkpoint
    historical_block_hashes: List[str] = field(default_factory=list)
    justified_slots: List[bool] = field(default_factory=list)
    justifications: Dict[str, List[bool]] = field(default_factory=dict)

@dataclass(frozen=True)
class FastVote:
    validator_id: int
    slot: int
    head: str

@dataclass(frozen=True)
class SlowVote:
    validator_id: int
    slot: int
    head: str
    source: Checkpoint
    target: Checkpoint

# A block
@dataclass
class Block:
    slot: int
    parent: Optional[str]
    fast_votes: List[FastVote] = field(default_factory=list)
    slow_votes: List[SlowVote] = field(default_factory=list)
    state_root: Optional[str] = None

class BackoffType(Enum):
    JUSTIFICATION = "justification"
    VOTING = "voting"


# Stub for computing block hash, state root...
# (in real life replace with SSZ hashing)
def compute_hash(obj: object):
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

# Given a state, output the new state after processing that block
def process_block(state: State, block: Block) -> State:
    state = copy.deepcopy(state)
    # Track historical blocks in the state
    state.historical_block_hashes.append(block.parent)
    state.justified_slots.append(False)
    while len(state.historical_block_hashes) < block.slot:
        state.justified_slots.append(False)
        state.historical_block_hashes.append(None)
    # Process votes
    for vote in block.slow_votes:
        if (
            vote.source.checkpoint_slot < state.latest_finalized.checkpoint_slot
            or state.justified_slots[vote.source.checkpoint_slot] is False
            or vote.source.hash != state.historical_block_hashes[vote.source.chain_slot]
            or vote.target.checkpoint_slot <= vote.source.checkpoint_slot
            or state.justified_slots[vote.target.checkpoint_slot] is True
        ):
            continue

        if (
            not is_justifiable_slot(vote.finalized_slot, vote.target.checkpoint_slot)
            or vote.target.hash != state.historical_block_hashes[vote.target.chain_slot]
        ):
            continue

        # Track attempts to justify new hashes
        justification_key = vote.source.hash + vote.target.hash
        if justification_key not in state.justifications:
            state.justifications[justification_key] = [False] * state.config.num_validators

        if not state.justifications[justification_key][vote.validator_id]:
            state.justifications[justification_key][vote.validator_id] = True

        count = sum(state.justifications[justification_key])

        # If 2/3 voted for the same new valid hash to justify
        if count == (2 * state.config.num_validators) // 3:
            state.latest_justified = vote.target
            state.justified_slots[vote.target.checkpoint_slot] = True
            del state.justifications[justification_key]

            if vote.source.checkpoint_slot + 1 == vote.target.checkpoint_slot:
                state.latest_finalized = vote.source

    return state

# Get the highest-slot justified block that we know about
def get_latest_justified_checkpoint(post_states: Dict[str, State]) -> Checkpoint:
    latest = max(   
        post_states.values(),
        key=lambda s: s.latest_justified.checkpoint_slot
    )
    return latest.latest_justified

def get_fork_choice_head(blocks: Dict[str, Block],
        root: str,
        fast_votes: List[FastVote],
        latest_slow_votes: List[SlowVote],
        min_score: int = 0) -> str:
    majority_fc_output = majority_fork_choice(blocks, root, latest_slow_votes)
    return ghost_fork_choice(blocks, majority_fc_output, fast_votes, require_relative_majority=False, min_score=min_score)

def majority_fork_choice(blocks: Dict[str, Block],
        root: str,
        latest_slow_votes: List[SlowVote],
        min_score: int = 0) -> str:
    # Start at genesis by default
    if root == ZERO_HASH:
        root = min(blocks.keys(), key=lambda block: blocks[block].slot)
    return ghost_fork_choice(blocks, root, latest_slow_votes, require_relative_majority=True, min_score=min_score)


def ghost_fork_choice(blocks: Dict[str, Block],
        root: str,
        votes: List[FastVote | SlowVote],
        require_relative_majority: bool,
        min_score: int = 0) -> str:

    # For each block, count the number of votes for that block. A vote
    # for any descendant of a block also counts as a vote for that block
    vote_weights: Dict[str, int] = {}

    total_weight = 0
    for vote in votes:
        total_weight += 1
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
        if require_relative_majority:
            children = [child for child in children if vote_weights.get(child, 0) * 2 > total_weight]
        if not children:
            return current
        current = max(children,
                      key=lambda x: (vote_weights.get(x, 0), blocks[x].slot, x))