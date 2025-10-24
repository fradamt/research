from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import hashlib
import json
import copy

ZERO_HASH = '0'*64
MAX_BACKOFF_INTERVAL_EXPONENT = 4
SLOTS_PER_EPOCH = 8
SLOW_VOTE_EXPIRATION_SLOTS = 128

# Chain configuration
@dataclass
class Config:
    num_validators: int
@dataclass(frozen=True)
class Checkpoint:
    hash: str
    slot: int
    epoch: int

# Blockchain state
@dataclass
class State:
    config: Config
    latest_justified: Checkpoint
    latest_finalized: Checkpoint
    historical_block_hashes: List[str] = field(default_factory=list)
    justified_checkpoints: List[Checkpoint] = field(default_factory=list)
    justifications: Dict[str, List[bool]] = field(default_factory=dict)

@dataclass(frozen=True)
class FastVote:
    validator_id: int
    slot: int
    head: str

@dataclass(frozen=True)
class SlowVote:
    validator_id: int
    finalized_epoch: int
    source: Checkpoint
    target: Checkpoint

@dataclass
class GHOSTVote:
    validator_id: int
    head: str

# A block
@dataclass
class Block:
    slot: int
    parent: Optional[str]
    fast_votes: List[FastVote] = field(default_factory=list)
    slow_votes: List[SlowVote] = field(default_factory=list)
    state_root: Optional[str] = None

# Stub for computing block hash, state root...
# (in real life replace with SSZ hashing)
def compute_hash(obj: object):
    if isinstance(obj, tuple):
        serialized = json.dumps([asdict(item) if hasattr(item, '__dataclass_fields__') else item for item in obj], sort_keys=True).encode()
    else:
        serialized = json.dumps(asdict(obj), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()

def slot_to_epoch(slot: int) -> int:
    return slot // SLOTS_PER_EPOCH

# Determines if slow voting should take place in a given epoch, based on an exponential backoff mechanism (with a cap).
# A backoff interval is calculated based on the distance from the last finalized epoch, and slow voting should
# only take place if the epoch is a multiple of the interval. Slow votes carry a `finalized_epoch`, and are invalid
# if they violate this rule. The mechanism helps finality progress under high latency.
def is_slow_voting_epoch(finalized_epoch: int, target_epoch: int):
    max_backoff_interval = 2**MAX_BACKOFF_INTERVAL_EXPONENT
    delta = (target_epoch - finalized_epoch) * (MAX_BACKOFF_INTERVAL_EXPONENT - 1)
    delta = delta // (2 * max_backoff_interval)
    backoff_interval = min(2**delta, max_backoff_interval)
    return target_epoch % backoff_interval == 0

# Given a state, output the new state after processing that block
def process_block(state: State, block: Block) -> State:
    state = copy.deepcopy(state)
    # Track historical blocks in the state
    state.historical_block_hashes.append(block.parent)
    while len(state.historical_block_hashes) < block.slot:
        state.historical_block_hashes.append(None)
    # Process votes
    for vote in block.slow_votes:

        if (
            not is_slow_voting_epoch(vote.finalized_epoch, vote.target.epoch)
            or vote.source.epoch < state.latest_finalized.epoch
            or vote.source not in state.justified_checkpoints
            or vote.target in state.justified_checkpoints
            or vote.target.hash != state.historical_block_hashes[vote.target.slot]
            or vote.target.slot < vote.source.slot
            or vote.target.epoch <= vote.source.epoch
        ):
            continue

        # Track attempts to justify new hashes
        justification_key = compute_hash((vote.finalized_epoch, vote.source, vote.target))
        if justification_key not in state.justifications:
            state.justifications[justification_key] = [False] * state.config.num_validators

        if not state.justifications[justification_key][vote.validator_id]:
            state.justifications[justification_key][vote.validator_id] = True

        count = sum(state.justifications[justification_key])

        # If 2/3 voted for the same new valid hash to justify
        if count == (2 * state.config.num_validators) // 3:
            state.latest_justified = vote.target
            state.justified_checkpoints.append(vote.target)
            del state.justifications[justification_key]


            # Finalization: if the target is the next valid slow voting
            # epoch after the source, wrt the finalized epoch in the votes.
            if not any(
                is_slow_voting_epoch(vote.finalized_epoch, epoch)
                for epoch in range(vote.source.epoch + 1, vote.target.epoch)
            ):
                state.latest_finalized = vote.source
                # Prune old checkpoints
                state.justified_checkpoints = [
                    checkpoint for checkpoint in state.justified_checkpoints 
                    if checkpoint.epoch >= state.latest_finalized.epoch
                ]

    return state

# Get the highest-slot justified block that we know about
def get_latest_justified_checkpoint(post_states: Dict[str, State]) -> Checkpoint:
    latest = max(   
        post_states.values(),
        key=lambda s: s.latest_justified.epoch
    )
    return latest.latest_justified

def get_fork_choice_head(blocks: Dict[str, Block],
        slot: int,
        root: str,
        fast_votes: List[FastVote],
        latest_slow_votes: List[SlowVote],
        min_score: int = 0) -> str:
    majority_fc_output = majority_fork_choice(blocks, slot, root, latest_slow_votes)
    ghost_votes = [GHOSTVote(validator_id=vote.validator_id, head=vote.head) for vote in fast_votes]
    return ghost_fork_choice(blocks, majority_fc_output, ghost_votes, require_relative_majority=False, min_score=min_score)

def majority_fork_choice(blocks: Dict[str, Block],
        slot: int,
        root: str,
        latest_slow_votes: List[SlowVote]) -> str:
    # Start at genesis by default
    if root == ZERO_HASH:
        root = min(blocks.keys(), key=lambda block: blocks[block].slot)
    last_unexpired_epoch = slot_to_epoch(slot) - (SLOW_VOTE_EXPIRATION_SLOTS // SLOTS_PER_EPOCH)
    ghost_votes = [
        GHOSTVote(validator_id=vote.validator_id, head=vote.target.hash)
        for vote in latest_slow_votes
        if vote.target.epoch > last_unexpired_epoch
    ]
    return ghost_fork_choice(blocks, root, ghost_votes, require_relative_majority=True)


def ghost_fork_choice(blocks: Dict[str, Block],
        root: str,
        votes: List[GHOSTVote],
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