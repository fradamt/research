from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import hashlib
import json
import copy

ZERO_HASH = '0'*64
MAX_BACKOFF_INTERVAL_EXPONENT = 4
EPOCHS_FOR_LONG_EXPIRATION_PERIOD = 64
EPOCHS_FOR_SHORT_EXPIRATION_PERIOD = 2
FINALIZATION_THRESHOLD_NUMERATOR = 5
FINALIZATION_THRESHOLD_DENOMINATOR = 6
JUSTIFICATION_THRESHOLD_NUMERATOR = 1
JUSTIFICATION_THRESHOLD_DENOMINATOR = 2
SKIP_THRESHOLD_NUMERATOR = 1
SKIP_THRESHOLD_DENOMINATOR = 3


# Chain configuration
@dataclass
class Config:
    num_validators: int
    slots_per_epoch: int = 4
@dataclass(frozen=True)
class Checkpoint:
    hash: str
    slot: int
    height: int

# Blockchain state
@dataclass
class State:
    config: Config
    latest_finalized: Checkpoint
    latest_justified: Checkpoint
    height: int
    historical_block_hashes: List[str] = field(default_factory=list)
    height_to_target_hash: Dict[int, List[Optional[str]]] = field(default_factory=dict)
    has_equivocated: Dict[int, List[bool]] = field(default_factory=dict)

@dataclass(frozen=True)
class FastVote:
    validator_id: int
    slot: int
    head: str

@dataclass(frozen=True)
class BeaconVote(FastVote):
    pass

@dataclass(frozen=True)
class PayloadVote(FastVote):
    payload_available: bool

@dataclass(frozen=True)
class SlowVote:
    validator_id: int
    epoch: int
    target: Optional[Checkpoint]
    confirmed: str
@dataclass
class GHOSTVote:
    validator_id: int
    head: str

# A block
@dataclass
class Block:
    slot: int
    parent: Optional[str]
    payload_votes: List[PayloadVote] = field(default_factory=list)
    slow_votes: List[SlowVote] = field(default_factory=list)
    state_root: Optional[str] = None
    parent_has_payload: bool = False

# Stub for computing block hash, state root...
# (in real life replace with SSZ hashing)
def compute_hash(obj: object):
    if isinstance(obj, tuple):
        serialized = json.dumps([asdict(item) if hasattr(item, '__dataclass_fields__') else item for item in obj], sort_keys=True).encode()
    else:
        serialized = json.dumps(asdict(obj), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()

def slot_to_epoch(slot: int, config: Config) -> int:
    return slot // config.slots_per_epoch

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
            vote.target is None
            or vote.target.height <= state.latest_finalized.height
            or vote.target.height > state.height
        ):
            continue

        height = vote.target.height
        # Track votes for non-finalized heights
        if height not in state.height_to_target_hash:
            state.height_to_target_hash[height] = [None] * state.config.num_validators
            state.has_equivocated[height] = [False] * state.config.num_validators

        # Skip if already equivocated at this height
        if state.has_equivocated[height][vote.validator_id]:
            continue

        prior = state.height_to_target_hash[height][vote.validator_id]
        if prior is None:
            state.height_to_target_hash[height][vote.validator_id] = vote.target.hash
        elif prior == vote.target.hash:
            continue
        elif prior != vote.target.hash:
            # Equivocation: zero out vote and mark
            state.height_to_target_hash[height][vote.validator_id] = None
            state.has_equivocated[height][vote.validator_id] = True

        # Number of votes for this target hash at this height
        count_for_target = len(
            [
                h
                for h in state.height_to_target_hash[height]
                if h == vote.target.hash
            ]
        )
        equiv_count = sum(state.has_equivocated[height])
        count_for_target += equiv_count

        # Justify if 1/2 voted for a checkpoint
        is_known_target =  vote.target.hash == state.historical_block_hashes[vote.target.slot]
        justification_threshold = JUSTIFICATION_THRESHOLD_NUMERATOR * (state.config.num_validators) // JUSTIFICATION_THRESHOLD_DENOMINATOR
        justification = is_known_target and count_for_target >= justification_threshold
        if justification and state.latest_justified.height < height:
            state.latest_justified = vote.target

        # Move to next height if there's a justification or a skip (allVotes - maxVotes >= 1/3)
        if state.height == height:
            hashes = [h for h in state.height_to_target_hash[height] if h is not None]
            max_count = max((hashes.count(h) for h in hashes), default=0)
            total_count = len(hashes) + equiv_count
            skip_threshold = SKIP_THRESHOLD_NUMERATOR * (state.config.num_validators) // SKIP_THRESHOLD_DENOMINATOR
            skip = total_count - max_count >= skip_threshold
            if justification or skip:
                state.height += 1
            
        # Finalize if 5/6 voted for a checkpoint
        finalization_threshold = FINALIZATION_THRESHOLD_NUMERATOR * (state.config.num_validators) // FINALIZATION_THRESHOLD_DENOMINATOR
        if count_for_target >= finalization_threshold:
            state.latest_finalized = vote.target
            # Clear vote tracking for this height once finalized
            del state.height_to_target_hash[height]
            del state.has_equivocated[height]

    return state

def get_fork_choice_head(blocks: Dict[str, Block],
        root: str,
        fast_votes: List[FastVote],
        slow_votes: List[SlowVote],
        min_score: int = 0) -> str:
    root = majority_fork_choice(blocks, root, slow_votes)
    ghost_votes = [GHOSTVote(validator_id=vote.validator_id, head=vote.head) for vote in fast_votes]
    return ghost_fork_choice(blocks, root, ghost_votes, min_score=min_score)

def majority_fork_choice(blocks: Dict[str, Block],
        root: str,
        slow_votes: List[SlowVote]) -> str:
    # Start at genesis by default
    if root == ZERO_HASH:
        root = min(blocks.keys(), key=lambda block: blocks[block].slot)
    if len(slow_votes) == 0:
        return root

    max_epoch = max(vote.epoch for vote in slow_votes)
    long_expiration_ghost_votes = [
        GHOSTVote(validator_id=vote.validator_id, head=vote.confirmed)
        for vote in slow_votes
        if vote.epoch >= max_epoch - EPOCHS_FOR_LONG_EXPIRATION_PERIOD
    ]
    majority_threshold = (len(long_expiration_ghost_votes)+1) // 2
    root = ghost_fork_choice(blocks, root, long_expiration_ghost_votes, min_score=majority_threshold + 1)

    short_expiration_ghost_votes = [
        GHOSTVote(validator_id=vote.validator_id, head=vote.confirmed)
        for vote in slow_votes
        if vote.epoch >= max_epoch - EPOCHS_FOR_SHORT_EXPIRATION_PERIOD
    ]
    majority_threshold = (len(short_expiration_ghost_votes)+1) // 2
    return ghost_fork_choice(blocks, root, short_expiration_ghost_votes, min_score=majority_threshold + 1)


def ghost_fork_choice(blocks: Dict[str, Block],
        root: str,
        votes: List[GHOSTVote],
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
        if not children:
            return current
        current = max(children,
                      key=lambda x: (vote_weights.get(x, 0), blocks[x].slot, x))
