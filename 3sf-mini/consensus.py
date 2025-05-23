from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import hashlib
import json
import copy

ZERO_HASH = '0'*64

# Chain configuration
@dataclass
class Config:
    num_validators: int
    max_checkpoint_interval_for_backoff: int

@dataclass
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

# A vote. In a live implementation this would also include a signature
@dataclass
class Vote:
    validator_id: int
    slot: int
    head: str
    source: Optional[Checkpoint] = None
    target: Optional[Checkpoint] = None

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
    serialized = json.dumps(asdict(obj), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()

# We allow justification of slots either <= 5 or a perfect square or oblong after
# the latest finalized slot. This gives us a backoff technique and ensures
# finality keeps progressing even under high latency
def is_justifiable_slot(config: Config, finalized_slot: int, candidate: int):
    assert candidate >= finalized_slot
    delta = candidate - finalized_slot
    checkpoint_interval = min(2**(delta // 8), config.max_checkpoint_interval_for_backoff)
    return candidate % checkpoint_interval == 0

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
    for vote in block.votes:
        # Ignore votes without a source or target, or with source later than
        # the latest finalized slot or not already justified, or whose target
        # or source is not in the history, or whose target is not a
        # valid justifiable slot
        if (
            vote.source is None or vote.target is None
            or vote.source.checkpoint_slot < state.latest_finalized.checkpoint_slot
            or state.justified_slots[vote.source.checkpoint_slot] is False
            or vote.source.hash != state.historical_block_hashes[vote.source.chain_slot]
            or vote.target.hash != state.historical_block_hashes[vote.target.chain_slot]
            or vote.target.checkpoint_slot <= vote.source.checkpoint_slot
            or not is_justifiable_slot(state.config, state.latest_finalized.checkpoint_slot, vote.target.checkpoint_slot)
        ):
            continue

        # Track attempts to justify new hashes
        if vote.target.hash not in state.justifications:
            state.justifications[vote.target.hash] = [False] * state.config.num_validators

        if not state.justifications[vote.target.hash][vote.validator_id]:
            state.justifications[vote.target.hash][vote.validator_id] = True

        count = sum(state.justifications[vote.target.hash])

        # If 2/3 voted for the same new valid hash to justify
        if count == (2 * state.config.num_validators) // 3:
            state.latest_justified = vote.target
            state.justified_slots[vote.target.checkpoint_slot] = True
            del state.justifications[vote.target.hash]

            # Finalization: if the target is the next valid justifiable
            # hash after the source
            if not any(
                is_justifiable_slot(state.config, state.latest_finalized.checkpoint_slot, slot)
                for slot in range(vote.source.checkpoint_slot + 1, vote.target.checkpoint_slot)
            ):
                state.latest_finalized = vote.source

    return state

# Get the highest-slot justified block that we know about
def get_latest_justified_checkpoint(post_states: Dict[str, State]) -> Checkpoint:
    latest = max(   
        post_states.values(),
        key=lambda s: s.latest_justified.checkpoint_slot
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
    for vote in sorted(votes, key=lambda vote: vote.slot):
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
