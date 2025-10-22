import random
import heapq
from typing import List, Dict, Union, Set, Tuple
import copy
from consensus import (
    State, SlowVote, FastVote, Block, Checkpoint, majority_fork_choice,
    process_block, get_latest_justified_checkpoint, get_fork_choice_head,
    compute_hash, is_slow_voting_slot
)
from collections import defaultdict

SLOT_DURATION = 12  # time units
ZERO_HASH = '0'*64
KAPPA = 32

# A basic Staker node implementation
class Staker:
    def __init__(self, validator_id: int, network: 'P2PNetwork', genesis_block: Block, genesis_state: State, use_backoff: bool = True):
        # This node's validator ID
        self.validator_id = validator_id
        # Hook to the p2p network
        self.network = network
        # {block hash: block} for all blocks that we know about
        self.chain: Dict[str, Block] = {}
        # {block hash: post state} for all blocks that we know about
        self.post_states: Dict[str, State] = {}
        self.slow_votes: Set[SlowVote] = set()
        # Latest unexpired slow votes of each validator
        self.latest_slow_votes: Dict[int, SlowVote] = {}
        # Accepted fast votes
        self.fast_votes: Set[FastVote] = set()
        # Fast votes that have not been view-merged yet
        self.fast_votes_buffer: Set[FastVote] = set()
        # Current valid slot for fast votes
        self.current_fast_vote_slot: int = 0
        # Objects that we will process once we have processed their parents
        self.dependencies: Dict[str, List[Block]] = {}
        # Initialize the chain with the genesis block
        self.genesis_hash = compute_hash(genesis_block)
        self.chain[self.genesis_hash] = genesis_block
        self.post_states[self.genesis_hash] = genesis_state
        self.num_validators = genesis_state.config.num_validators
        # Block that it is safe to use to vote as the target
        self.confirmed_hash: str = self.genesis_hash
        # Head of the chain
        self.head = self.genesis_hash
        # Whether to use k-th ancestor backoff
        self.use_backoff = use_backoff
        # Join the p2p network
        self.network.register_staker(self)

    @property
    def latest_justified(self):
        return get_latest_justified_checkpoint(self.post_states)

    @property
    def latest_finalized(self):
        latest = max(   
            self.post_states.values(),
            key=lambda s: s.latest_finalized.checkpoint_slot
        )
        return latest.latest_finalized

    def get_current_slot(self):
        return self.network.time // SLOT_DURATION + 2

    # Called every second
    def tick(self):
        time_in_slot = (self.network.time % SLOT_DURATION)
        # t=0: propose a block
        if time_in_slot == 0:
            if self.get_current_slot() % self.num_validators == self.validator_id:
                self.merge_fast_votes()
                self.propose_block()
        # t=1/4: vote
        elif time_in_slot == SLOT_DURATION // 4:
            self.fast_vote()
        # t=2/4: compute the safe target (this must be done here to ensure
        # that, assuming network latency assumptions are satisfied, anything that
        # one honest node receives by this time, every honest node will receive by
        # the general attestation deadline)
        elif time_in_slot == (SLOT_DURATION * 2) // 4:
            self.merge_fast_votes()
            self.confirm()
            self.slow_vote()
        # Deadline to accept attestations except for those included in a block
        elif time_in_slot == (SLOT_DURATION * 3) // 4:
            self.merge_fast_votes()

    # Called when it's the staker's turn to propose a block
    def propose_block(self):
        new_slot = self.get_current_slot()
        head_state = self.post_states[self.head]
        # naively just adds all votes
        new_block = Block(
                slot=new_slot,
                parent=self.head,
                slow_votes=list(self.slow_votes),
                fast_votes=list(self.fast_votes),
            )
        state = process_block(head_state, new_block)
        new_block.state_root = compute_hash(state)
        new_hash = compute_hash(new_block)

        self.chain[new_hash] = new_block
        self.post_states[new_hash] = state
        self.network.submit(new_block, self.validator_id)


    # Done upon processing new votes or a new block
    def recompute_head(self):
        root = self.latest_justified.hash
        self.head = get_fork_choice_head(self.chain, root, self.fast_votes, self.latest_slow_votes.values())

    # Process new votes tha the staker has received. Vote processing is done
    # at a particular time, because of view-merge rules
    def merge_fast_votes(self):
        self.fast_votes.update(self.fast_votes_buffer)
        self.fast_votes_buffer = set()
        self.recompute_head()

    # Called when it's the staker's turn to vote
    def fast_vote(self):
        slot = self.get_current_slot()
        vote = FastVote(
            validator_id=self.validator_id,
            slot=slot,
            head=self.head,
        )
        
        self.fast_votes = set()
        self.fast_votes_buffer = set()
        self.current_fast_vote_slot = slot
        self.receive(vote)
        self.network.submit(vote, self.validator_id)

        # Called when it's the staker's turn to vote
    def slow_vote(self):
        if not is_slow_voting_slot(self.latest_finalized.checkpoint_slot, self.get_current_slot()):
            return
        
        vote =  SlowVote(
            validator_id=self.validator_id,
            finalized_slot=self.latest_finalized.checkpoint_slot,
            source=self.latest_justified,
            target=self.get_target()
        )
        
        self.receive(vote)
        self.network.submit(vote, self.validator_id)

    def confirm(self):
        self.recompute_head()
        fast_confirmed_hash = get_fork_choice_head(
            self.chain,
            self.latest_justified.hash,
            self.fast_votes,
            self.latest_slow_votes.values(),
            min_score=self.num_validators * 3 // 4
        )
        fast_confirmed_block = self.chain[fast_confirmed_hash]
        if fast_confirmed_block.slot >= self.get_current_slot() - KAPPA:
            self.confirmed_hash = fast_confirmed_hash
        else:
            kappa_deep_slot = self.get_current_slot() - KAPPA
            self.confirmed_hash = compute_hash(self.get_block_at_slot(kappa_deep_slot))

    def get_block_at_slot(self, slot: int):
        if slot <= self.chain[self.genesis_hash].slot:
            return self.chain[self.genesis_hash]
        current_block = self.chain[self.head]
        while current_block.slot > slot:
            current_block = self.chain[current_block.parent]
        return current_block

    def get_target(self):
        if self.latest_justified.checkpoint_slot + 1 == self.get_current_slot():
            target_block = self.chain[self.confirmed_hash]
        else:
            majority_hash = majority_fork_choice(
                self.chain,
                self.latest_justified.hash,
                self.latest_slow_votes.values()
            )
            majority_block = self.chain[majority_hash]
            kappa_deep_slot = self.get_current_slot() - KAPPA
            target_block = self.get_block_at_slot(max(majority_block.slot, kappa_deep_slot))

        return Checkpoint(
            hash=compute_hash(target_block),
            chain_slot=target_block.slot,
            checkpoint_slot=self.get_current_slot()
        )
    
    # Called by the p2p network
    def receive(self, item: Union[Block, FastVote, SlowVote]):
        if isinstance(item, Block):
            block_hash = compute_hash(item)
            # If the block is already known, ignore it
            if block_hash in self.chain:
                return
            parent_state = self.post_states.get(item.parent)
            if parent_state:
                state = process_block(copy.deepcopy(parent_state), item)
                self.chain[block_hash] = item
                self.post_states[block_hash] = state
                # Receive fast votes if the block is timely and from the current slot
                time_in_slot = (self.network.time % SLOT_DURATION)
                timely_block = item.slot == self.get_current_slot() and time_in_slot <= SLOT_DURATION // 4
                if timely_block and all(vote.slot == self.current_fast_vote_slot for vote in item.fast_votes):
                    self.fast_votes.update(item.fast_votes)
                    self.recompute_head()
                # Receive slow votes
                for vote in item.slow_votes:
                    self.receive(vote)
                # Once we have received a block, also process all of
                # its dependencies
                if block_hash in self.dependencies:
                    for item2 in self.dependencies[block_hash]:
                        self.receive(item2)
                    del self.dependencies[block_hash]
            else:
                # If we have not yet seen the block's parent, ignore for now,
                # process later once we actually see the parent
                self.dependencies.setdefault(item.parent, []).append(item)
        elif isinstance(item, SlowVote):
            if item.target.hash in self.chain:
                self.slow_votes.add(item)
                if (
                    item.validator_id not in self.latest_slow_votes
                    or item.target.checkpoint_slot > self.latest_slow_votes[item.validator_id].target.checkpoint_slot
                ):
                    self.latest_slow_votes[item.validator_id] = item
            else:
                self.dependencies.setdefault(item.target.hash, []).append(item)
        elif isinstance(item, FastVote):
            if item.slot == self.current_fast_vote_slot:
                if item.head in self.chain:
                    self.fast_votes_buffer.add(item)
                else:
                    self.dependencies.setdefault(item.head, []).append(item)

# Simulates a p2p network
class P2PNetwork:
    def __init__(self, latency_func):
        self.time = 0
        self.stakers: Dict[int, Staker] = {}
        self.queues: Dict[int, List[Tuple[int, Union[Block, FastVote, SlowVote]]]] = defaultdict(list)
        self.latency_func = latency_func

    def register_staker(self, staker: Staker):
        self.stakers[staker.validator_id] = staker

    def submit(self, item: Union[Block, FastVote, SlowVote], sender_id: int):
        for recipient_id, _ in self.stakers.items():
            if recipient_id == sender_id:
                continue
            deliver_at = self.time + self.latency_func(self.time)
            self.queues[recipient_id].append((deliver_at, item))

    def time_step(self):
        self.time += 1
        for validator_id, queue in self.queues.items():
            deliver_now = [item for (t, item) in queue if t <= self.time]
            self.queues[validator_id] = [(t, item) for (t, item) in queue if t > self.time]
            for item in deliver_now:
                self.stakers[validator_id].receive(item)
