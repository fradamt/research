from calendar import c
import random
import heapq
from re import A
from typing import List, Dict, Union, Set, Tuple
import copy
from consensus import (
    State, SlowVote, FastVote, BeaconVote, PayloadVote, Block, Checkpoint, majority_fork_choice,
    process_block, get_latest_justified_checkpoint, get_fork_choice_head,
    compute_hash, is_slow_voting_epoch, slot_to_epoch
)
from collections import defaultdict

SLOT_DURATION = 10  # time units
ZERO_HASH = '0'*64
KAPPA = 32
ConsensusObject = Union[Block, BeaconVote, PayloadVote, SlowVote]


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
        # Initialize the chain with the genesis block
        self.genesis_hash = compute_hash(genesis_block)
        self.chain[self.genesis_hash] = genesis_block
        self.post_states[self.genesis_hash] = genesis_state
        self.config = genesis_state.config
        # Store all slow votes (keyed by validator_id, epoch)
        self.slow_votes: Dict[Tuple[int, int], SlowVote] = {}
        # Latest slow vote for each validator
        self.latest_slow_votes: Dict[int, SlowVote] = {}
        # Boolean vector tracking which validators sent timely beacon votes (by 2*slot_fifth)
        self.timely_beacon_voters: List[bool] = [False] * self.config.num_validators
        # First-seen beacon vote of each validator for the current slot
        self.beacon_votes: Dict[int, BeaconVote] = {}
        # Beacon votes equivocation: second-seen vote per validator when equivocation detected
        self.beacon_vote_equivocations: Dict[int, BeaconVote] = {}
        # First-seen payload vote of each validator for the current slot
        self.payload_votes: Dict[int, PayloadVote] = {}
        # Payload votes equivocation: second-seen vote per validator when equivocation detected
        self.payload_vote_equivocations: Dict[int, PayloadVote] = {}
        # Persistent equivocators (for slow votes)
        self.equivocators: Set[int] = set()
        # Objects that we will process once we have processed their parents
        self.dependencies: Dict[str, List[Block]] = {}
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
            key=lambda s: s.latest_finalized.epoch
        )
        return latest.latest_finalized

    def get_current_slot(self):
        return self.network.time // SLOT_DURATION + 2

    def get_current_epoch(self):
        return slot_to_epoch(self.get_current_slot(), self.config)


    # Called every second
    def tick(self):
        time_in_slot = (self.network.time % SLOT_DURATION)
        slot_fifth = SLOT_DURATION // 5
        
        if time_in_slot == 0:
            self.beacon_vote()
        elif time_in_slot == slot_fifth:
            self.fast_confirm()
            if self.should_slow_vote():
                self.slow_vote()
        elif time_in_slot == slot_fifth * 2:
            self.payload_vote()
        elif time_in_slot == slot_fifth * 3:
            self.available_confirm()
        elif time_in_slot == slot_fifth * 4:
            if self.is_proposer():
                self.propose_block()


    def is_proposer(self):
        return self.get_current_slot() % self.config.num_validators == self.validator_id

    # Called when it's the staker's turn to propose a block
    def propose_block(self):
        slot = self.get_current_slot()
        root = self.latest_justified.hash
        self.head = get_fork_choice_head(
            blocks=self.chain,
            slot=slot,
            root=root,
            fast_votes=self.get_payload_votes_for_fork_choice(),
            latest_slow_votes=self.latest_slow_votes.values(),
        )
        head_state = self.post_states[self.head]
        finalized_epoch = head_state.latest_finalized.epoch
        slow_votes_to_include = [
                vote for (_, epoch), vote in self.slow_votes.items()
                if epoch > finalized_epoch
            ]
        # Include all payload votes (both first-seen and equivocating)
        payload_votes_to_include = (
            list[PayloadVote](self.payload_votes.values())
            + list[PayloadVote](self.payload_vote_equivocations.values())
        )
        new_block = Block(
                slot=slot,
                parent=self.head,
                slow_votes=slow_votes_to_include,
                payload_votes=payload_votes_to_include,
            )
        state = process_block(head_state, new_block)
        new_block.state_root = compute_hash(state)
        new_hash = compute_hash(new_block)

        self.chain[new_hash] = new_block
        self.post_states[new_hash] = state
        self.network.submit(new_block, self.validator_id)

    def beacon_vote(self):
        slot = self.get_current_slot()
        root = self.latest_justified.hash
        self.head = get_fork_choice_head(
            blocks=self.chain,
            slot=slot,
            root=root,
            fast_votes=self.get_payload_votes_for_fork_choice(),
            latest_slow_votes=self.latest_slow_votes.values()
        )
        vote = BeaconVote(
            validator_id=self.validator_id,
            slot=slot,
            head=self.head,
        )
        
        # Clear payload vote trackers
        self.payload_votes.clear()
        self.payload_vote_equivocations.clear()
        self.receive(vote)
        self.network.submit(vote, self.validator_id)


    def slow_vote(self):
        vote =  SlowVote(
            validator_id=self.validator_id,
            finalized_epoch=self.latest_finalized.epoch,
            source=self.latest_justified,
            target=self.get_target_checkpoint()
        )
        
        self.receive(vote)
        self.network.submit(vote, self.validator_id)

    def fast_confirm(self):
        slot = self.get_current_slot()
        root = self.latest_justified.hash
        beacon_votes = self.get_beacon_votes_for_fork_choice()
        latest_slow_votes = self.latest_slow_votes.values()

        self.head = get_fork_choice_head(
            blocks=self.chain,
            slot=slot,
            root=root,
            fast_votes=beacon_votes,
            latest_slow_votes=latest_slow_votes,
        )
        fast_confirmed_hash = get_fork_choice_head(
            self.chain,
            slot,
            root,
            fast_votes=beacon_votes,
            latest_slow_votes=latest_slow_votes,
            min_score=self.config.num_validators * 3 // 4
        )
        fast_confirmed_block = self.chain[fast_confirmed_hash]
        if fast_confirmed_block.slot >= self.get_current_slot() - KAPPA:
            self.confirmed_hash = fast_confirmed_hash
        else:
            kappa_deep_slot = self.get_current_slot() - KAPPA
            self.confirmed_hash = compute_hash(self.get_block_at_slot(kappa_deep_slot))

    def payload_vote(self):
        slot = self.get_current_slot()
        beacon_votes = self.get_beacon_votes_for_fork_choice()
        equivocations = len(self.beacon_vote_equivocations)
        # Majority threshold including all received votes (equivocations as well)
        total_votes = len(beacon_votes) + equivocations
        majority_threshold = (total_votes+1) // 2
        # weight >= min_score => weight + equivocations > majority_threshold
        # Then we can consider the block timely: at least one honest node voted for it
        min_score = max(0, majority_threshold - equivocations + 1)
        self.head = get_fork_choice_head(
            blocks=self.chain,
            slot=slot,
            root=self.latest_justified.hash,
            fast_votes=beacon_votes,
            latest_slow_votes=self.latest_slow_votes.values(),
            min_score=min_score,
        )
        vote = PayloadVote(
            validator_id=self.validator_id,
            slot=slot,
            head=self.head,
        )
        
        self.receive(vote)
        self.network.submit(vote, self.validator_id)


    def available_confirm(self):
        if self.confirmed_hash != self.head:
            beacon_votes = self.get_beacon_votes_for_fork_choice()
            timely_beacon_votes = [vote for vote in beacon_votes if vote.validator_id in self.timely_beacon_voters]
            total_votes = len(beacon_votes) + len(self.beacon_vote_equivocations)
            majority_threshold = (total_votes+1) // 2
            new_confirmed_hash = get_fork_choice_head(
                blocks=self.chain,
                slot=self.get_current_slot(),
                root=self.latest_justified.hash,
                latest_slow_votes=self.latest_slow_votes.values(),
                fast_votes=timely_beacon_votes,
                min_score=majority_threshold + 1,
            )

            new_confirmed_block = self.chain[new_confirmed_hash]
            confirmed_block = self.chain[self.confirmed_hash]
            if new_confirmed_block.slot > confirmed_block.slot:
                self.confirmed_hash = new_confirmed_hash

        # Clear beacon vote trackers
        self.timely_beacon_voters = [False] * self.config.num_validators
        self.beacon_votes.clear()
        self.beacon_vote_equivocations.clear()

    # Get beacon votes to use in fork choice (including equivocating votes)
    def get_beacon_votes_for_fork_choice(self):
        """Returns beacon votes excluding those from equivocating validators."""
        return [vote for vid, vote in self.beacon_votes.items() if vid not in self.beacon_vote_equivocations]

    # Get payload votes to use in fork choice (including equivocating votes)
    def get_payload_votes_for_fork_choice(self):
        """Returns payload votes excluding those from equivocating validators."""
        return [vote for vid, vote in self.payload_votes.items() if vid not in self.payload_vote_equivocations]
        
    def should_slow_vote(self):
        first_slot_of_epoch = self.get_current_slot() % self.config.slots_per_epoch == 0
        slow_voting_epoch = is_slow_voting_epoch(self.latest_finalized.epoch, self.get_current_epoch())
        return first_slot_of_epoch and slow_voting_epoch

    def get_block_at_slot(self, slot: int):
        if slot <= self.chain[self.genesis_hash].slot:
            return self.chain[self.genesis_hash]
        current_block = self.chain[self.head]
        while current_block.slot > slot:
            current_block = self.chain[current_block.parent]
        return current_block

    def get_target_checkpoint(self):
        if self.latest_justified.epoch + 1 == self.get_current_epoch():
            target_block = self.chain[self.confirmed_hash]
        else:
            majority_hash = majority_fork_choice(
                self.chain,
                self.get_current_epoch(),
                self.latest_justified.hash,
                self.latest_slow_votes.values()
            )
            majority_block = self.chain[majority_hash]
            kappa_deep_slot = self.get_current_slot() - KAPPA
            target_block = self.get_block_at_slot(max(majority_block.slot, kappa_deep_slot))

        return Checkpoint(
            hash=compute_hash(target_block),
            slot=target_block.slot,
            epoch=self.get_current_epoch(),
        )
    
    # Called by the p2p network
    def receive(self, item: ConsensusObject):
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
                # Receive payload votes if the block is timely
                timely_block = item.slot == self.get_current_slot()
                if timely_block:
                    # Receive payload votes from block (always overwrite existing)
                    for vote in item.payload_votes:
                        self.receive_payload_vote(vote, from_block=True)
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
            if item.validator_id in self.equivocators:
                return
            if item.target.hash in self.chain:
                vote_epoch = item.target.epoch
                vote_key = (item.validator_id, vote_epoch)
                
                # Already have a vote for this epoch, either already seen or an equivocation
                if vote_key in self.slow_votes:
                    # Check for equivocation: same validator making 
                    # two different votes for same epoch
                    if self.slow_votes[vote_key] != item:
                        self.handle_equivocation(item.validator_id)
                    return
                
                # Store the vote
                self.slow_votes[vote_key] = item
                
                # Update latest slow vote if this is a newer vote
                if (
                    item.validator_id not in self.latest_slow_votes
                    or vote_epoch > self.latest_slow_votes[item.validator_id].target.epoch
                ):
                    self.latest_slow_votes[item.validator_id] = item
            else:
                self.dependencies.setdefault(item.target.hash, []).append(item)
        elif isinstance(item, BeaconVote):
            self.receive_beacon_vote(item)
        elif isinstance(item, PayloadVote):
            self.receive_payload_vote(item, from_block=False)

    def receive_beacon_vote(self, vote: BeaconVote):
        # Only process votes for the current slot
        if vote.slot != self.get_current_slot():
            return

        # Ignore votes from known equivocators
        if vote.validator_id in self.beacon_vote_equivocations:
            return

        # Beacon votes are only accepted between slot_fifth and 3*slot_fifth
        time_in_slot = (self.network.time % SLOT_DURATION)
        slot_fifth = SLOT_DURATION // 5
        if not time_in_slot <= 3 * slot_fifth:
            return

        # Record the vote if it is the first-seen
        if vote.validator_id not in self.beacon_votes:
            self.beacon_votes[vote.validator_id] = vote
            if time_in_slot <= slot_fifth:
                self.timely_beacon_voters[vote.validator_id] = True
            return
        
        # If one has been seen, check for equivocation
        if self.beacon_votes[vote.validator_id] != vote:
            self.beacon_vote_equivocations[vote.validator_id] = vote
            self.handle_equivocation(vote.validator_id)

    def receive_payload_vote(self, vote: PayloadVote, from_block: bool = True):
        # Only process votes for the current slot
        if vote.slot != self.get_current_slot():
            return

        # Ignore votes from known equivocators
        if vote.validator_id in self.payload_vote_equivocations:
            return

        time_in_slot = (self.network.time % SLOT_DURATION)
        slot_fifth = SLOT_DURATION // 5
        # Ignore payload votes received before payload voting time (2*slot_fifth)
        if time_in_slot < 2*slot_fifth:
            return

        # View-merge mechanism:
        # Non-proposers ignore payload votes after
        # the view-merge deadline, unless from a block.
        if time_in_slot > 3 * slot_fifth:
            if not from_block and not self.is_proposer():
                return

        # Record the vote if it is the first-seen
        if vote.validator_id not in self.payload_votes:
            self.payload_votes[vote.validator_id] = vote
            return
        
        # If one has been seen, check for equivocation
        if self.payload_votes[vote.validator_id] != vote:
            self.payload_vote_equivocations[vote.validator_id] = vote
            self.handle_equivocation(vote.validator_id)

    def handle_equivocation(self, validator_id: int):
        self.equivocators.add(validator_id)
        self.latest_slow_votes.pop(validator_id, None)

# Simulates a p2p network
class P2PNetwork:
    def __init__(self, latency_func):
        self.time = 0
        self.stakers: Dict[int, Staker] = {}
        self.queues: Dict[int, List[Tuple[int, ConsensusObject]]] = defaultdict(list)
        self.latency_func = latency_func

    def register_staker(self, staker: Staker):
        self.stakers[staker.validator_id] = staker

    def submit(self, item: ConsensusObject, sender_id: int):
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
