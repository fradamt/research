from calendar import c
import random
import heapq
from re import A
from typing import List, Dict, Union, Set, Tuple
import copy
from consensus import (
    State, SlowVote, FastVote, BeaconVote, PayloadVote, Block, Checkpoint, majority_fork_choice,
    process_block, get_fork_choice_head,
    compute_hash, slot_to_epoch
)
from collections import defaultdict

ConsensusObject = Union[Block, BeaconVote, PayloadVote, SlowVote]

SLOT_DURATION = 10  # time units
ZERO_HASH = '0'*64
KAPPA = 32
MAX_BACKOFF_INTERVAL_EXPONENT = 4



# A basic Staker node implementation
class Staker:
    def __init__(self, validator_id: int, network: 'P2PNetwork', genesis_block: Block, genesis_state: State):
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
        # Last height this node voted for (to prevent duplicate height votes)
        self.last_voted_height: int = 0
        # Join the p2p network
        self.network.register_staker(self)

    @property
    def height(self):
        return max(s.height for s in self.post_states.values())

    @property
    def latest_justified(self):
        latest = max(   
            self.post_states.values(),
            key=lambda s: s.latest_justified.height
        )
        return latest.latest_justified

    @property
    def latest_finalized(self):
        latest = max(   
            self.post_states.values(),
            key=lambda s: s.latest_finalized.height
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
                self.propose()


    def is_proposer(self):
        return (self.get_current_slot() + 1) % self.config.num_validators == self.validator_id

    def propose(self):
        root = self.latest_justified.hash
        slot = self.get_current_slot() + 1
        self.head = get_fork_choice_head(
            blocks=self.chain,
            root=root,
            fast_votes=self.get_payload_votes_for_fork_choice(),
            slow_votes=self.latest_slow_votes.values(),
        )
        head_state = self.post_states[self.head]
        finalized_epoch = slot_to_epoch(head_state.latest_finalized.slot, self.config)
        finalized_height = head_state.latest_finalized.height
        slow_votes_to_include = [
                vote for (_, epoch), vote in self.slow_votes.items()
                if epoch > finalized_epoch and (vote.target is None or vote.target.height > finalized_height)
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
        root = self.latest_justified.hash
        self.head = get_fork_choice_head(
            blocks=self.chain,
            root=root,
            fast_votes=self.get_payload_votes_for_fork_choice(),
            slow_votes=self.latest_slow_votes.values()
        )
        vote = BeaconVote(
            validator_id=self.validator_id,
            slot=self.get_current_slot(),
            head=self.head,
        )
        
        # Clear payload vote trackers
        self.payload_votes.clear()
        self.payload_vote_equivocations.clear()
        self.network.submit(vote, self.validator_id)


    def slow_vote(self):
        target = self.get_target_checkpoint()
        vote =  SlowVote(
            validator_id=self.validator_id,
            epoch=self.get_current_epoch(),
            target=target,
            confirmed=self.confirmed_hash
        )
        if target is not None:
            self.last_voted_height = target.height
        self.network.submit(vote, self.validator_id)

    def fast_confirm(self):
        root = self.latest_justified.hash
        beacon_votes = self.get_beacon_votes_for_fork_choice()
        latest_slow_votes = self.latest_slow_votes.values()

        self.head = get_fork_choice_head(
            blocks=self.chain,
            root=root,
            fast_votes=beacon_votes,
            slow_votes=latest_slow_votes,
        )
        fast_confirmed_hash = get_fork_choice_head(
            blocks=self.chain,
            root=root,
            fast_votes=beacon_votes,
            slow_votes=latest_slow_votes,
            min_score=self.config.num_validators * 3 // 4
        )
        fast_confirmed_block = self.chain[fast_confirmed_hash]
        if fast_confirmed_block.slot >= self.get_current_slot() - KAPPA:
            self.update_confirmed(fast_confirmed_hash)
        else:
            kappa_deep_slot = self.get_current_slot() - KAPPA
            kappa_deep_hash = self.get_ancestor_at_slot(self.head, kappa_deep_slot)
            self.update_confirmed(kappa_deep_hash)


    def payload_vote(self):
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
            root=self.latest_justified.hash,
            fast_votes=beacon_votes,
            slow_votes=self.latest_slow_votes.values(),
            min_score=min_score,
        )
        vote = PayloadVote(
            validator_id=self.validator_id,
            slot=self.get_current_slot(),
            head=self.head,
            payload_available=True, # no payload for now, TODO: add payload
        )
        

        self.network.submit(vote, self.validator_id)


    def available_confirm(self):
        if self.confirmed_hash != self.head:
            beacon_votes = self.get_beacon_votes_for_fork_choice()
            timely_beacon_votes = [vote for vote in beacon_votes if self.timely_beacon_voters[vote.validator_id]]
            total_votes = len(beacon_votes) + len(self.beacon_vote_equivocations)
            majority_threshold = (total_votes+1) // 2
            new_confirmed_hash = get_fork_choice_head(
                blocks=self.chain,
                root=self.latest_justified.hash,
                slow_votes=self.latest_slow_votes.values(),
                fast_votes=timely_beacon_votes,
                min_score=majority_threshold + 1,
            )

            self.update_confirmed(new_confirmed_hash)

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
        return self.get_current_slot() % self.config.slots_per_epoch == 0

    def get_ancestor_at_slot(self, hash: str, slot: int):
        genesis_slot = self.chain[self.genesis_hash].slot
        current_block = self.chain[hash]
        while current_block.slot > slot >= genesis_slot:
            current_block = self.chain[current_block.parent]
        return compute_hash(current_block)

    def get_target_checkpoint(self):
        # Do not set a target if you have already voted at this height
        if self.last_voted_height == self.height:
            return None
        backoff_interval = self.compute_backoff_interval()
        # Do not set a target if the current epoch is not a multiple of the backoff interval.
        if not self.get_current_epoch() % backoff_interval == 0:
            return None
        # If the backoff is not active (interval is 1), use the confirmed block as target
        if backoff_interval == 1:
            target_hash = self.confirmed_hash 
        # If the backoff is active, fallback touse the k-deep block as target
        else:
            majority_hash = majority_fork_choice(
                self.chain,
                self.latest_justified.hash,
                self.latest_slow_votes.values()
            )
            majority_block = self.chain[majority_hash]
            kappa_deep_slot = self.get_current_slot() - KAPPA
            target_hash = self.get_ancestor_at_slot(self.head, max(majority_block.slot, kappa_deep_slot))

        target_block = self.chain[target_hash]
        return Checkpoint(
            hash=target_hash,
            slot=target_block.slot,
            height=self.height,
        )

    # A backoff interval is calculated based on an exponential backoff mechanism (with a cap),
    # using the distance from the last finalized epoch. The target should be set only if the epoch
    # is a multiple of the interval. The mechanism helps finality progress under high latency.
    def compute_backoff_interval(self):
        epoch = self.get_current_epoch()
        finalized_epoch = slot_to_epoch(self.latest_finalized.slot, self.config)
        max_backoff_interval = 2**MAX_BACKOFF_INTERVAL_EXPONENT
        # if backoff_interval = max_backoff_interval and we finalize in a single "backoff epoch",
        # i.e. epoch - finalized_epoch = max_backoff_interval, we should scale back the backoff,
        # because we finalized as soon as possible given the backoff. In that case,
        # delta = MAX_BACKOFF_INTERVAL_EXPONENT - 1, so we move to backoff_interval = 2**(MAX_BACKOFF_INTERVAL_EXPONENT - 1)
        delta = (epoch - finalized_epoch) * (MAX_BACKOFF_INTERVAL_EXPONENT - 1)
        delta = delta // max_backoff_interval
        return min(2**delta, max_backoff_interval) 
        
    def update_confirmed(self, new_confirmed_hash: str):
        new_confirmed_block = self.chain[new_confirmed_hash]
        confirmed_block = self.chain[self.confirmed_hash]
        
        # If the new confirmed slot is greater than the old, update
        if new_confirmed_block.slot >= confirmed_block.slot:
            self.confirmed_hash = new_confirmed_hash
            return
        
        if self.get_ancestor_at_slot(self.confirmed_hash, new_confirmed_block.slot) != new_confirmed_hash:
            # New is not an ancestor of old, update to new
            self.confirmed_hash = new_confirmed_hash
    
    # Called by the p2p network
    def receive(self, item: ConsensusObject):
        if isinstance(item, Block):
            self.receive_block(item)
        elif isinstance(item, SlowVote):
            self.receive_slow_vote(item)
        elif isinstance(item, BeaconVote):
            self.receive_beacon_vote(item)
        elif isinstance(item, PayloadVote):
            self.receive_payload_vote(item)

    def receive_block(self, block: Block):
        block_hash = compute_hash(block)
        # If the block is already known, ignore it
        if block_hash in self.chain:
            return
        # Ignore blocks from too far in the future
        if self.is_block_from_future(block.slot):
            return
        parent_state = self.post_states.get(block.parent)
        if parent_state:
            state = process_block(copy.deepcopy(parent_state), block)
            self.chain[block_hash] = block
            self.post_states[block_hash] = state
            # Receive slow votes
            for vote in block.slow_votes:   
                self.receive(vote)
            # Receive payload votes if the block is timely, i.e. we're still in the prior slot 
            # (block is received during the last fifth of the slot)
            timely_block = block.slot == self.get_current_slot() + 1
            if timely_block:
                for vote in block.payload_votes:
                    self.receive_payload_vote(vote, from_block=True)
            # Once we have received a block, also process all of
            # its dependencies
            if block_hash in self.dependencies:
                for item2 in self.dependencies[block_hash]:
                    self.receive(item2)
                del self.dependencies[block_hash]
        else:
            # If we have not yet seen the block's parent, ignore for now,
            # process later once we actually see the parent
            self.dependencies.setdefault(block.parent, []).append(block)

    def receive_slow_vote(self, slow_vote: SlowVote):
        # Ignore votes from future epochs
        if slow_vote.epoch > self.get_current_epoch():
            return
        # Ignore votes from equivocators
        if slow_vote.validator_id in self.equivocators:
            return

        
        if slow_vote.confirmed in self.chain:
            # if set, target must be an ancestor of confirmed
            if slow_vote.target is not None: 
                if self.get_ancestor_at_slot(slow_vote.confirmed, slow_vote.target.slot) != slow_vote.target.hash:
                    return

            vote_epoch = slow_vote.epoch
            vote_key = (slow_vote.validator_id, vote_epoch)
            
            # Already have a vote for this epoch, either already seen or an equivocation
            if vote_key in self.slow_votes:
                # Check for equivocation: two different votes from same (validator, epoch) pair
                if self.slow_votes[vote_key] != slow_vote:
                    self.handle_equivocation(slow_vote.validator_id)
                return
            
            # Store the vote
            self.slow_votes[vote_key] = slow_vote
            
            # Update latest slow vote if this is a newer vote
            if (
                slow_vote.validator_id not in self.latest_slow_votes
                or vote_epoch > self.latest_slow_votes[slow_vote.validator_id].epoch
            ):
                self.latest_slow_votes[slow_vote.validator_id] = slow_vote
        else:
            self.dependencies.setdefault(slow_vote.confirmed, []).append(slow_vote)


                    
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


    def receive_payload_vote(self, vote: PayloadVote, from_block: bool = False):
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

    def is_block_from_future(self, block_slot: int):
            current_slot = self.get_current_slot()
            if block_slot <= current_slot:
                return False
            # Ignore blocks for a future slot other than the next
            if block_slot > current_slot + 1:
                return True
            if block_slot == current_slot + 1:
                time_in_slot = (self.network.time % SLOT_DURATION)
                slot_fifth = SLOT_DURATION // 5
                return time_in_slot < 4*slot_fifth

                
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
            # Immediately receive your own messages
            if recipient_id == sender_id:
                self.stakers[recipient_id].receive(item)
            deliver_at = self.time + self.latency_func(self.time)
            self.queues[recipient_id].append((deliver_at, item))

    def time_step(self):
        self.time += 1
        for validator_id, queue in self.queues.items():
            deliver_now = [item for (t, item) in queue if t <= self.time]
            self.queues[validator_id] = [(t, item) for (t, item) in queue if t > self.time]
            for item in deliver_now:
                self.stakers[validator_id].receive(item)
