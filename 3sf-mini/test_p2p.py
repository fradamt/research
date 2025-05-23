from consensus import (
    State, Vote, Block, Config, Checkpoint,
    get_latest_justified_checkpoint, get_fork_choice_head,
    compute_hash
)
from p2p import Staker, P2PNetwork
from consensus import is_justifiable_slot
from typing import Optional, List, Dict
import random
import argparse

SLOT_DURATION = 12
NUM_STAKERS = 10
ZERO_HASH = '0'*64

import matplotlib.pyplot as plt
import networkx as nx


import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_view(fig, ax, staker: Staker, title="Staker's View", prune: bool = True):
    G = nx.DiGraph()
    plt.clf()
    
    # Helper function to get all blocks in the valid chain (finalized chain and its descendants)
    def get_valid_chain_blocks():
        if not prune:
            return set(staker.chain.keys())  # If not pruning, show all blocks
            
        valid_blocks = set()
        if staker.latest_finalized.hash == ZERO_HASH:
            return set(staker.chain.keys())  # If nothing is finalized, show all blocks
            
        # Start from finalized block and traverse up to genesis
        current = staker.latest_finalized.hash
        while current != ZERO_HASH:
            valid_blocks.add(current)
            current = staker.chain[current].parent
            
        # Add all descendants of the finalized chain
        to_process = [staker.latest_finalized.hash]
        while to_process:
            current = to_process.pop()
            for block in staker.chain.values():
                if block.parent == current:
                    valid_blocks.add(compute_hash(block))
                    to_process.append(compute_hash(block))
                    
        return valid_blocks

    valid_blocks = get_valid_chain_blocks()
    
    # First, build the graph structure
    children_map = defaultdict(list)
    for block in staker.chain.values():
        h = compute_hash(block)
        if h not in valid_blocks:
            continue
        if block.parent != ZERO_HASH and block.parent in valid_blocks:
            children_map[block.parent].append(h)
        G.add_node(h[:8], slot=block.slot)

    for parent, children in children_map.items():
        for child in children:
            G.add_edge(parent[:8], child[:8])

    # DFS traversal to assign consistent x positions
    pos = {}
    x_counter = [0]
    max_validator_id = max(
        [staker.validator_id] +
        [vote.validator_id for vote in staker.known_votes]
    )

    def dfs(block_hash, depth=0, is_finalized=False):
        if block_hash not in children_map:
            if is_finalized:
                x = 0  # Keep finalized chain centered
            else:
                x = x_counter[0]
                x_counter[0] += 1
            pos[block_hash[:8]] = (x, -staker.chain[block_hash].slot)
            return x
        child_xs = []
        for child in sorted(children_map[block_hash]):  # sort for determinism
            # Check if this child is in the finalized chain
            child_is_finalized = False
            if staker.latest_finalized.hash != ZERO_HASH:
                current = staker.latest_finalized.hash
                while current != ZERO_HASH:
                    if current == child:
                        child_is_finalized = True
                        break
                    current = staker.chain[current].parent
            child_xs.append(dfs(child, depth + 1, child_is_finalized))
        if is_finalized:
            x = 0  # Keep finalized chain centered
        else:
            x = sum(child_xs) / len(child_xs)
        pos[block_hash[:8]] = (x, -staker.chain[block_hash].slot)
        return x

    # Start DFS from genesis
    genesis_is_finalized = staker.latest_finalized.hash == staker.genesis_hash
    dfs(staker.genesis_hash, is_finalized=genesis_is_finalized)

    # Color blocks
    justified_hash = get_latest_justified_checkpoint(staker.post_states).hash
    finalized_hash = staker.latest_finalized.hash
    head_block = get_fork_choice_head(staker.chain, justified_hash, staker.known_votes)

    node_colors = []
    node_sizes = []
    for node in G.nodes:
        if node == justified_hash[:8]:
            node_colors.append("blue")
            node_sizes.append(600)
        elif node == finalized_hash[:8]:
            node_colors.append("purple")
            node_sizes.append(600)
        elif node == head_block[:8]:
            node_colors.append("green")
            node_sizes.append(600)
        else:
            node_colors.append("black")
            # Check if this node is in the finalized chain
            is_finalized = False
            if staker.latest_finalized.hash != ZERO_HASH:
                current = staker.latest_finalized.hash
                while current != ZERO_HASH:
                    if current[:8] == node:
                        is_finalized = True
                        break
                    current = staker.chain[current].parent
            node_sizes.append(600 if is_finalized else 200)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw votes
    latest_votes = {}
    for vote in sorted(staker.known_votes, key = lambda vote: vote.slot):
        latest_votes[vote.validator_id] = vote

    for vote in latest_votes.values():
        if vote.head[:8] not in pos or (vote.target is not None and vote.target.hash[:8] not in pos):
            continue
        voter_node = f"v{vote.validator_id}"
        offset = (vote.validator_id - max_validator_id / 2) * 0.15
        pos[voter_node] = (pos[vote.head[:8]][0] + offset, pos[vote.head[:8]][1] - 0.5)

        G.add_node(voter_node, node_size=5)
        G.add_edge(voter_node, vote.head[:8])

        color = "orange"
        nx.draw_networkx_nodes(G, pos, nodelist=[voter_node], node_color=color, node_size=200)
        nx.draw_networkx_edges(G, pos, edgelist=[(voter_node, vote.head[:8])], edge_color=color,
                               style="dashed", arrowsize=8)
        # Only add target edge if target exists
        if vote.target is not None:
            G.add_edge(voter_node, vote.target.hash[:8])
            nx.draw_networkx_edges(G, pos, edgelist=[(voter_node, vote.target.hash[:8])], edge_color="grey",
                                   style="dashed", arrowsize=8)
            
        nx.draw_networkx_labels(G, pos, labels={voter_node: f"v{vote.validator_id}"}, font_size=6)

    # Add time and finality distance info
    current_slot = staker.get_current_slot()
    finalized_slot = staker.post_states[staker.head].latest_finalized.checkpoint_slot
    distance_from_finality = current_slot - finalized_slot
    
    info_text = f"Time: {staker.network.time}\nDistance from finality: {distance_from_finality} slots"
    plt.figtext(0.7, 0.7, info_text, 
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=12)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_progression(justified_slots, finalized_slots):
    plt.figure(figsize=(12, 6))
    
    plt.plot(range(len(justified_slots)), label='Slot', color='green', linestyle='--')
    plt.plot(justified_slots, label='Max Justified Slot', color='blue')
    plt.plot(finalized_slots, label='Max Finalized Slot', color='purple')
    
    plt.xlabel('Time (simulation slots elapsed)')
    plt.ylabel('Slot Number')
    plt.title('Progression of Justified, Finalized')
    plt.legend()
    plt.grid(True)
    # No plt.show() here, it's called once at the end of the script

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a P2P network simulation')
    parser.add_argument('--no-backoff', action='store_true', help='Disable k-th ancestor backoff')
    parser.add_argument('--no-pruning', action='store_true', help='Prune conflicting branches when finalized')
    parser.add_argument('--max-backoff', type=int, default=8, help='Maximum checkpoint interval for backoff')
    parser.add_argument('--latency', type=int, help='latency to use')
    parser.add_argument('--time', type=int, default=1000, help='Number of time steps to run')
    parser.add_argument('--no-viz', action='store_true', help='Disable interactive graph visualization')
    args = parser.parse_args()

    SLOT_DURATION = 12
    NUM_STAKERS = 10

    if not args.no_viz:
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)  # Show interactive plot window non-blockingly
    else:
        fig, ax = None, None # Ensure fig and ax are defined even if no_viz is true

    # Create genesis block and state
    genesis_block = Block(slot=1, parent=ZERO_HASH)
    config = Config(num_validators=NUM_STAKERS, max_checkpoint_interval_for_backoff=args.max_backoff)
    genesis_state = State(
        config=config,
        latest_justified=Checkpoint(hash=ZERO_HASH, chain_slot=0, checkpoint_slot=0),
        latest_finalized=Checkpoint(hash=ZERO_HASH, chain_slot=0, checkpoint_slot=0),
        historical_block_hashes=[ZERO_HASH],
        justified_slots=[True],
    )
    genesis_block.state_root = compute_hash(genesis_state)
    genesis_hash = compute_hash(genesis_block)

    def latency_func(t):
            if t < 2 * args.time // 3:
                if args.latency is not None:
                    return args.latency * SLOT_DURATION
                else:
                    return int(SLOT_DURATION * 2.5 * random.random() ** 3)
            else:
                return 1


    network = P2PNetwork(latency_func)
    stakers = [Staker(i, network, genesis_block, genesis_state, use_backoff=not args.no_backoff) for i in range(NUM_STAKERS)]

    # Initialize all stakers with genesis
    for staker in stakers:
        assert staker.head == genesis_hash

    # Initialize data collection for progression plot
    justified_slots = []
    finalized_slots = []
    actual_slots_data = [] # New list for actual slots

    # Simulation loop
    for time in range(args.time):
        # Deliver messages
        network.time_step()

        # Run staker code
        for staker in stakers:
            staker.tick()

        # Collect data for progression plot
        if time % SLOT_DURATION == 0:
            current_slot = time // SLOT_DURATION
            max_justified = max(staker.latest_justified.checkpoint_slot for staker in stakers)
            max_finalized = max(staker.latest_finalized.checkpoint_slot for staker in stakers)
            justified_slots.append(max_justified)
            finalized_slots.append(max_finalized)
            actual_slots_data.append(current_slot) # Store current slot

        # Periodic printout
        if time % SLOT_DURATION == 0:
            print(f"\n=== Time {time}, Slot {time // SLOT_DURATION} === ")
            for staker in stakers:
                head = staker.head
                ljs = staker.latest_justified.checkpoint_slot
                ljh = staker.latest_justified.hash
                lfs = staker.latest_finalized.checkpoint_slot
                lfh = staker.latest_finalized.hash
                target_block = staker.get_target_block()
                tbh = compute_hash(target_block)
                tbs = target_block.slot
                is_justifiable = is_justifiable_slot(config, staker.post_states[staker.head].latest_finalized.checkpoint_slot, time // SLOT_DURATION)
                print(f"Staker {staker.validator_id}: Head={head[:8]} ({staker.chain[head].slot}) | Target={tbh[:8]} ({tbs}) {'✓' if is_justifiable else '✗'} | Justified={ljh[:8]} ({ljs}) | Finalized={lfh[:8]} ({lfs})")
        if not args.no_viz and time % 60 == 9:
            plot_view(fig, ax, stakers[0], "Chain View", prune=not args.no_pruning) # Pass prune correctly

    if not args.no_viz:
        plt.ioff() # Turn off interactive mode before the final blocking show

    # Plot the progression at the end
    plot_progression(justified_slots, finalized_slots)
    plt.show()  # Show all figures and block until closed
