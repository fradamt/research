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
    for vote in sorted(staker.known_votes, key = lambda v: v.target.slot):
        latest_votes[vote.validator_id] = vote

    for vote in latest_votes.values():
        if vote.head[:8] not in pos:
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
        
        # Only add FFG target edge if target.hash is not ZERO_HASH and its position is known
        if vote.target.hash != ZERO_HASH and vote.target.hash[:8] in pos:
            G.add_edge(voter_node, vote.target.hash[:8]) # Add FFG target edge
            nx.draw_networkx_edges(G, pos, edgelist=[(voter_node, vote.target.hash[:8])], edge_color="grey",
                                   style="dashed", arrowsize=8)
            
        nx.draw_networkx_labels(G, pos, labels={voter_node: f"v{vote.validator_id}"}, font_size=6)

    # Add time and finality distance info
    current_slot = staker.get_current_slot()
    finalized_slot = staker.post_states[staker.head].latest_finalized.slot
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

def plot_progression(justified_slots, finalized_slots, justified_block_slots, finalized_block_slots):
    plt.figure(figsize=(12, 6))
    
    time_axis = list(range(2, len(justified_slots) + 2))
    plt.plot(time_axis, justified_slots, label='Max Justified Checkpoint Slot', color='blue')
    plt.plot(time_axis, finalized_slots, label='Max Finalized Checkpoint Slot', color='purple')
    plt.plot(time_axis, justified_block_slots, label='Max Justified Block Slot', color='cyan', linestyle='--')
    plt.plot(time_axis, finalized_block_slots, label='Max Finalized Block Slot', color='magenta', linestyle='--')
    
    plt.xlabel('Time (simulation slots elapsed)')
    plt.ylabel('Slot Number')
    plt.title('Progression of Checkpoint vs Block Slots')
    plt.legend()
    plt.grid(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a P2P network simulation')
    parser.add_argument('--no-pruning', action='store_true', help='Prune conflicting branches when finalized')
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
    config = Config(num_validators=NUM_STAKERS)
    genesis_state = State(
        config=config,
        latest_justified=Checkpoint(hash=ZERO_HASH, slot=0),
        latest_finalized=Checkpoint(hash=ZERO_HASH, slot=0),
        historical_block_hashes=[ZERO_HASH],
        justified_checkpoints=[Checkpoint(hash=ZERO_HASH, slot=0)],
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
    stakers = [Staker(i, network, genesis_block, genesis_state) for i in range(NUM_STAKERS)]

    # Initialize all stakers with genesis
    for staker in stakers:
        assert staker.head == genesis_hash

    # Initialize data collection for progression plot
    justified_slots = []
    finalized_slots = []
    justified_block_slots = []
    finalized_block_slots = []

    # Simulation loop
    for time in range(args.time):
        # Deliver messages
        network.time_step()

        # Run staker code
        for staker in stakers:
            staker.tick()

        # Collect data for progression plot
        if time % SLOT_DURATION == 0:
            slot = time // SLOT_DURATION + 2
            print(f"\n=== Time {time}, Slot {slot} ===")
            max_justified = max(s.latest_justified.slot for s in stakers)
            max_finalized = max(s.latest_finalized.slot for s in stakers)
            justified_slots.append(max_justified)
            finalized_slots.append(max_finalized)

            # Track corresponding block slots
            max_justified_block = max(
                staker.chain[staker.latest_justified.hash].slot if staker.latest_justified.hash in staker.chain else 0
                for staker in stakers
            )
            max_finalized_block = max(
                staker.chain[staker.latest_finalized.hash].slot if staker.latest_finalized.hash in staker.chain else 0
                for staker in stakers
            )
            justified_block_slots.append(max_justified_block)
            finalized_block_slots.append(max_finalized_block)

            for staker in stakers:
                head = staker.head
                # Access .slot for FFG checkpoint slots
                ljs = staker.latest_justified.slot
                ljh = staker.latest_justified.hash
                lfs = staker.latest_finalized.slot
                lfh = staker.latest_finalized.hash
                
                is_justifiable_now = is_justifiable_slot(staker.latest_finalized.slot, slot)
                
                ffg_target_display_str: str
                if is_justifiable_now:
                    target_block = staker.get_target_block() # The block whose hash is used for FFG target
                    ffg_target_hash_display = compute_hash(target_block)
                    # FFG target slot is staker_current_block_slot. The block itself is actual_target_block.
                    ffg_target_display_str = f"{ffg_target_hash_display[:8]} (block slot {target_block.slot}, FFG target slot {slot})"
                else:
                    # FFG target is ZERO_HASH, FFG target slot is staker_current_block_slot
                    ffg_target_display_str = f"{ZERO_HASH[:8]} (FFG target slot {slot})"

                print(f"Staker {staker.validator_id}: Head={head[:8]} ({staker.chain[head].slot}) | FFG Target={ffg_target_display_str} {'✓' if is_justifiable_now else '✗'} | Justified={ljh[:8]} ({ljs}) | Finalized={lfh[:8]} ({lfs})")
        if not args.no_viz and time % 60 == 9:
            plot_view(fig, ax, stakers[0], "Chain View", prune=not args.no_pruning) # Pass prune correctly

    if not args.no_viz:
        plt.ioff() # Turn off interactive mode before the final blocking show

    # Plot the progression at the end
    plot_progression(justified_slots, finalized_slots, justified_block_slots, finalized_block_slots)
    plt.show()  # Show all figures and block until closed
