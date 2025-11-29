"""Demonstration generation for DQN Warm Start."""
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

from src.dqn_modified.env import ConnectionsEnv
from src.dqn.actions import get_action_idx

def generate_demo_trajectory(env: ConnectionsEnv, puzzle: Dict) -> List[Tuple]:
    """
    Generates a perfect play trajectory for a single puzzle.
    Returns a list of (state, action, reward, next_state, done).
    """
    transitions = []
    
    # Reset env to specific puzzle
    state = env.reset(puzzle=puzzle)
    done = False
    
    # Get group IDs and words
    # puzzle["group_ids"] is list of 16 ints
    # puzzle["words"] is list of 16 strings
    
    # We need to identify the 4 groups
    groups = [[] for _ in range(4)]
    for i, g_id in enumerate(puzzle["group_ids"]):
        groups[g_id].append(i)
        
    # Iterate through groups (order doesn't strictly matter for correctness, 
    # but let's just go 0 to 3)
    for group_indices in groups:
        if done:
            break
            
        # Convert indices to action index
        action_idx = get_action_idx(tuple(group_indices))
        
        # Get action mask for current state
        action_mask = env.get_action_mask()
        
        # Step env
        next_state, reward, done, _, _ = env.step(action_idx)
        
        # Get action mask for next state
        next_action_mask = env.get_action_mask()
        
        transitions.append((state, action_idx, reward, next_state, done, action_mask, next_action_mask))
        state = next_state
        
    return transitions

def build_demo_dataset(env: ConnectionsEnv, puzzles: List[Dict]) -> List[Tuple]:
    """
    Builds a dataset of perfect trajectories for all provided puzzles.
    """
    all_transitions = []
    print(f"Generating demonstrations for {len(puzzles)} puzzles...")
    
    for puzzle in tqdm(puzzles):
        traj = generate_demo_trajectory(env, puzzle)
        all_transitions.extend(traj)
        
    return all_transitions
