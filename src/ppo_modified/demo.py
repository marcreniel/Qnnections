"""Demo generation for PPO Warm Start."""
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from src.dqn_modified.env import ConnectionsEnv
from src.dqn.actions import get_action_idx
from src.ppo_modified.env_utils import encode_state

def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns (G_t)."""
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def generate_demo_trajectory(env: ConnectionsEnv, puzzle: Dict) -> Dict:
    """
    Generates a perfect play trajectory for a single puzzle.
    Returns dict with lists of states, actions, rewards, returns.
    """
    # Reset env to specific puzzle
    env.reset(puzzle=puzzle)
    done = False
    
    states = []
    actions = []
    rewards = []
    
    # Identify groups
    groups = [[] for _ in range(4)]
    for i, g_id in enumerate(puzzle["group_ids"]):
        groups[g_id].append(i)
        
    for group_indices in groups:
        if done: break
        
        # Encode state
        state_vec = encode_state(env)
        
        # Get action
        action_idx = get_action_idx(tuple(sorted(group_indices)))
        
        # Step
        _, reward, done, _, _ = env.step(action_idx)
        
        states.append(state_vec)
        actions.append(action_idx)
        rewards.append(reward)
        
    # Compute returns
    returns = compute_returns(rewards)
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "returns": returns
    }

def build_demo_dataset(env: ConnectionsEnv, puzzles: List[Dict], num_demos: int) -> List[Dict]:
    """
    Builds a dataset of perfect trajectories.
    Returns a list of flattened (state, action, return) dicts for easy sampling.
    """
    print(f"Generating demonstrations for {num_demos} puzzles...")
    dataset = []
    
    # Use subset
    demo_puzzles = puzzles[:num_demos]
    
    for puzzle in tqdm(demo_puzzles):
        traj = generate_demo_trajectory(env, puzzle)
        
        # Flatten trajectory into individual transitions
        for i in range(len(traj["states"])):
            dataset.append({
                "state": traj["states"][i],
                "action": traj["actions"][i],
                "return": traj["returns"][i]
            })
            
    return dataset
