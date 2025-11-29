"""Training script for PPO agent on Connections."""
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from src.common.data_loader import DataLoader
from src.dqn.env import ConnectionsEnv
from src.ppo.agent import PPOAgent
from src.common.embeddings import _GEMMA_DIM

def evaluate(agent: PPOAgent, env: ConnectionsEnv, puzzles: list, num_episodes: int = 20):
    """Evaluates the agent on a set of puzzles."""
    total_reward = 0
    successes = 0
    total_groups = 0
    
    # Use a fixed subset of puzzles for consistent evaluation
    eval_puzzles = puzzles[:num_episodes]
    if len(eval_puzzles) < num_episodes:
        eval_puzzles = (eval_puzzles * (num_episodes // len(eval_puzzles) + 1))[:num_episodes]
        
    for puzzle in eval_puzzles:
        obs = env.reset(puzzle=puzzle)
        done = False
        episode_reward = 0
        groups_found = 0
        
        while not done:
            action_mask = env.get_action_mask()
            # Greedy selection for eval
            action, _, _ = agent.select_action(obs, action_mask, greedy=True)
            
            next_obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            
            if info["result"] == "correct":
                groups_found += 1
                
            obs = next_obs
            
        total_reward += episode_reward
        if info["episode_end"] == "success":
            successes += 1
        total_groups += groups_found
        
    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes,
        "avg_groups": total_groups / num_episodes
    }

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Connections")
    parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes")
    parser.add_argument("--eval_freq", type=int, default=50, help="Evaluation frequency (episodes)")
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json", help="Path to puzzles JSON")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--steps_per_update", type=int, default=2048, help="Env steps per PPO update")
    
    # Reward shaping weights
    parser.add_argument("--w_correct", type=float, default=2.0, help="Weight for correct group")
    parser.add_argument("--w_first", type=float, default=0.1, help="Weight for 1st order cohesion")
    parser.add_argument("--w_second", type=float, default=0.05, help="Weight for 2nd order cohesion")
    
    # Embedding config
    parser.add_argument("--gemma_device", type=str, default=None, help="Device for Gemma")
    
    args = parser.parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data
    data_loader = DataLoader(args.data_path, embedding_source="gemma")
    data_loader.load_data()
    print(f"Loaded {len(data_loader.puzzles)} puzzles.")
    
    # Split Data
    # Shuffle and split 90/10
    random.shuffle(data_loader.puzzles)
    split_idx = int(len(data_loader.puzzles) * 0.9)
    train_puzzles = data_loader.puzzles[:split_idx]
    eval_puzzles = data_loader.puzzles[split_idx:]
    
    # Reward Weights
    reward_weights = {
        "correctness": args.w_correct,
        "first_order": args.w_first,
        "second_order": args.w_second
    }
    
    # Init Env and Agent
    env = ConnectionsEnv(data_loader=data_loader, reward_weights=reward_weights)
    agent = PPOAgent(
        embedding_dim=_GEMMA_DIM,
        learning_rate=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        device=device
    )
    
    # Training Loop
    # PPO updates happen after collecting a batch of trajectories
    total_steps = 0
    pbar = tqdm(total=args.episodes, desc="Training")
    
    # We iterate by episodes for consistency with DQN loop structure, 
    # but PPO updates happen based on steps.
    
    current_obs = env.reset(puzzle=random.choice(train_puzzles))
    episode_rewards = []
    current_ep_reward = 0
    
    episodes_completed = 0
    
    while episodes_completed < args.episodes:
        # Collect Rollout
        for _ in range(args.steps_per_update):
            action_mask = env.get_action_mask()
            action, log_prob, value = agent.select_action(current_obs, action_mask)
            
            next_obs, reward, done, _, info = env.step(action)
            
            agent.buffer.add(current_obs, action, reward, done, log_prob, value, action_mask)
            
            current_ep_reward += reward
            current_obs = next_obs
            total_steps += 1
            
            if done:
                episode_rewards.append(current_ep_reward)
                pbar.set_description(f"Ep {episodes_completed} | R: {current_ep_reward:.2f}")
                pbar.update(1)
                episodes_completed += 1
                
                # Evaluation
                if episodes_completed % args.eval_freq == 0:
                    metrics = evaluate(agent, env, eval_puzzles)
                    tqdm.write(f"Evaluation at episode {episodes_completed}:")
                    tqdm.write(f"  Success Rate: {metrics['success_rate']:.2%}")
                    tqdm.write(f"  Avg Reward: {metrics['avg_reward']:.2f}")
                    tqdm.write(f"  Avg Groups: {metrics['avg_groups']:.2f}")
                
                if episodes_completed >= args.episodes:
                    break
                    
                current_ep_reward = 0
                current_obs = env.reset(puzzle=random.choice(train_puzzles))
                
        # Update Policy
        agent.update()
        
    pbar.close()
    
    # Save Model
    torch.save(agent.policy.state_dict(), "ppo_connections.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
