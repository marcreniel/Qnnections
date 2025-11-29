"""Main training script for DQN Connections Solver."""
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from src.common.data_loader import DataLoader
from src.dqn.env import ConnectionsEnv
from src.dqn.agent import DQNAgent
from src.common.embeddings import _GEMMA_DIM

def evaluate(agent: DQNAgent, env: ConnectionsEnv, puzzles: list, num_episodes: int = 20):
    """Evaluates the agent on a set of puzzles."""
    total_reward = 0
    successes = 0
    groups_found = 0
    total_mistakes = 0
    
    # Use a subset of puzzles
    eval_puzzles = puzzles[:num_episodes] if len(puzzles) > num_episodes else puzzles
    
    for puzzle in eval_puzzles:
        obs = env.reset(puzzle=puzzle)
        done = False
        episode_reward = 0
        
        while not done:
            action_mask = env.get_action_mask()
            # Greedy action
            action = agent.act(obs, action_mask, epsilon=0.0)
            
            next_obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            obs = next_obs
            
            if info["result"] == "correct":
                groups_found += 1
                
        total_reward += episode_reward
        if info["episode_end"] == "success":
            successes += 1
        total_mistakes += (env.mistakes_allowed - info["mistakes_left"])
            
    metrics = {
        "avg_reward": total_reward / len(eval_puzzles),
        "success_rate": successes / len(eval_puzzles),
        "avg_groups": groups_found / len(eval_puzzles),
        "avg_mistakes": total_mistakes / len(eval_puzzles)
    }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    
    # Ablations
    parser.add_argument("--w_correct", type=float, default=1.0)
    parser.add_argument("--w_first", type=float, default=0.1)
    parser.add_argument("--w_second", type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    # Load Data
    loader = DataLoader(args.data_path)
    loader.load_data()
    all_puzzles = loader.get_dataset()
    
    # Split train/eval
    random.shuffle(all_puzzles)
    split_idx = int(len(all_puzzles) * 0.9)
    train_puzzles = all_puzzles[:split_idx]
    eval_puzzles = all_puzzles[split_idx:]
    
    # Initialize Env and Agent
    reward_weights = {
        "correctness": args.w_correct,
        "first_order": args.w_first,
        "second_order": args.w_second
    }
    env = ConnectionsEnv(data_loader=loader, reward_weights=reward_weights)
    
    # Determine embedding dim from loader (load one to check)
    # Or import from embeddings.py if constant
    # But loader.get_embeddings returns what?
    # Let's check one puzzle
    sample_embeds = loader.get_embeddings(train_puzzles[0]["words"])
    embedding_dim = sample_embeds.shape[1]
    print(f"Embedding dim: {embedding_dim}")
    
    agent = DQNAgent(
        embedding_dim=embedding_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=device
    )
    
    # Training Loop
    epsilon = args.epsilon_start
    
    pbar = tqdm(range(args.episodes))
    for episode in pbar:
        # Sample puzzle
        puzzle = random.choice(train_puzzles)
        obs = env.reset(puzzle=puzzle)
        done = False
        total_reward = 0
        
        while not done:
            action_mask = env.get_action_mask()
            action = agent.act(obs, action_mask, epsilon)
            
            next_obs, reward, done, _, info = env.step(action)
            next_action_mask = env.get_action_mask() # For next state
            
            agent.buffer.push(obs, action, reward, next_obs, done, action_mask, next_action_mask)
            
            loss = agent.update()
            
            obs = next_obs
            total_reward += reward
            
        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        
        pbar.set_description(f"Ep {episode} | R: {total_reward:.2f} | Eps: {epsilon:.2f}")
        
        if episode > 0 and episode % args.eval_freq == 0:
            metrics = evaluate(agent, env, eval_puzzles)
            print(f"\nEvaluation at episode {episode}:")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"  Avg Groups: {metrics['avg_groups']:.2f}")
            
    # Save model
    torch.save(agent.q_net.state_dict(), "dqn_connections.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
