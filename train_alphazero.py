"""Training script for AlphaZero agent on Connections."""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import deque

from src.common.data_loader import DataLoader
from src.dqn.env import ConnectionsEnv
from src.alphazero.net import AlphaZeroNet
from src.alphazero.mcts import MCTS, EnvSimulator
from src.common.embeddings import _GEMMA_DIM
from src.dqn.actions import NUM_ACTIONS

def evaluate(network: AlphaZeroNet, env: ConnectionsEnv, puzzles: list, num_episodes: int = 20, device: str = "cpu"):
    """Evaluates the agent on a set of puzzles using MCTS with low temperature (greedy)."""
    total_reward = 0
    successes = 0
    total_groups = 0
    
    # Use a fixed subset of puzzles for consistent evaluation
    eval_puzzles = puzzles[:num_episodes]
    if len(eval_puzzles) < num_episodes:
        eval_puzzles = (eval_puzzles * (num_episodes // len(eval_puzzles) + 1))[:num_episodes]
        
    mcts = MCTS(network, device)
    
    for puzzle in eval_puzzles:
        obs = env.reset(puzzle=puzzle)
        done = False
        episode_reward = 0
        groups_found = 0
        
        while not done:
            # Create simulator for MCTS
            sim = EnvSimulator(obs["embeddings"], obs["mask"], obs["mistakes_left"], puzzle)
            
            # Run MCTS (fewer sims for eval speed)
            policy = mcts.run(obs, num_simulations=50, env_simulator=sim)
            
            # Greedy action selection (argmax of visit counts)
            action = int(np.argmax(policy))
            
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
    parser = argparse.ArgumentParser(description="Train AlphaZero Agent for Connections")
    parser.add_argument("--episodes", type=int, default=1000, help="Total self-play episodes")
    parser.add_argument("--eval_freq", type=int, default=50, help="Evaluation frequency (episodes)")
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json", help="Path to puzzles JSON")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--buffer_size", type=int, default=5000, help="Replay buffer size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per batch of games")
    parser.add_argument("--games_per_batch", type=int, default=10, help="Number of self-play games before training")
    
    args = parser.parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data
    data_loader = DataLoader(args.data_path, embedding_source="gemma")
    data_loader.load_data()
    print(f"Loaded {len(data_loader.puzzles)} puzzles.")
    
    # Split Data
    random.shuffle(data_loader.puzzles)
    split_idx = int(len(data_loader.puzzles) * 0.9)
    train_puzzles = data_loader.puzzles[:split_idx]
    eval_puzzles = data_loader.puzzles[split_idx:]
    
    # Init Network and Optimizer
    network = AlphaZeroNet(_GEMMA_DIM).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    
    # Replay Buffer
    # Stores (state, policy, value)
    replay_buffer = deque(maxlen=args.buffer_size)
    
    # Training Loop
    env = ConnectionsEnv(data_loader=data_loader) # Reward weights don't matter for MCTS value target logic
    mcts = MCTS(network, device)
    
    pbar = tqdm(total=args.episodes, desc="Self-Play")
    episodes_completed = 0
    
    while episodes_completed < args.episodes:
        # Collect Self-Play Games
        new_games = []
        
        for _ in range(args.games_per_batch):
            puzzle = random.choice(train_puzzles)
            obs = env.reset(puzzle=puzzle)
            done = False
            
            game_history = [] # List of (state, policy)
            
            while not done:
                sim = EnvSimulator(obs["embeddings"], obs["mask"], obs["mistakes_left"], puzzle)
                
                # Run MCTS
                policy = mcts.run(obs, num_simulations=args.sims, env_simulator=sim)
                
                # Sample action (exploration)
                # Add Dirichlet noise to root prior? (Optional, skipping for simplicity)
                action = np.random.choice(NUM_ACTIONS, p=policy)
                
                # Store state and policy
                game_history.append((obs, policy))
                
                obs, _, done, _, info = env.step(action)
                
            # Determine game outcome (z)
            # +1 for success, -1 for failure
            z = 1.0 if info["episode_end"] == "success" else -1.0
            
            # Add to buffer
            for state, pi in game_history:
                replay_buffer.append((state, pi, z))
                
            episodes_completed += 1
            pbar.update(1)
            
            if episodes_completed % args.eval_freq == 0:
                metrics = evaluate(network, env, eval_puzzles, device=device)
                tqdm.write(f"Evaluation at episode {episodes_completed}:")
                tqdm.write(f"  Success Rate: {metrics['success_rate']:.2%}")
                tqdm.write(f"  Avg Groups: {metrics['avg_groups']:.2f}")
                
            if episodes_completed >= args.episodes:
                break
                
        # Train Network
        if len(replay_buffer) >= args.batch_size:
            for _ in range(args.epochs):
                batch = random.sample(replay_buffer, args.batch_size)
                states, policies, values = zip(*batch)
                
                # Prepare tensors
                s_embeds = torch.FloatTensor(np.array([s["embeddings"] for s in states])).to(device)
                s_masks = torch.FloatTensor(np.array([s["mask"] for s in states])).to(device)
                s_mistakes = torch.FloatTensor(np.array([[s["mistakes_left"]] for s in states])).to(device)
                
                target_pis = torch.FloatTensor(np.array(policies)).to(device)
                target_vs = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)
                
                # Forward
                pred_logits, pred_vs = network(s_embeds, s_masks, s_mistakes)
                
                # Losses
                # Policy: Cross Entropy
                # pred_logits are unnormalized, target_pis are probs
                # log_softmax(logits) * target
                log_probs = F.log_softmax(pred_logits, dim=1)
                policy_loss = -torch.sum(target_pis * log_probs, dim=1).mean()
                
                # Value: MSE
                value_loss = F.mse_loss(pred_vs, target_vs)
                
                total_loss = policy_loss + value_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
    pbar.close()
    torch.save(network.state_dict(), "alphazero_connections.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
