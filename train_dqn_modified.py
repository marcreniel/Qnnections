"""Training script for DQN Warm Start on Connections."""
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from src.common.data_loader import DataLoader
from src.dqn_modified.env import ConnectionsEnv
from src.dqn_modified.agent import DQNAgentModified
from src.dqn_modified.demo import build_demo_dataset
from src.common.embeddings import _GEMMA_DIM

def evaluate(agent: DQNAgentModified, env: ConnectionsEnv, puzzles: list, num_episodes: int = 20):
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
            action = agent.select_action(obs, action_mask, eval_mode=True)
            
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

def pretrain_on_demos(agent: DQNAgentModified, replay_buffer, num_steps: int, batch_size: int, margin: float, lambda_demo: float):
    """
    Run DQN updates using only the demonstration transitions.
    """
    if len(replay_buffer) < batch_size:
        print("Not enough demo transitions for pretraining.")
        return
        
    print(f"Pretraining for {num_steps} steps...")
    pbar = tqdm(range(num_steps), desc="Pretraining")
    losses = []
    
    for i in pbar:
        loss_dict = agent.update(demo_margin_loss=True, margin=margin, lambda_demo=lambda_demo)
        losses.append(loss_dict["loss"])
        
        if i % 100 == 0:
            pbar.set_postfix({
                "L": f"{loss_dict['loss']:.2f}",
                "TD": f"{loss_dict['td_loss']:.2f}",
                "M": f"{loss_dict['margin_loss']:.2f}",
                "Q": f"{loss_dict['q_max']:.2f}"
            })
        
    print(f"Pretraining complete. Avg Loss: {np.mean(losses):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train DQN Warm Start for Connections")
    parser.add_argument("--episodes", type=int, default=1000, help="Total online training episodes")
    parser.add_argument("--pretrain_steps", type=int, default=5000, help="Number of pretraining steps")
    parser.add_argument("--max_mistakes", type=int, default=8, help="Max mistakes allowed in env")
    parser.add_argument("--eval_freq", type=int, default=50, help="Evaluation frequency (episodes)")
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json", help="Path to puzzles JSON")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.8, help="Margin for supervised loss")
    parser.add_argument("--lambda_demo", type=float, default=10.0, help="Weight for demo loss")
    
    parser.add_argument("--num_demos", type=int, default=100, help="Number of demonstration puzzles to use")
    
    # Reward shaping weights
    parser.add_argument("--w_correct", type=float, default=2.0, help="Weight for correct group")
    parser.add_argument("--w_first", type=float, default=0.1, help="Weight for 1st order cohesion")
    parser.add_argument("--w_second", type=float, default=0.05, help="Weight for 2nd order cohesion")
    
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
    
    # Reward Weights
    reward_weights = {
        "correctness": args.w_correct,
        "first_order": args.w_first,
        "second_order": args.w_second
    }
    
    # Init Env and Agent
    env = ConnectionsEnv(data_loader=data_loader, max_mistakes=args.max_mistakes, reward_weights=reward_weights)
    agent = DQNAgentModified(
        embedding_dim=_GEMMA_DIM,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=device
    )
    
    # 1. Build Demo Dataset
    # Limit number of demos to avoid long startup times
    demo_puzzles = train_puzzles[:args.num_demos]
    demo_transitions = build_demo_dataset(env, demo_puzzles)
    print(f"Generated {len(demo_transitions)} demonstration transitions from {len(demo_puzzles)} puzzles.")
    
    # 2. Pre-fill Replay Buffer
    for t in demo_transitions:
        agent.memory.push(*t)
        
    # 3. Pretrain
    print(f"Before pretrain, first layer weight mean: {agent.policy_net.net[0].weight.mean().item():.6f}")
    
    pretrain_on_demos(
        agent, 
        agent.memory, 
        num_steps=args.pretrain_steps, 
        batch_size=args.batch_size, 
        margin=args.margin, 
        lambda_demo=args.lambda_demo
    )
    
    print(f"After pretrain, first layer weight mean: {agent.policy_net.net[0].weight.mean().item():.6f}")
    
    # Validate Warm Start
    print("\nValidating Warm Start Performance...")
    # 1. Check performance on the demos it just saw (Memorization check)
    print("Evaluating on Training Puzzles (Memorization)...")
    train_metrics = evaluate(agent, env, demo_puzzles[:20]) # Check first 20 demos
    print(f"  Success Rate: {train_metrics['success_rate']:.2%}")
    print(f"  Avg Groups: {train_metrics['avg_groups']:.2f}")
    
    # 2. Check performance on held-out puzzles (Generalization check)
    print("Evaluating on Held-out Puzzles (Generalization)...")
    eval_metrics = evaluate(agent, env, eval_puzzles[:20])
    print(f"  Success Rate: {eval_metrics['success_rate']:.2%}")
    print(f"  Avg Groups: {eval_metrics['avg_groups']:.2f}")
    print("-" * 30)
    
    # 4. Online Training
    print("Starting Online Training...")
    pbar = tqdm(total=args.episodes, desc="Online Training")
    
    for episode in range(args.episodes):
        obs = env.reset(puzzle=random.choice(train_puzzles))
        done = False
        episode_reward = 0
        
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(obs, action_mask)
            
            next_obs, reward, done, _, info = env.step(action)
            
            next_action_mask = env.get_action_mask()
            agent.memory.push(obs, action, reward, next_obs, done, action_mask, next_action_mask)
            
            # Update (standard DQN loss, no margin for online phase usually, or reduced)
            # For simplicity, we turn off margin loss during online phase as per plan
            loss_dict = agent.update(demo_margin_loss=False)
            loss = loss_dict["loss"] if loss_dict else 0.0
            
            episode_reward += reward
            obs = next_obs
            
        pbar.set_description(f"Ep {episode} | R: {episode_reward:.2f}")
        pbar.update(1)
        
        if episode % args.eval_freq == 0:
            metrics = evaluate(agent, env, eval_puzzles)
            tqdm.write(f"Evaluation at episode {episode}:")
            tqdm.write(f"  Success Rate: {metrics['success_rate']:.2%}")
            tqdm.write(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            tqdm.write(f"  Avg Groups: {metrics['avg_groups']:.2f}")
            
    pbar.close()
    
    # Save Model
    torch.save(agent.policy_net.state_dict(), "dqn_warm_start_connections.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
