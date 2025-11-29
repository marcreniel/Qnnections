"""Training script for PPO Warm Start on Connections."""
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from src.common.data_loader import DataLoader
from src.dqn_modified.env import ConnectionsEnv
from src.ppo_modified.agent import ActorCritic, PPOAgent
from src.ppo_modified.demo import build_demo_dataset
from src.ppo_modified.env_utils import encode_state, get_action_mask
from src.common.embeddings import _GEMMA_DIM
from src.dqn.actions import NUM_ACTIONS, ALL_ACTIONS

def evaluate(actor_critic, env, puzzles, num_episodes: int = 20):
    """Evaluates the agent on a set of puzzles."""
    total_reward = 0
    successes = 0
    total_groups = 0
    
    eval_puzzles = puzzles[:num_episodes]
    if len(eval_puzzles) < num_episodes:
        eval_puzzles = (eval_puzzles * (num_episodes // len(eval_puzzles) + 1))[:num_episodes]
        
    for puzzle in eval_puzzles:
        env.reset(puzzle=puzzle)
        done = False
        episode_reward = 0
        groups_found = 0
        
        while not done:
            state_vec = encode_state(env)
            action_mask = get_action_mask(env)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(next(actor_critic.parameters()).device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(next(actor_critic.parameters()).device)
            
            with torch.no_grad():
                action_idx, _, _ = actor_critic.act(state_tensor, mask_tensor, greedy=True)
                
            action = action_idx.item()
            _, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            if info["result"] == "correct":
                groups_found += 1
                
        total_reward += episode_reward
        if info["episode_end"] == "success":
            successes += 1
        total_groups += groups_found
        
    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes,
        "avg_groups": total_groups / num_episodes
    }

def behavior_cloning_train(actor_critic, demo_dataset, num_epochs, batch_size, device):
    print(f"\nStarting Behavior Cloning (Policy) for {num_epochs} epochs...")
    
    # Prepare data
    states = torch.tensor(np.array([d['state'] for d in demo_dataset]), dtype=torch.float32)
    actions = torch.tensor(np.array([d['action'] for d in demo_dataset]), dtype=torch.long)
    
    dataset = TensorDataset(states, actions)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    
    actor_critic.train()
    actor_critic.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_states, batch_actions in loader:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
            
            logits, _ = actor_critic(batch_states)
            loss = F.cross_entropy(logits, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / len(loader):.4f}")

def critic_pretrain(actor_critic, demo_dataset, num_epochs, batch_size, device):
    print(f"\nStarting Critic Pretraining (Value) for {num_epochs} epochs...")
    
    states = torch.tensor(np.array([d['state'] for d in demo_dataset]), dtype=torch.float32)
    returns = torch.tensor(np.array([d['return'] for d in demo_dataset]), dtype=torch.float32)
    
    dataset = TensorDataset(states, returns)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    
    actor_critic.train()
    actor_critic.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_states, batch_returns in loader:
            batch_states, batch_returns = batch_states.to(device), batch_returns.to(device)
            
            _, values = actor_critic(batch_states)
            loss = F.mse_loss(values, batch_returns)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / len(loader):.4f}")

def collect_trajectories(env, agent, num_episodes, max_steps, puzzles, device):
    trajectories = []
    
    for _ in range(num_episodes):
        puzzle = random.choice(puzzles)
        env.reset(puzzle=puzzle)
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            state_vec = encode_state(env)
            action_mask = get_action_mask(env)
            
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_idx, log_prob, value = agent.actor_critic.act(state_tensor, mask_tensor, greedy=False)
                
            action = action_idx.item()
            _, reward, done, _, _ = env.step(action)
            
            trajectories.append({
                "state": state_vec,
                "action": action,
                "reward": reward,
                "log_prob": log_prob.item(),
                "value": value.item(),
                "done": done
            })
            steps += 1
            
    return trajectories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json")
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--bc_epochs", type=int, default=20)
    parser.add_argument("--critic_epochs", type=int, default=20)
    parser.add_argument("--ppo_episodes", type=int, default=1000)
    parser.add_argument("--max_mistakes", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--collect_episodes", type=int, default=10)
    
    # Reward weights
    parser.add_argument("--w_correct", type=float, default=2.0)
    parser.add_argument("--w_first", type=float, default=0.1)
    parser.add_argument("--w_second", type=float, default=0.05)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data
    data_loader = DataLoader(args.data_path, embedding_source="gemma")
    data_loader.load_data()
    
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
    
    # Init Env
    env = ConnectionsEnv(data_loader=data_loader, max_mistakes=args.max_mistakes, reward_weights=reward_weights)
    
    # Build Demo Dataset
    demo_dataset = build_demo_dataset(env, train_puzzles, args.num_demos)
    
    # Init Agent
    state_dim = (_GEMMA_DIM * 16) + 16 + 1
    actor_critic = ActorCritic(state_dim, NUM_ACTIONS)
    
    # 1. BC Pretraining
    behavior_cloning_train(actor_critic, demo_dataset, args.bc_epochs, 64, device)
    
    # Validate BC
    print("Validating BC Policy...")
    metrics = evaluate(actor_critic, env, train_puzzles[:20])
    print(f"BC Success Rate: {metrics['success_rate']:.2%}")
    
    # 2. Critic Pretraining
    critic_pretrain(actor_critic, demo_dataset, args.critic_epochs, 64, device)
    
    # 3. PPO Training
    print("\nStarting PPO Training...")
    ppo_agent = PPOAgent(actor_critic, device=device, bc_coef=0.2)
    
    pbar = tqdm(range(0, args.ppo_episodes, args.collect_episodes))
    for ep_start in pbar:
        trajectories = collect_trajectories(env, ppo_agent, args.collect_episodes, 20, train_puzzles, device)
        
        # Anneal BC coef
        if ep_start > 200:
            ppo_agent.bc_coef = 0.0
            
        loss = ppo_agent.update(trajectories, demo_dataset)
        pbar.set_description(f"Loss: {loss:.4f}")
        
        if ep_start % args.eval_freq == 0:
            metrics = evaluate(ppo_agent.actor_critic, env, eval_puzzles)
            tqdm.write(f"Eval @ {ep_start}: Success={metrics['success_rate']:.2%} | Groups={metrics['avg_groups']:.2f}")
            
    torch.save(actor_critic.state_dict(), "ppo_warm_start_connections.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
