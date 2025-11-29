"""PPO Agent with Warm-Start and BC Regularization."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)  # logits
        self.value_head = nn.Linear(hidden_dim, 1)             # scalar V(s)

    def forward(self, state_tensor):
        """
        state_tensor: [batch, state_dim]
        Returns: logits [batch, num_actions], values [batch]
        """
        x = self.shared(state_tensor)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def get_logits_and_value(self, state_tensor):
        return self.forward(state_tensor)

    def act(self, state_tensor, action_mask=None, greedy: bool = False):
        """
        - If greedy=False: sample from masked Categorical for PPO rollouts.
        - If greedy=True: take argmax over masked logits for evaluation.
        Returns: action_idx, log_prob, value
        """
        logits, value = self.forward(state_tensor)
        
        if action_mask is not None:
            # Mask illegal actions
            # action_mask is boolean tensor where True = Valid
            # We set invalid to -1e9
            logits = logits.clone()
            logits[~action_mask] = -1e9
            
        if greedy:
            action_idx = logits.argmax(dim=-1)
            log_prob = torch.zeros_like(action_idx, dtype=torch.float) # Dummy
            return action_idx, log_prob, value
        else:
            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            return action_idx, log_prob, value

class PPOAgent:
    def __init__(
        self,
        actor_critic: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        bc_coef: float = 0.2,
        device: str = "cpu"
    ):
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.bc_coef = bc_coef
        self.device = device
        
    def compute_gae(self, rewards, values, dones):
        """
        rewards, values, dones: lists or tensors for a trajectory
        returns advantages and returns (targets for V)
        """
        advantages = []
        last_gae_lam = 0
        
        # Append 0 for value of terminal state
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae_lam = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae_lam
            last_gae_lam = gae_lam
            advantages.insert(0, gae_lam)
            
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, trajectories, demo_dataset=None):
        # Flatten trajectories
        states = torch.tensor(np.array([t['state'] for t in trajectories]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t['action'] for t in trajectories]), dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(np.array([t['log_prob'] for t in trajectories]), dtype=torch.float32).to(self.device)
        rewards = [t['reward'] for t in trajectories]
        values = [t['value'] for t in trajectories]
        dones = [t['done'] for t in trajectories]
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Epochs
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                logits, values_pred = self.actor_critic(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # BC Regularization
                if self.bc_coef > 0.0 and demo_dataset is not None:
                    # Sample from demo dataset
                    demo_batch = np.random.choice(demo_dataset, size=self.mini_batch_size)
                    demo_states = torch.tensor(np.array([d['state'] for d in demo_batch]), dtype=torch.float32).to(self.device)
                    demo_actions = torch.tensor(np.array([d['action'] for d in demo_batch]), dtype=torch.long).to(self.device)
                    
                    demo_logits, _ = self.actor_critic(demo_states)
                    bc_loss = F.cross_entropy(demo_logits, demo_actions)
                    
                    loss = loss + self.bc_coef * bc_loss
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        return loss.item()
