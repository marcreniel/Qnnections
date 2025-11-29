"""PPO Agent for Connections."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict

from src.dqn.actions import NUM_ACTIONS, WORDS_PER_PUZZLE

class PPOActorCritic(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Input: 
        # - Flattened embeddings: 16 * embedding_dim
        # - Mask: 16
        # - Mistakes left: 1
        self.input_dim = (WORDS_PER_PUZZLE * embedding_dim) + WORDS_PER_PUZZLE + 1
        
        # Shared backbone (same as DQN)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (Policy logits)
        self.actor = nn.Linear(hidden_dim, NUM_ACTIONS)
        
        # Critic head (Value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings, mask, mistakes):
        # embeddings: (batch, 16, d) -> flatten -> (batch, 16*d)
        batch_size = embeddings.size(0)
        flat_embeds = embeddings.view(batch_size, -1)
        
        x = torch.cat([flat_embeds, mask, mistakes], dim=1)
        features = self.backbone(x)
        
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value

class PPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        
    def add(self, state, action, reward, done, log_prob, value, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.action_masks.append(action_mask)

class PPOAgent:
    def __init__(
        self,
        embedding_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.policy = PPOActorCritic(embedding_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = PPORolloutBuffer()
        
    def select_action(self, state, action_mask, greedy=False):
        """
        Selects an action using the current policy.
        Args:
            state: Dict with 'embeddings', 'mask', 'mistakes_left'
            action_mask: Boolean array (True=Valid)
            greedy: If True, select argmax (for eval)
        """
        with torch.no_grad():
            embeddings = torch.FloatTensor(state["embeddings"]).unsqueeze(0).to(self.device)
            mask = torch.FloatTensor(state["mask"]).unsqueeze(0).to(self.device)
            mistakes = torch.FloatTensor([state["mistakes_left"]]).unsqueeze(0).to(self.device)
            
            logits, value = self.policy(embeddings, mask, mistakes)
            
            # Mask invalid actions
            torch_mask = torch.BoolTensor(action_mask).to(self.device)
            logits[0, ~torch_mask] = float("-1e9")
            
            if greedy:
                action = logits.argmax(dim=1).item()
                return action, 0.0, value.item()
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
            
    def update(self):
        # Convert buffer to tensors
        # Note: states are dicts, need careful batching
        s_embeds = torch.FloatTensor(np.array([s["embeddings"] for s in self.buffer.states])).to(self.device)
        s_masks = torch.FloatTensor(np.array([s["mask"] for s in self.buffer.states])).to(self.device)
        s_mistakes = torch.FloatTensor(np.array([[s["mistakes_left"]] for s in self.buffer.states])).to(self.device)
        
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values
        action_masks = torch.BoolTensor(np.array(self.buffer.action_masks)).to(self.device)
        
        # Compute Returns and Advantages (GAE could be added, using simple MC for now)
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        advantages = returns - torch.FloatTensor(values).to(self.device)
        
        # PPO Update Loop
        dataset_size = len(self.buffer.states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # Mini-batch data
                mb_embeds = s_embeds[idx]
                mb_masks = s_masks[idx]
                mb_mistakes = s_mistakes[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_action_masks = action_masks[idx]
                
                # Forward pass
                logits, new_values = self.policy(mb_embeds, mb_masks, mb_mistakes)
                new_values = new_values.squeeze(1)
                
                # Mask logits
                logits[~mb_action_masks] = float("-1e9")
                
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = nn.MSELoss()(new_values, mb_returns)
                
                # Total Loss
                loss = policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        self.buffer.clear()
