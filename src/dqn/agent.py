"""DQN Agent for Connections."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from .actions import NUM_ACTIONS, WORDS_PER_PUZZLE

class QNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Input: 
        # - Flattened embeddings: 16 * embedding_dim
        # - Mask: 16
        # - Mistakes left: 1
        self.input_dim = (WORDS_PER_PUZZLE * embedding_dim) + WORDS_PER_PUZZLE + 1
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS)
        )
        
    def forward(self, embeddings, mask, mistakes):
        # embeddings: (batch, 16, d) -> flatten -> (batch, 16*d)
        batch_size = embeddings.size(0)
        flat_embeds = embeddings.view(batch_size, -1)
        
        # mask: (batch, 16)
        # mistakes: (batch, 1)
        
        x = torch.cat([flat_embeds, mask, mistakes], dim=1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
        
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self, 
        embedding_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.q_net = QNetwork(embedding_dim).to(device)
        self.target_net = QNetwork(embedding_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
        
    def act(self, state, action_mask, epsilon: float = 0.1):
        """
        Selects an action using epsilon-greedy with masking.
        Args:
            state: Dict with 'embeddings', 'mask', 'mistakes_left'
            action_mask: Boolean array (True=Valid)
        """
        if random.random() < epsilon:
            # Random valid action
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) == 0:
                return 0 # Should not happen if game logic is correct
            return int(random.choice(valid_indices))
        
        # Greedy action
        with torch.no_grad():
            embeddings = torch.FloatTensor(state["embeddings"]).unsqueeze(0).to(self.device)
            mask = torch.FloatTensor(state["mask"]).unsqueeze(0).to(self.device)
            mistakes = torch.FloatTensor([state["mistakes_left"]]).unsqueeze(0).to(self.device)
            
            q_values = self.q_net(embeddings, mask, mistakes)
            
            # Apply mask to Q-values (set invalid to -inf)
            # action_mask is numpy bool
            torch_mask = torch.BoolTensor(action_mask).to(self.device)
            q_values[0, ~torch_mask] = float("-inf")
            
            return int(q_values.argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, _, next_action_masks = zip(*batch)
        
        # Prepare tensors
        # State unpacking
        s_embeds = torch.FloatTensor(np.array([s["embeddings"] for s in states])).to(self.device)
        s_masks = torch.FloatTensor(np.array([s["mask"] for s in states])).to(self.device)
        s_mistakes = torch.FloatTensor(np.array([[s["mistakes_left"]] for s in states])).to(self.device)
        
        ns_embeds = torch.FloatTensor(np.array([s["embeddings"] for s in next_states])).to(self.device)
        ns_masks = torch.FloatTensor(np.array([s["mask"] for s in next_states])).to(self.device)
        ns_mistakes = torch.FloatTensor(np.array([[s["mistakes_left"]] for s in next_states])).to(self.device)
        
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_masks_t = torch.BoolTensor(np.array(next_action_masks)).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.q_net(s_embeds, s_masks, s_mistakes)
        current_q = q_values.gather(1, actions_t)
        
        # Compute Target Q
        with torch.no_grad():
            next_q_values = self.target_net(ns_embeds, ns_masks, ns_mistakes)
            # Mask invalid next actions
            next_q_values[~next_masks_t] = float("-inf")
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            # If all actions invalid (e.g. done), max is -inf, but done handles it.
            # Actually if done, max_next_q doesn't matter.
            # But if not done and no valid actions (shouldn't happen), we need care.
            # Replace -inf with 0 just in case for stability if done
            max_next_q[max_next_q == float("-inf")] = 0.0
            
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()
