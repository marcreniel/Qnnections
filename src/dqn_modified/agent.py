"""DQN Agent with Margin Loss for Warm Start."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

from src.dqn.agent import QNetwork, ReplayBuffer
from src.dqn.actions import NUM_ACTIONS

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'action_mask', 'next_action_mask'))

class DQNAgentModified:
    def __init__(
        self, 
        embedding_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.policy_net = QNetwork(embedding_dim).to(device)
        self.target_net = QNetwork(embedding_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        
    def select_action(self, state, action_mask, eval_mode=False):
        """
        Selects an action using epsilon-greedy strategy.
        Args:
            state: Dict with 'embeddings', 'mask', 'mistakes_left'
            action_mask: Boolean array (True=Valid)
            eval_mode: If True, greedy action (epsilon=0)
        """
        eps = 0.0 if eval_mode else self.epsilon
        
        if random.random() > eps:
            with torch.no_grad():
                embeddings = torch.FloatTensor(state["embeddings"]).unsqueeze(0).to(self.device)
                mask = torch.FloatTensor(state["mask"]).unsqueeze(0).to(self.device)
                mistakes = torch.FloatTensor([state["mistakes_left"]]).unsqueeze(0).to(self.device)
                
                q_values = self.policy_net(embeddings, mask, mistakes)
                
                # Mask invalid actions
                # Set q-values of invalid actions to -infinity so they are not selected
                torch_mask = torch.BoolTensor(action_mask).to(self.device)
                q_values[0, ~torch_mask] = float("-inf")
                
                return q_values.max(1)[1].item()
        else:
            # Random valid action
            valid_indices = np.where(action_mask)[0]
            return random.choice(valid_indices)

    def update(self, demo_margin_loss: bool = False, margin: float = 0.8, lambda_demo: float = 1.0):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Prepare batch data
        # State
        s_embeds = torch.FloatTensor(np.array([s["embeddings"] for s in batch.state])).to(self.device)
        s_masks = torch.FloatTensor(np.array([s["mask"] for s in batch.state])).to(self.device)
        s_mistakes = torch.FloatTensor(np.array([[s["mistakes_left"]] for s in batch.state])).to(self.device)
        
        # Next State
        ns_embeds = torch.FloatTensor(np.array([s["embeddings"] for s in batch.next_state])).to(self.device)
        ns_masks = torch.FloatTensor(np.array([s["mask"] for s in batch.next_state])).to(self.device)
        ns_mistakes = torch.FloatTensor(np.array([[s["mistakes_left"]] for s in batch.next_state])).to(self.device)
        
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(s_embeds, s_masks, s_mistakes)
        state_action_values = q_values.gather(1, actions)
        
        # Compute V(s') = max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(ns_embeds, ns_masks, ns_mistakes)
            # We should mask invalid actions in next state too, but for simplicity/standard DQN we often skip
            # However, in Connections, invalid actions are impossible.
            # Let's assume the environment handles termination correctly so V(terminal) = 0
            next_state_values = next_q_values.max(1)[0]
            
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
        
        # TD Loss
        loss = nn.MSELoss()(state_action_values.squeeze(1), expected_state_action_values)
        
        # Margin Loss (for demonstrations)
        if demo_margin_loss:
            # We assume ALL samples in this batch are demonstrations if this flag is on
            # Ideally we'd have a flag per transition, but for pretraining this is fine.
            
            # Q(s, a_demo)
            q_demo = state_action_values.squeeze(1)
            
            # Q(s, a_other)
            # We want max_{a != a_demo} Q(s, a)
            # Clone q_values and mask out the demo action
            q_others = q_values.clone()
            # Scatter -inf to the demo action indices
            # actions is (batch, 1)
            q_others.scatter_(1, actions, float("-inf"))
            
            q_max_other = q_others.max(dim=1)[0]
            
            # Margin loss: max(0, Q(other) + margin - Q(demo))
            margin_loss = torch.relu(q_max_other + margin - q_demo).mean()
            
            loss = loss + (lambda_demo * margin_loss)
            
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return {
            "loss": loss.item(),
            "td_loss": nn.MSELoss()(state_action_values.squeeze(1), expected_state_action_values).item(),
            "margin_loss": margin_loss.item() if demo_margin_loss else 0.0,
            "q_mean": q_values.mean().item(),
            "q_max": q_values.max().item()
        }
