"""AlphaZero Network for Connections."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dqn.actions import NUM_ACTIONS, WORDS_PER_PUZZLE

class AlphaZeroNet(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Input: 
        # - Flattened embeddings: 16 * embedding_dim
        # - Mask: 16
        # - Mistakes left: 1
        self.input_dim = (WORDS_PER_PUZZLE * embedding_dim) + WORDS_PER_PUZZLE + 1
        
        # Shared backbone (same as DQN/PPO)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (Actor)
        # Outputs unnormalized logits for all 1820 actions
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        
        # Value head (Critic)
        # Outputs scalar value in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, embeddings, mask, mistakes):
        """
        Args:
            embeddings: (batch, 16, d)
            mask: (batch, 16)
            mistakes: (batch, 1)
            
        Returns:
            policy_logits: (batch, NUM_ACTIONS)
            value: (batch, 1)
        """
        # Flatten embeddings
        batch_size = embeddings.size(0)
        flat_embeds = embeddings.view(batch_size, -1)
        
        x = torch.cat([flat_embeds, mask, mistakes], dim=1)
        features = self.backbone(x)
        
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
