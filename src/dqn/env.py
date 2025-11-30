"""Gymnasium-compatible environment for Connections."""
from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np

from .actions import get_action_from_idx, get_action_mask, NUM_ACTIONS, WORDS_PER_PUZZLE, GROUP_SIZE
from .reward import compute_reward
from src.common.data_loader import DataLoader

class ConnectionsEnv:
    """
    RL Environment for Connections.
    
    Observation:
        - embeddings: (16, d) float32
        - mask: (16,) bool (0=available, 1=used)
        - mistakes_left: int
        
    Action:
        - Discrete(1820) - index into all 4-word combinations.
    """
    
    def __init__(self, data_loader: DataLoader = None, mistakes_allowed: int = 4, reward_weights: Dict[str, float] = None):
        self.data_loader = data_loader
        self.mistakes_allowed = mistakes_allowed
        self.reward_weights = reward_weights or {
            "correctness": 1.0,
            "first_order": 0.1,
            "second_order": 0.05
        }
        
        self.current_puzzle = None
        self.embeddings = None
        self.used_mask = None
        self.mistakes_left = 0
        self.done = True
        
        # Cache for current puzzle's group sets for fast checking
        self.group_sets = []
        
    def reset(self, puzzle: Dict = None, seed: int = None) -> Dict[str, Any]:
        """
        Resets the environment.
        Args:
            puzzle: Optional puzzle dict. If None, samples from data_loader.
        """
        if puzzle is None:
            if self.data_loader and self.data_loader.puzzles:
                import random
                if seed is not None:
                    random.seed(seed)
                puzzle = random.choice(self.data_loader.puzzles)
            else:
                raise ValueError("No puzzle provided and no data_loader available.")
                
        self.current_puzzle = puzzle
        self.mistakes_left = self.mistakes_allowed
        self.used_mask = np.zeros(WORDS_PER_PUZZLE, dtype=bool) # 0 = available
        self.done = False
        
        # Pre-compute embeddings if not already in puzzle
        if "embeddings" in puzzle:
            self.embeddings = puzzle["embeddings"]
        elif self.data_loader:
            self.embeddings = self.data_loader.get_embeddings(puzzle["words"])
        else:
            # Fallback or error
            raise ValueError("Puzzle has no embeddings and no data_loader to generate them.")
            
        # Build group sets for checking
        # puzzle["group_ids"] is a list of 16 ints (0-3)
        # We want sets of indices for each group
        self.group_sets = [set() for _ in range(4)]
        for idx, g_id in enumerate(puzzle["group_ids"]):
            self.group_sets[g_id].add(idx)
            
        return self._get_obs()

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes an action.
        Args:
            action_idx: Index into ALL_ACTIONS (0-1819)
        """
        if self.done:
            raise RuntimeError("Step called on finished episode.")
            
        # Decode action
        indices = get_action_from_idx(action_idx)
        
        # Validate action (check if any word is already used)
        # In a proper RL loop, the agent should mask these out.
        # If the agent picks an invalid action, we can either:
        # 1. Return large negative reward and end episode (strict)
        # 2. Return large negative reward and continue (lenient)
        # 3. Ignore it (but DQN needs a transition).
        # Let's assume masking works, but if not, penalize heavily.
        
        if any(self.used_mask[i] for i in indices):
            # Invalid action selected!
            reward = -5
            info = {"result": "invalid", "message": "Selected used words"}
            return self._get_obs(), reward, self.done, False, info
            
        # Check guess
        guess_set = set(indices)
        
        # Find best overlap
        best_overlap = 0
        best_group_id = -1
        
        for g_id, g_set in enumerate(self.group_sets):
            # We only care about overlap with REMAINING words in that group?
            # No, the group definition is static. 
            # But if we already found a group, its words are used.
            # So we just check overlap with the static groups.
            overlap = len(guess_set & g_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_group_id = g_id
                
        # Determine outcome
        result_str = "wrong"
        if best_overlap == 4:
            result_str = "correct"
            # Mark words as used
            for i in indices:
                self.used_mask[i] = True
        elif best_overlap == 3:
            result_str = "one_away"
            self.mistakes_left -= 1
        else:
            result_str = "wrong"
            self.mistakes_left -= 1
            
        # Check termination
        episode_end = None
        if np.all(self.used_mask):
            episode_end = "success"
            self.done = True
        elif self.mistakes_left <= 0:
            episode_end = "failure"
            self.done = True
            
        # Compute Reward
        # We need the actual words for semantic reward
        guess_words = [self.current_puzzle["words"][i] for i in indices]
        
        reward = compute_reward(
            result_str=result_str,
            episode_end=episode_end,
            group_words=guess_words,
            embed_source=self.data_loader.embedding_source if self.data_loader else "gemma",
            weights=self.reward_weights,
            overlap_count=best_overlap
        )
        
        info = {
            "result": result_str,
            "best_overlap": best_overlap,
            "mistakes_left": self.mistakes_left,
            "episode_end": episode_end
        }
        
        return self._get_obs(), reward, self.done, False, info

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings, # (16, d)
            "mask": self.used_mask.copy(), # (16,)
            "mistakes_left": self.mistakes_left
        }
        
    def get_action_mask(self) -> np.ndarray:
        """Returns the valid action mask for the current state."""
        return get_action_mask(self.used_mask)
