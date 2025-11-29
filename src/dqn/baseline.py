"""Heuristic baseline for Connections."""
import numpy as np
from typing import List, Tuple
from src.common.embeddings import _cosine_similarity, get_word_embedding

def solve_heuristic(words: List[str], embedding_source: str = "gemma") -> List[List[str]]:
    """
    Greedy heuristic:
    1. Compute embeddings for all remaining words.
    2. Find group of 4 with highest cohesion.
    3. Guess it.
    4. Repeat until solved or mistakes exhausted (though this function just proposes groups).
    """
    
    # This function simulates the agent's logic.
    # In the env loop, we need to pick an action index.
    pass

class HeuristicAgent:
    def __init__(self, embedding_source: str = "gemma"):
        self.embedding_source = embedding_source
        
    def act(self, state: dict, action_mask: np.ndarray, words: List[str]) -> int:
        """
        Selects the best action based on cosine similarity.
        Args:
            state: dict with 'mask'
            action_mask: bool array
            words: list of 16 words (original order)
        """
        # Identify available indices
        available_indices = [i for i, used in enumerate(state["mask"]) if not used]
        
        if len(available_indices) < 4:
            return 0 # Should not happen
            
        # We need to find the best 4-tuple among available_indices
        # This is expensive to do exhaustively every step if we re-compute everything.
        # But N is small (at most 16). 16 choose 4 = 1820.
        
        # Get embeddings for available words
        # We can cache this if we had the object instance persist, but for now re-compute/fetch.
        embeddings = {i: get_word_embedding(words[i], self.embedding_source) for i in available_indices}
        
        best_score = -1.0
        best_action_idx = -1
        
        # Iterate over all valid actions
        # We can iterate over ALL_ACTIONS and check mask, OR iterate combinations of available_indices.
        # Iterating combinations of available is better.
        import itertools
        from .actions import ALL_ACTIONS
        
        # We need to map back to action_idx. 
        # This is tricky without a reverse map or searching.
        # Let's search in ALL_ACTIONS or use the pre-computed list if we can index it.
        # Actually, we can just iterate through all valid actions in the mask.
        
        valid_action_indices = np.where(action_mask)[0]
        
        for idx in valid_action_indices:
            # Get words indices
            indices = ALL_ACTIONS[idx]
            
            # Compute cohesion
            # Mean pairwise sim
            score = 0.0
            count = 0
            for i in range(4):
                for j in range(i+1, 4):
                    u = embeddings[indices[i]]
                    v = embeddings[indices[j]]
                    score += _cosine_similarity(u, v)
                    count += 1
            avg_score = score / count
            
            if avg_score > best_score:
                best_score = avg_score
                best_action_idx = idx
                
        return int(best_action_idx)
