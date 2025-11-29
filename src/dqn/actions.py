"""Action utilities for Connections."""
import itertools
from typing import List, Tuple

import numpy as np

WORDS_PER_PUZZLE = 16
GROUP_SIZE = 4

# Pre-compute all possible combinations of 4 indices from 0 to 15
# Total combinations = 16 choose 4 = 1820
ALL_ACTIONS = list(itertools.combinations(range(WORDS_PER_PUZZLE), GROUP_SIZE))
NUM_ACTIONS = len(ALL_ACTIONS)

def get_all_actions() -> List[Tuple[int, int, int, int]]:
    """Returns the list of all possible 4-word combinations."""
    return ALL_ACTIONS

def get_action_from_idx(action_idx: int) -> Tuple[int, int, int, int]:
    """Returns the tuple of indices for a given action index."""
    return ALL_ACTIONS[action_idx]

def get_action_mask(used_mask: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask of valid actions given the current used_mask.
    True means the action is INVALID (masked out), False means valid.
    Wait, usually in RL masking: 
    - 1/True = Valid
    - 0/False = Invalid
    OR
    - -inf for Invalid in logits.
    
    Let's return a boolean mask where True = Valid action, False = Invalid action.
    
    Args:
        used_mask: Boolean array of shape (16,), True if word is already used.
    """
    # This might be slow if done naively in Python every step. 
    # But 1820 is small.
    
    # Vectorized approach?
    # We can pre-compute a matrix of shape (1820, 16) where each row is the binary mask of the action.
    # Then valid = (ActionMatrix @ used_mask) == 0
    
    # Let's do the pre-computation once at module level if possible, or lazy load.
    global _ACTION_MATRIX
    if _ACTION_MATRIX is None:
        _ACTION_MATRIX = np.zeros((NUM_ACTIONS, WORDS_PER_PUZZLE), dtype=bool)
        for i, indices in enumerate(ALL_ACTIONS):
            _ACTION_MATRIX[i, indices] = True
            
    # If any word in the action is used, the action is invalid.
    # used_mask shape: (16,)
    # _ACTION_MATRIX shape: (1820, 16)
    # We want to find rows where ANY of the used words are present.
    # Dot product (logical OR)
    
    # valid_actions = NOT (Matrix . used_mask)
    # If a word is used (1), and action uses it (1), result is 1 (invalid).
    
    is_invalid = _ACTION_MATRIX @ used_mask
    return ~is_invalid

_ACTION_MATRIX = None

def get_action_idx_from_words(words: List[str]) -> int:
    """
    Returns the action index for a given list of 4 words.
    This assumes the words are from the current puzzle and we need to find their indices.
    Wait, this function signature is tricky because we need the full list of puzzle words to map words -> indices.
    
    Actually, the caller (demo.py) has the puzzle words. It should map words -> indices first.
    So let's rename this to get_action_idx_from_indices or just use a lookup.
    
    But demo.py calls `get_action_idx_from_words(group_words)`.
    It seems demo.py logic was slightly flawed or expected this helper to do the reverse lookup.
    
    Let's change the contract: The caller should provide indices.
    But wait, `demo.py` has:
        group_words = [puzzle["words"][i] for i in group_indices]
        action_idx = get_action_idx_from_words(group_words)
        
    It ALREADY has `group_indices`! It doesn't need to look up words.
    It just needs to map `group_indices` (tuple of 4 ints) to `action_idx`.
    
    So let's implement `get_action_idx(indices: Tuple[int, ...]) -> int`.
    """
    raise NotImplementedError("Use get_action_idx instead.")

# Reverse lookup map
_ACTION_TO_IDX = {action: i for i, action in enumerate(ALL_ACTIONS)}

def get_action_idx(indices: Tuple[int, int, int, int]) -> int:
    """Returns the action index for a tuple of 4 sorted indices."""
    return _ACTION_TO_IDX[tuple(sorted(indices))]
