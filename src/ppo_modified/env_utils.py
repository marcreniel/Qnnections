"""Environment utilities for PPO Warm-Start."""
import numpy as np
import itertools
from src.dqn.actions import ALL_ACTIONS, NUM_ACTIONS, WORDS_PER_PUZZLE, get_action_mask as dqn_get_action_mask

def encode_state(env) -> np.ndarray:
    """
    Returns a 1D float32 vector representing the current state.
    [flattened embeddings (16*d), used_mask (16,), mistakes_left_scalar (1,)]
    """
    # Get observation from env
    # If env is just the ConnectionsEnv instance, it has _get_obs() but that returns dict.
    # We can access env.embeddings, env.used_mask, env.mistakes_left directly if we are sure.
    # But better to use the public interface if possible.
    # The env.step() returns obs dict.
    
    # Let's assume we are passing the env instance itself.
    embeddings = env.embeddings # (16, d)
    mask = env.used_mask # (16,)
    mistakes = env.mistakes_left
    
    # Flatten embeddings
    flat_embeds = embeddings.flatten()
    
    # Convert mask to float
    mask_float = mask.astype(np.float32)
    
    # Mistakes as array
    mistakes_arr = np.array([mistakes], dtype=np.float32)
    
    # Concatenate
    return np.concatenate([flat_embeds, mask_float, mistakes_arr])

def get_action_mask(env) -> np.ndarray:
    """
    Returns a boolean mask of shape [num_actions] where False marks illegal actions.
    Note: The DQN implementation returns True for VALID actions (usually).
    Wait, DQN implementation:
    dqn_get_action_mask returns ~is_invalid. So True = Valid.
    
    The user spec says: "False marks illegal actions".
    So True = Valid.
    """
    return dqn_get_action_mask(env.used_mask)
