"""Reward shaping utilities for the DQN agent."""
from __future__ import annotations

from typing import Sequence
import numpy as np

from src.common.embeddings import get_word_embedding, _cosine_similarity

BASE_REWARDS = {
    "correct": 3.0,
    "one_away": 1.0,
    "wrong": -2.0,
}
SUCCESS_BONUS = 5.0
FAILURE_PENALTY = -3.0
PER_WORD_REWARD = 0.2

# MAX_REWARD_MAGNITUDE = 8.0  <-- Removed normalization for DQN

def get_second_order_cohesion(words: Sequence[str], source: str) -> float:
    """
    Computes second-order cohesion: similarity of pairwise difference vectors.
    For a group of 4 words, we can look at relations like (w2-w1) vs (w4-w3).
    A high score implies analogous relationships (e.g. A is to B as C is to D).
    
    For a set of 4 words, there are multiple permutations. 
    We can compute the average cosine similarity of all pairs of difference vectors.
    """
    if len(words) < 4:
        return 0.0
        
    embeddings = [get_word_embedding(w, source) for w in words]
    
    # Compute all difference vectors (v_j - v_i)
    diffs = []
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                diffs.append(embeddings[j] - embeddings[i])
    
    # Compute average cosine similarity between these difference vectors
    # This is O(N^2) where N is number of diffs (12 for 4 words). 12^2 = 144. Fast enough.
    sims = []
    for i in range(len(diffs)):
        for j in range(i + 1, len(diffs)):
            sims.append(_cosine_similarity(diffs[i], diffs[j]))
            
    if not sims:
        return 0.0
    return float(np.mean(sims))

def compute_reward(
    result_str: str,
    episode_end: str | None,
    group_words: Sequence[str],
    embed_source: str,
    weights: dict[str, float],
    overlap_count: int = 0
) -> float:
    """
    Computes the scalar reward.
    
    Args:
        result_str: "correct", "one_away", "wrong"
        episode_end: "success", "failure", or None
        group_words: List of 4 words guessed
        embed_source: "gemma" or "glove"
        weights: Dictionary of weights for components:
            - "correctness": Weight for game feedback
            - "first_order": Weight for 1st order cohesion
            - "second_order": Weight for 2nd order cohesion
        overlap_count: Number of correct words in the guess (0-4)
    """
    
    # 1. Correctness Term
    base = BASE_REWARDS.get(result_str, 0.0)
    
    # Add per-word reward (skip for one_away as requested)
    if result_str != "one_away":
        base += overlap_count * PER_WORD_REWARD
    
    if episode_end == "success":
        base += SUCCESS_BONUS
    elif episode_end == "failure":
        base += FAILURE_PENALTY
        
    # If it's a wrong guess, return the penalty immediately (unnormalized).
    if result_str == "wrong":
        return base
        
    # If it's one_away, return neutral or base (unnormalized).
    if result_str == "one_away":
        return base

    # If correct, add semantic bonuses
    r_correctness = base
    
    # 2. Semantic Terms
    w_first = weights.get("first_order", 0.0)
    w_second = weights.get("second_order", 0.0)
    
    r_first = 0.0
    r_second = 0.0
    
    if w_first != 0 or w_second != 0:
        # We need embeddings
        from src.common.embeddings import group_cohesion
        
        if w_first != 0:
            r_first = group_cohesion(group_words, embed_source)
            
        if w_second != 0:
            r_second = get_second_order_cohesion(group_words, embed_source)
            
    total_reward = (
        weights.get("correctness", 1.0) * r_correctness +
        w_first * r_first +
        w_second * r_second
    )
    
    # Return unnormalized reward
    return total_reward
