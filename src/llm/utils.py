"""Utility functions for LLM PPO: Parsing and Rewards."""
import json
import re
from typing import List, Optional, Set

def normalize_word(w: str) -> str:
    return w.strip().upper()

def parse_solution(output_text: str, original_words: List[str]) -> Optional[List[List[str]]]:
    """
    Try to:
      - parse output_text as JSON
      - read 'groups' -> list of dicts with 'members'
      - ensure:
          * exactly 4 groups
          * each group has exactly 4 strings
          * after normalization, the 16 guessed words form a permutation
            of original_words (no extras, no duplicates, no missing)
      - return list of 4 lists of normalized words if valid,
        else return None
    """
    # Attempt to extract JSON if embedded in other text
    # Look for { ... }
    match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = output_text
        
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
        
    if "groups" not in data:
        return None
        
    groups = data["groups"]
    if not isinstance(groups, list) or len(groups) != 4:
        return None
        
    parsed_groups = []
    all_guessed_words = []
    
    for g in groups:
        if "members" not in g:
            return None
        members = g["members"]
        if not isinstance(members, list) or len(members) != 4:
            return None
            
        norm_members = [normalize_word(w) for w in members]
        parsed_groups.append(norm_members)
        all_guessed_words.extend(norm_members)
        
    # Validate against original words
    norm_original = sorted([normalize_word(w) for w in original_words])
    norm_guessed = sorted(all_guessed_words)
    
    if norm_original != norm_guessed:
        return None
        
    return parsed_groups

def compute_reward(
    pred_groups: Optional[List[List[str]]],
    true_groups: List[List[str]],
    strict: bool = True
) -> float:
    """
    Computes the reward for the one-shot LLM agent using the unified scheme:
    - Correct Group: +3
    - One-away (3/4): +1
    - Wrong Group: -2
    - Win Bonus (all 4 correct): +5
    - Lose Penalty (otherwise): -3
    - Per-word Reward: +0.2 per correct word
    
    Normalized by max possible score (20.2) to range roughly [-1, 1].
    """
    if pred_groups is None:
        # Invalid format is treated as a severe failure
        # Min possible score is -11 (4 wrong + lose). -1.0 is a fair proxy.
        return -1.0
        
    score = 0.0
    correct_count = 0
    
    # Convert true groups to sets for easier matching
    true_sets = [set(g) for g in true_groups]
    
    for pg in pred_groups:
        pg_set = set(pg)
        # Find best match among true groups
        best_overlap = 0
        for tg in true_sets:
            overlap = len(pg_set & tg)
            if overlap > best_overlap:
                best_overlap = overlap
        
        # Add per-word reward (skip for one_away/3 overlap)
        if best_overlap != 3:
            score += best_overlap * 0.2
        
        if best_overlap == 4:
            score += 3.0
            correct_count += 1
        elif best_overlap == 3:
            score += 1.0
        else:
            score += -2.0
            
    if correct_count == 4:
        score += 5.0 # Win Bonus
    else:
        score += -3.0 # Failure Penalty
        
    # Normalize by max possible score (4 * (3 + 0.8) + 5 = 20.2)
    return score / 20.2
