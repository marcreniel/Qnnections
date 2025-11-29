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
    Strict version:
      - If pred_groups is None -> reward = -1.0 (invalid format)
      - Else if partition exactly matches -> 1.0
      - Else -> 0.0
      
    Shaped version (if strict=False):
      - Returns fraction of correct groups (0.0 to 1.0)
    """
    if pred_groups is None:
        return -1.0
        
    pred_set = {frozenset(g) for g in pred_groups}
    true_set = {frozenset(g) for g in true_groups}
    
    if strict:
        if pred_set == true_set:
            return 1.0
        else:
            return 0.0
    else:
        # Shaped reward: fraction of correct groups
        correct_count = len(pred_set & true_set)
        return correct_count / 4.0
