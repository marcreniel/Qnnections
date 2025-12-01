"""Utility helpers for parsing model outputs and computing rewards."""

import json
import re
from typing import List, Optional

def normalize_word(w: str) -> str:
    return w.strip().upper()

def _extract_json_snippet(text: str) -> str:
    """Best-effort extraction of the JSON block inside ``text``."""

    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_solution(output_text: str, original_words: List[str]) -> Optional[List[List[str]]]:
    """Best-effort parser that extracts four groups with four words each."""
    json_str = _extract_json_snippet(output_text)

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
        
    # Relaxed validation: Just check if we have 4 groups of 4 words.
    # We no longer enforce that they are a perfect permutation of original_words.
    # This allows the model to get partial credit even if it hallucinates or repeats words.
    
    return parsed_groups

def count_correct_words(
    pred_groups: Optional[List[List[str]]],
    true_groups: List[List[str]],
) -> int:
    """Count how many words are in their exact ground-truth group."""

    if pred_groups is None:
        return 0

    normalized_true = [set(normalize_word(word) for word in group) for group in true_groups]
    matched_true = [False] * len(normalized_true)
    correct_words = 0

    for group in pred_groups:
        if len(group) != 4:
            continue
        normalized_group = set(normalize_word(word) for word in group)
        match_idx = None
        for idx, (true_set, already_matched) in enumerate(zip(normalized_true, matched_true)):
            if already_matched:
                continue
            if normalized_group == true_set:
                match_idx = idx
                break
        if match_idx is not None:
            matched_true[match_idx] = True
            correct_words += 4

    return correct_words


def compute_reward(
    pred_groups: Optional[List[List[str]]],
    true_groups: List[List[str]],
) -> float:
    """Smooth reward scaled to [-1/3, 1] based on correct words."""

    correct_words = count_correct_words(pred_groups, true_groups)
    reward = (correct_words - 4) / 12.0
    return max(-1.0 / 3.0, min(1.0, reward))
