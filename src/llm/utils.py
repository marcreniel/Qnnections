"""Utility helpers for parsing model outputs and computing rewards."""

from dataclasses import dataclass
import json
import math
import re
from typing import List, Optional, Sequence, Tuple


@dataclass
class RewardSettings:
    """Configurable knobs for Connections rewards.

    reward_stage curriculum:
        1 = structure only (valid JSON + 4x4 boards)
        2 = word coverage (correct words anywhere)
        3 = group correctness + strong win bonus (full solves)
    """

    reward_stage: int = 3
    invalid_penalty: float = -1.0
    json_bonus: float = 0.2
    shape_bonus: float = 0.2
    uniqueness_bonus: float = 0.2
    stage2_scale: float = 1.5
    stage3_scale: float = 2.0
    coverage_power: float = 1.6
    stage2_word_weight: float = 0.7
    stage3_word_weight: float = 0.4


@dataclass
class GameSimulationResult:
    solved_groups: int
    mistakes: int
    success: bool


@dataclass
class GroupGuess:
    members: List[str]
    theme: Optional[str] = None


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
        return _fallback_parse(output_text, original_words)
        
    if "groups" not in data:
        return None
        
    groups = data["groups"]
    if not isinstance(groups, list) or len(groups) != 4:
        return None
        
    normalized_original = {normalize_word(word) for word in original_words} if original_words else None
    parsed_groups = []
    all_guessed_words: List[str] = []
    
    for g in groups:
        if "members" not in g:
            return None
        members = g["members"]
        if not isinstance(members, list) or len(members) != 4:
            return None
            
        norm_members = [normalize_word(w) for w in members]
        if normalized_original:
            for word in norm_members:
                if word not in normalized_original:
                    return None
        parsed_groups.append(norm_members)
        all_guessed_words.extend(norm_members)
        
    if len(set(all_guessed_words)) != len(all_guessed_words):
        return None
    if normalized_original and not set(all_guessed_words) <= normalized_original:
        return None
    return parsed_groups


def _fallback_parse(output_text: str, original_words: List[str]) -> Optional[List[List[str]]]:
    """Heuristic parser that extracts words in order even if JSON is broken."""

    if not original_words:
        return None

    normalized_original = {normalize_word(word) for word in original_words}
    tokens = re.findall(r"[A-Za-z']+", output_text.upper())
    filtered = [tok for tok in tokens if tok in normalized_original]
    if len(filtered) < 16:
        return None

    groups: List[List[str]] = []
    for idx in range(0, len(filtered), 4):
        chunk = filtered[idx : idx + 4]
        if len(chunk) < 4:
            break
        groups.append(chunk)
        if len(groups) == 4:
            break

    if len(groups) != 4:
        return None
    flat = [word for group in groups for word in group]
    if len(set(flat)) != len(flat):
        return None
    return groups


def _flatten(pred_groups: List[List[str]]) -> List[str]:
    return [normalize_word(word) for group in pred_groups for word in group]

def count_correct_words(
    pred_groups: Optional[List[List[str]]],
    true_groups: Sequence[Sequence[str]],
) -> int:
    """Count how many unique true words appear anywhere in the prediction."""

    if not pred_groups:
        return 0

    true_set = {normalize_word(word) for group in true_groups for word in group}
    pred_set = {normalize_word(word) for group in pred_groups for word in group}
    return len(true_set & pred_set)


def count_full_groups(
    pred_groups: Optional[List[List[str]]],
    true_groups: Sequence[Sequence[str]],
) -> int:
    """Return how many predicted groups exactly match a true group."""

    if not pred_groups:
        return 0

    true_sets = [set(normalize_word(word) for word in group) for group in true_groups]
    pred_sets = [set(normalize_word(word) for word in group) for group in pred_groups]
    matched = [False] * len(true_sets)
    full = 0
    for ps in pred_sets:
        if len(ps) != 4:
            continue
        for idx, (ts, already) in enumerate(zip(true_sets, matched)):
            if not already and ps == ts:
                matched[idx] = True
                full += 1
                break
    return full


def simulate_nyt_game(
    pred_groups: Optional[List[List[str]]],
    true_groups: Sequence[Sequence[str]],
    max_mistakes: int = 4,
) -> GameSimulationResult:
    """Simulate the NYT Connections rules for a sequence of guesses.

    Words that belong to solved groups are removed from the available bank.
    Any incorrect guess increments the mistake counter; reaching ``max_mistakes``
    is an immediate failure.
    """

    max_mistakes = max(1, max_mistakes)
    if not pred_groups:
        return GameSimulationResult(0, max_mistakes, False)

    remaining_groups = [set(normalize_word(word) for word in group) for group in true_groups]
    target_total = len(remaining_groups)
    available_words = set(word for group in remaining_groups for word in group)

    solved = 0
    mistakes = 0

    for guess in pred_groups:
        guess_set = {normalize_word(word) for word in guess}

        if len(guess_set) != 4 or not guess_set:
            mistakes += 1
        elif not guess_set <= available_words:
            mistakes += 1
        else:
            match_idx = -1
            for idx, true_set in enumerate(remaining_groups):
                if guess_set == true_set:
                    match_idx = idx
                    break
            if match_idx != -1:
                solved += 1
                for word in guess_set:
                    available_words.discard(word)
                del remaining_groups[match_idx]
            else:
                mistakes += 1

        if solved == target_total:
            break
        if mistakes >= max_mistakes:
            mistakes = max_mistakes
            break

    success = solved == target_total and mistakes < max_mistakes
    return GameSimulationResult(solved, mistakes, success)
def _extract_words_from_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.upper())


def _members_from_json(value) -> Optional[List[str]]:  # type: ignore[no-untyped-def]
    if not isinstance(value, list):
        return None
    members: List[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        members.append(normalize_word(item))
    return members


def parse_group_guess(output_text: str, allowed_words: Sequence[str]) -> Optional[GroupGuess]:
    """Parse a single Connections group guess from ``output_text``."""

    allowed_set = {normalize_word(word) for word in allowed_words}
    if not allowed_set:
        return None

    json_str = _extract_json_snippet(output_text)
    theme: Optional[str] = None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = None

    members: Optional[List[str]] = None

    if isinstance(data, dict):
        if "guess" in data and isinstance(data["guess"], dict):
            guess_obj = data["guess"]
            members = _members_from_json(guess_obj.get("members"))
            theme_val = guess_obj.get("theme")
            if isinstance(theme_val, str):
                theme = theme_val.strip()
        if members is None and "members" in data:
            members = _members_from_json(data.get("members"))
        if members is None and "groups" in data and isinstance(data["groups"], list):
            for group in data["groups"]:
                if isinstance(group, dict) and "members" in group:
                    candidate = _members_from_json(group["members"])
                    if candidate:
                        members = candidate
                        theme_val = group.get("theme")
                        if isinstance(theme_val, str):
                            theme = theme_val.strip()
                        break

    if not members:
        tokens = [tok for tok in _extract_words_from_text(output_text) if tok in allowed_set]
        deduped: List[str] = []
        for tok in tokens:
            if tok not in deduped:
                deduped.append(tok)
            if len(deduped) == 4:
                break
        members = deduped

    filtered = [word for word in members if word in allowed_set]
    unique_members: List[str] = []
    for word in filtered:
        if word not in unique_members:
            unique_members.append(word)

    if len(unique_members) < 4:
        return None

    return GroupGuess(unique_members[:4], theme)


def _structure_bonuses(
    pred_groups: Optional[List[List[str]]],
    config: RewardSettings,
) -> Tuple[float, bool]:
    """Return (bonus, structurally_valid) for the candidate groups."""

    if not pred_groups:
        return 0.0, False

    bonus = 0.0
    structurally_valid = True

    if len(pred_groups) == 4:
        bonus += 0.5 * config.shape_bonus
    else:
        structurally_valid = False

    all_size_four = all(len(group) == 4 for group in pred_groups)
    if all_size_four:
        bonus += 0.5 * config.shape_bonus
    else:
        structurally_valid = False

    flat = [word for group in pred_groups for word in group]
    if len(flat) == 16 and len(set(flat)) == 16:
        bonus += config.uniqueness_bonus

    bonus += config.json_bonus
    return bonus, structurally_valid


def _smooth_progress(value: float, power: float) -> float:
    value = max(0.0, min(1.0, value))
    power = max(1.0, power)
    return 1.0 - math.pow(1.0 - value, power)


def compute_reward(
    pred_groups: Optional[List[List[str]]],
    true_groups: Sequence[Sequence[str]],
    config: RewardSettings | None = None,
) -> float:
    """Solve-aware reward for Connections puzzles."""

    if config is None:
        config = RewardSettings()

    stage = max(1, min(3, config.reward_stage))
    bonus, structurally_valid = _structure_bonuses(pred_groups, config)

    if not structurally_valid:
        reward = config.invalid_penalty + bonus
        return max(-1.0, min(1.0, reward))

    correct_words = count_correct_words(pred_groups, true_groups)
    full_groups = count_full_groups(pred_groups, true_groups)
    word_score = correct_words / 16.0
    group_score = full_groups / 4.0

    if stage == 1:
        reward = config.invalid_penalty + bonus
        return max(-1.0, min(1.0, reward))

    if stage == 2:
        coverage = config.stage2_word_weight * word_score + (1.0 - config.stage2_word_weight) * group_score
        smooth = _smooth_progress(coverage, config.coverage_power)
        reward = bonus + config.stage2_scale * smooth
        return max(-1.0, min(1.0, reward))

    coverage = config.stage3_word_weight * word_score + (1.0 - config.stage3_word_weight) * group_score
    smooth = _smooth_progress(coverage, config.coverage_power)
    reward = bonus + config.stage3_scale * smooth

    if full_groups == 4:
        reward = 1.0 + 0.5 * bonus

    return max(-1.0, min(1.0, reward))


def _test_reward() -> None:
    true = [
        ["A", "B", "C", "D"],
        ["E", "F", "G", "H"],
        ["I", "J", "K", "L"],
        ["M", "N", "O", "P"],
    ]
    cfg = RewardSettings(reward_stage=3)
    assert compute_reward(true, true, cfg) == 1.0
    assert compute_reward([["A", "B"], ["C", "D"]], true, cfg) < -0.2
    partial = [true[0], true[1], ["I", "X", "Y", "Z"], ["M", "Q", "R", "S"]]
    assert compute_reward(partial, true, cfg) > -0.1
    game = simulate_nyt_game(true, true)
    assert game.success and game.mistakes == 0
    fail_game = simulate_nyt_game([["A", "B", "C", "X"]], true)
    assert not fail_game.success


if __name__ == "__main__":
    _test_reward()
