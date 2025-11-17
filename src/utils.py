"""Utility helpers for loading puzzles, managing masks, and seeding RNGs."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, List, Sequence
import json
import math
import random

import numpy as np

WORDS_PER_PUZZLE = 16
GROUP_SIZE = 4


def load_puzzles(json_path: str | Path) -> List[dict]:
    """Load the Connections puzzles JSON file and normalize legacy formats."""

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        puzzles = data.get("puzzles")
        if puzzles is None:
            raise ValueError("JSON object must contain a 'puzzles' list.")
        raw_list = list(puzzles)
    elif isinstance(data, list):
        raw_list = list(data)
    else:
        raise ValueError("Puzzle JSON must be a list or an object with 'puzzles'.")

    normalized: list[dict] = []
    for idx, entry in enumerate(raw_list):
        try:
            normalized.append(_normalize_puzzle(entry))
        except ValueError as exc:
            puzzle_id = entry.get("id", idx)
            warnings.warn(f"Skipping puzzle {puzzle_id}: {exc}")
    if not normalized:
        raise ValueError("No valid puzzles could be loaded from dataset.")
    return normalized


def _normalize_puzzle(entry: dict) -> dict:
    """Ensure every puzzle has 16 `words` and 4 `groups` entries."""

    if "words" in entry and "groups" in entry:
        if len(entry["words"]) != WORDS_PER_PUZZLE:
            raise ValueError("Connections puzzles must have exactly 16 words.")
        return entry

    answers = entry.get("answers")
    if not answers:
        raise ValueError("Puzzle entry must include 'words' or 'answers'.")

    groups: list[dict] = []
    words: list[str] = []
    for idx, info in enumerate(answers):
        members = info.get("members") or info.get("words")
        if members is None:
            raise ValueError("Each group must provide 4 members.")
        cleaned = [str(word).strip().upper() for word in members if str(word).strip()]
        if len(cleaned) != GROUP_SIZE:
            raise ValueError("Each group must provide 4 non-empty members.")
        label = info.get("group") or info.get("name") or f"group_{idx}"
        groups.append({"name": label, "members": cleaned})
        words.extend(cleaned)

    if len(words) != WORDS_PER_PUZZLE:
        raise ValueError("Derived puzzle does not contain 16 words.")

    normalized = dict(entry)
    normalized["words"] = words
    normalized["groups"] = groups
    return normalized


def mask_to_int(mask: Sequence[bool]) -> int:
    """Convert a boolean mask (length 16) to an integer bitmask."""

    if len(mask) != WORDS_PER_PUZZLE:
        raise ValueError("Mask must be length 16.")
    value = 0
    for idx, flag in enumerate(mask):
        if flag:
            value |= 1 << idx
    return value


def int_to_mask(mask_int: int, length: int = WORDS_PER_PUZZLE) -> np.ndarray:
    """Convert an integer bitmask back to a boolean numpy array."""

    mask = np.zeros(length, dtype=bool)
    for idx in range(length):
        mask[idx] = bool(mask_int & (1 << idx))
    return mask


def remaining_indices(mask: Sequence[bool]) -> List[int]:
    """Return the indices that are still unused according to the mask."""

    return [idx for idx, flag in enumerate(mask) if not flag]


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and torch (if available) RNGs for reproducibility."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def decay_schedule(initial_value: float, visit_count: int) -> float:
    """1/sqrt(n) decay helper for learning-rate style schedules."""

    if visit_count <= 0:
        return initial_value
    return initial_value / math.sqrt(visit_count)
