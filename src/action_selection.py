"""Helpers for generating candidate actions (word group guesses)."""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

Action = Tuple[int, int, int, int]


def enumerate_all_actions(unused_indices: Sequence[int]) -> List[Action]:
    """Enumerate every 4-combination from the unused indices."""

    return [tuple(combo) for combo in combinations(unused_indices, 4)]


def candidate_actions(used_mask: Sequence[bool]) -> List[Action]:
    """Return every 4-word action available under the current mask."""

    unused_indices = [idx for idx, used in enumerate(used_mask) if not used]
    if len(unused_indices) < 4:
        return []
    return enumerate_all_actions(unused_indices)
