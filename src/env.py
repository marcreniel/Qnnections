"""Environment definition for the NYT-style Connections game."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

WORDS_PER_PUZZLE = 16
GROUP_SIZE = 4


@dataclass(frozen=True)
class StepResult:
    """Structured summary returned by the environment after each step."""

    result: str
    guess_words: List[str]
    best_group_label: str
    mistakes_left: int
    episode_end: str | None = None


class ConnectionsEnv:
    """Simulates the mechanics of the Connections puzzle for RL training."""

    def __init__(self, mistakes_allowed: int = 3) -> None:
        self.mistakes_allowed = mistakes_allowed
        self.words: list[str] = []
        self.group_sets: list[set[int]] = []
        self.group_labels: list[str] = []
        self.used_mask = np.zeros(WORDS_PER_PUZZLE, dtype=bool)
        self.mistakes_left = mistakes_allowed
        self.done = False
        self.puzzle: dict | None = None

    def reset(self, puzzle: dict) -> Tuple[np.ndarray, int]:
        """Reset the environment to a fresh puzzle instance."""

        if "words" not in puzzle:
            raise ValueError("Puzzle must contain a 'words' field with 16 entries.")
        if len(puzzle["words"]) != WORDS_PER_PUZZLE:
            raise ValueError("Connections puzzles must have exactly 16 words.")

        self.puzzle = puzzle
        self.words = [w.strip() for w in puzzle["words"]]
        self.group_sets, self.group_labels = self._build_group_sets(puzzle)
        self.used_mask = np.zeros(WORDS_PER_PUZZLE, dtype=bool)
        self.mistakes_left = self.mistakes_allowed
        self.done = False
        return self.used_mask.copy(), self.mistakes_left

    def step(self, action: Sequence[int]) -> Tuple[Tuple[np.ndarray, int], StepResult, bool, dict]:
        """Apply an action (4 indices) and advance the environment."""

        if self.done:
            raise RuntimeError("Episode already finished; call reset() to start a new puzzle.")
        if len(action) != GROUP_SIZE:
            raise ValueError("Actions must contain exactly 4 indices.")

        action_set = set(action)
        if len(action_set) != GROUP_SIZE:
            raise ValueError("Action indices must be unique.")
        if not all(0 <= idx < WORDS_PER_PUZZLE for idx in action_set):
            raise ValueError("Action indices must lie in [0, 15].")
        if any(self.used_mask[idx] for idx in action_set):
            raise ValueError("Cannot select words that are already solved.")

        overlaps = [len(action_set & group) for group in self.group_sets]
        best_group_idx = int(np.argmax(overlaps))
        best_overlap = overlaps[best_group_idx]

        if best_overlap == GROUP_SIZE:
            result_str = "correct"
            for idx in action_set:
                self.used_mask[idx] = True
        elif best_overlap == GROUP_SIZE - 1:
            result_str = "one_away"
            self.mistakes_left -= 1
        else:
            result_str = "wrong"
            self.mistakes_left -= 1

        episode_end: str | None = None
        if self.used_mask.all():
            self.done = True
            episode_end = "success"
        elif self.mistakes_left <= 0:
            self.done = True
            self.mistakes_left = 0
            episode_end = "failure"

        step_summary = StepResult(
            result=result_str,
            guess_words=[self.words[idx] for idx in action],
            best_group_label=self.group_labels[best_group_idx],
            mistakes_left=self.mistakes_left,
            episode_end=episode_end,
        )

        next_state = (self.used_mask.copy(), self.mistakes_left)
        info = {"result": result_str, "step": step_summary}
        if self.puzzle is not None and "id" in self.puzzle:
            info["puzzle_id"] = self.puzzle["id"]
        return next_state, step_summary, self.done, info

    def available_actions(self) -> List[Tuple[int, int, int, int]]:
        """Return all 4-combinations of currently unused word indices."""

        unused = [idx for idx, used in enumerate(self.used_mask) if not used]
        if len(unused) < GROUP_SIZE:
            return []
        combos = []
        for i in range(len(unused)):
            for j in range(i + 1, len(unused)):
                for k in range(j + 1, len(unused)):
                    for l in range(k + 1, len(unused)):
                        combos.append((unused[i], unused[j], unused[k], unused[l]))
        return combos

    def _build_group_sets(self, puzzle: dict) -> Tuple[List[set[int]], List[str]]:
        groups_raw = puzzle.get("groups")
        if not groups_raw or len(groups_raw) != WORDS_PER_PUZZLE // GROUP_SIZE:
            raise ValueError("Puzzle must define 4 groups of 4 words.")

        word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        group_sets: list[set[int]] = []
        group_labels: list[str] = []
        for i, entry in enumerate(groups_raw):
            if isinstance(entry, dict):
                members: Iterable[str] | None = entry.get("members") or entry.get("words")
                if members is None:
                    raise ValueError("Group dicts must have 'members' or 'words'.")
                label = entry.get("name") or entry.get("title") or f"group_{i}"
            elif isinstance(entry, list):
                members = entry
                label = f"group_{i}"
            else:
                raise ValueError("Group entries must be dicts or lists of words.")

            member_indices = {word_to_idx[word] for word in members}
            if len(member_indices) != GROUP_SIZE:
                raise ValueError("Each group must contain exactly 4 unique words present in the puzzle.")
            group_sets.append(member_indices)
            group_labels.append(label)
        return group_sets, group_labels
