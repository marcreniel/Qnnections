"""Tabular Q-learning agent used to solve Connections puzzles."""
from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .utils import mask_to_int

StateKey = Tuple[int, int]
Action = Tuple[int, int, int, int]


class QLearningAgent:
    """Simple ε-greedy Q-learning agent for the Connections environment."""

    def __init__(
        self,
        gamma: float = 0.95,
        alpha: float = 0.5,
        epsilon: float = 0.2,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.999,
        lr_decay: bool = True,
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.lr_decay = lr_decay

        self.q_table: Dict[Tuple[StateKey, Action], float] = defaultdict(float)
        self.visit_counts: Dict[Tuple[StateKey, Action], int] = defaultdict(int)

    @staticmethod
    def encode_state(state: Tuple[Sequence[bool], int]) -> StateKey:
        used_mask, mistakes_left = state
        mask_int = mask_to_int(used_mask)
        return mask_int, int(mistakes_left)

    @staticmethod
    def encode_action(action: Sequence[int]) -> Action:
        return tuple(sorted(int(idx) for idx in action))  # type: ignore[return-value]

    def _q_value(self, state_key: StateKey, action_key: Action) -> float:
        return self.q_table[(state_key, action_key)]

    def select_action(
        self,
        state: Tuple[Sequence[bool], int],
        available_actions: Sequence[Sequence[int]],
        explore: bool = True,
    ) -> Action:
        if not available_actions:
            raise ValueError("No available actions to select from.")

        state_key = self.encode_state(state)

        if explore and random.random() < self.epsilon:
            choice = random.choice(available_actions)
            return self.encode_action(choice)

        # Greedy selection with tie-breaking.
        best_q = float("-inf")
        best_actions: list[Action] = []
        for action in available_actions:
            action_key = self.encode_action(action)
            q_val = self._q_value(state_key, action_key)
            if q_val > best_q:
                best_q = q_val
                best_actions = [action_key]
            elif q_val == best_q:
                best_actions.append(action_key)
        return random.choice(best_actions)

    def update(
        self,
        state: Tuple[Sequence[bool], int],
        action: Sequence[int],
        reward: float,
        next_state: Tuple[Sequence[bool], int],
        next_actions: Sequence[Sequence[int]],
    ) -> None:
        state_key = self.encode_state(state)
        action_key = self.encode_action(action)
        next_state_key = self.encode_state(next_state)

        self.visit_counts[(state_key, action_key)] += 1
        visit_count = self.visit_counts[(state_key, action_key)]
        alpha = self.alpha / (visit_count ** 0.5) if self.lr_decay and visit_count > 0 else self.alpha

        if next_actions:
            next_q_values = [
                self._q_value(next_state_key, self.encode_action(next_action))
                for next_action in next_actions
            ]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0.0

        current_q = self._q_value(state_key, action_key)
        target = reward + self.gamma * max_next_q
        self.q_table[(state_key, action_key)] = current_q + alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fp:
            pickle.dump(dict(self.q_table), fp)

    def load_q_table(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as fp:
            data = pickle.load(fp)
        self.q_table = defaultdict(float, data)
