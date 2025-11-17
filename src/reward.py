"""Reward shaping utilities for the tabular Q-learning agents."""
from __future__ import annotations

from typing import Sequence

from .embeddings import group_cohesion
from .env import StepResult

BASE_REWARDS = {
    "correct": 5.0,
    "one_away": -1.0,
    "wrong": -2.0,
}
SUCCESS_BONUS = 10.0
FAILURE_PENALTY = -10.0


def _extract_outcome(nyt_result: StepResult | dict | str) -> tuple[str, str | None]:
    if isinstance(nyt_result, StepResult):
        return nyt_result.result, nyt_result.episode_end
    if isinstance(nyt_result, dict):
        return str(nyt_result.get("result")), nyt_result.get("episode_end")
    return str(nyt_result), None


def compute_reward(
    nyt_result: StepResult | dict | str,
    group_words: Sequence[str],
    variant: str,
    lambda_embed: float,
    embed_source: str,
) -> float:
    """Combine base NYT rewards with embedding-based shaping."""

    variant = variant.lower()
    result_str, episode_end = _extract_outcome(nyt_result)
    base_reward = BASE_REWARDS.get(result_str)
    if base_reward is None:
        raise ValueError(f"Unknown NYT result: {result_str}")
    if episode_end == "success":
        base_reward += SUCCESS_BONUS
    elif episode_end == "failure":
        base_reward += FAILURE_PENALTY

    if variant == "baseline" or lambda_embed == 0:
        return base_reward

    cohesion = group_cohesion(group_words, embed_source)
    cohesion = max(-0.5, min(0.5, cohesion))
    shaped_reward = base_reward + lambda_embed * cohesion
    return shaped_reward
