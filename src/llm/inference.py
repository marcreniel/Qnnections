"""Inference helpers for best-of-N Connections puzzle solving."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from src.llm.data import build_prompt, get_true_groups
from src.llm.utils import RewardSettings, compute_reward, parse_solution


def solve_puzzle_best_of_n(
    model,
    tokenizer: AutoTokenizer,
    puzzle: Dict[str, Any],
    *,
    n_samples: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    device: torch.device | str | None = None,
    reward_stage: int = 3,
) -> Tuple[str, List[List[str]] | None, float]:
    """Solve a single puzzle using best-of-N sampling.

    Returns a tuple of (best_completion_text, parsed_groups, reward).
    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    true_groups = get_true_groups(puzzle)
    cfg = RewardSettings(reward_stage=reward_stage)

    best_reward = -1.0
    best_text: str | None = None
    best_groups: List[List[str]] | None = None

    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

            pred_groups = parse_solution(gen_text, puzzle.get("all_words", shuffled_words))
            reward = compute_reward(pred_groups, true_groups, cfg)

            if reward > best_reward:
                best_reward = reward
                best_text = gen_text
                best_groups = pred_groups

    model.train()
    return best_text or "", best_groups, best_reward
