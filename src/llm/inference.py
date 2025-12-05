"""Inference helpers for best-of-N Connections puzzle solving."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from src.llm.data import build_prompt, get_true_groups
from src.llm.utils import (
    RewardSettings,
    compute_reward,
    count_correct_words,
    count_full_groups,
    parse_solution,
)


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


def run_rlvr_eval(
    model,
    tokenizer: AutoTokenizer,
    puzzles: List[Dict[str, Any]],
    *,
    n_samples: int = 8,
    generation_kwargs: Dict[str, Any] | None = None,
    reward_stage: int = 3,
    device: torch.device | str | None = None,
    shuffle_words: bool = True,
    log_path: str | None = None,
) -> Dict[str, float]:
    """Run a best-of-N evaluation over ``puzzles`` and summarize results."""

    if device is None:
        device = next(model.parameters()).device

    gen_kwargs = generation_kwargs or {}
    max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
    temperature = gen_kwargs.get("temperature", 0.7)
    top_p = gen_kwargs.get("top_p", 0.9)
    do_sample = gen_kwargs.get("do_sample", True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    cfg = RewardSettings(reward_stage=reward_stage)
    model.eval()

    records: List[Dict[str, Any]] = []
    total_reward = 0.0
    total_correct = 0
    total_full_groups = 0
    solve_hits = 0

    for puzzle in puzzles:
        prompt, shuffled_words = build_prompt(puzzle, shuffle_words=shuffle_words)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        true_groups = get_true_groups(puzzle)
        best_text = ""
        best_groups: List[List[str]] | None = None
        best_reward = -1.0

        with torch.no_grad():
            for _ in range(max(1, n_samples)):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                )
                gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

                pred_groups = parse_solution(gen_text, puzzle.get("all_words", shuffled_words))
                reward = compute_reward(pred_groups, true_groups, cfg)

                if reward > best_reward:
                    best_reward = reward
                    best_text = gen_text
                    best_groups = pred_groups

        correct_words = count_correct_words(best_groups, true_groups)
        full_groups = count_full_groups(best_groups, true_groups)
        total_reward += best_reward
        total_correct += correct_words
        total_full_groups += full_groups
        solve_hits += 1 if full_groups == 4 else 0

        records.append(
            {
                "puzzle_id": puzzle.get("id"),
                "date": puzzle.get("date"),
                "best_response": best_text,
                "parsed_groups": best_groups,
                "reward": best_reward,
                "full_groups": full_groups,
                "correct_words": correct_words,
            }
        )

    model.train()

    denom = len(records) or 1
    summary = {
        "full_solve_rate": solve_hits / denom,
        "avg_full_groups": total_full_groups / denom,
        "avg_correct_words": total_correct / denom,
        "avg_reward": total_reward / denom,
    }

    if log_path:
        payload = {"summary": summary, "records": records}
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return summary
