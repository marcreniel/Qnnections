"""PPO (Bandit) training loop for the Connections LLM using TRL's PPOTrainer."""

from __future__ import annotations

import os
import random
import re
import statistics
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Set, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from trl import AutoModelForCausalLMWithValueHead
from trl.experimental.ppo import PPOConfig, PPOTrainer

from src.llm.data import build_prompt, get_true_groups, load_puzzles
from src.llm.inference import run_rlvr_eval
from src.llm.utils import (
    RewardSettings,
    compute_reward,
    count_correct_words,
    count_full_groups,
    normalize_word,
    parse_solution,
    simulate_nyt_game,
)


_STATE_DICT_PATCHED = False


def _patch_value_head_state_dict() -> None:
    global _STATE_DICT_PATCHED
    if _STATE_DICT_PATCHED:
        return

    original_state_dict = AutoModelForCausalLMWithValueHead.state_dict

    def safe_state_dict(self, *args, **kwargs):  # type: ignore[override]
        if not self.is_peft_model:
            pretrained_model_state = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state = {}

        v_head_state = self.v_head.state_dict(*args, **kwargs)
        for key, value in list(v_head_state.items()):
            pretrained_model_state[f"v_head.{key}"] = value
        return pretrained_model_state

    AutoModelForCausalLMWithValueHead._qn_orig_state_dict = original_state_dict  # type: ignore[attr-defined]
    AutoModelForCausalLMWithValueHead.state_dict = safe_state_dict  # type: ignore[assignment]
    _STATE_DICT_PATCHED = True


PUZZLE_TAG_PATTERN = re.compile(r"\[\[PUZZLE_ID=(.+?)\]\]")
RESPONSE_PREFIX = "\n\n### RESPONSE START\n"


@dataclass
class ConnectionsArguments:
    """Script-specific arguments that complement PPOConfig."""

    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    data_path: str = "data/raw/connections.json"
    eval_path: str | None = "data/raw/connections_test.json"
    eval_ratio: float = 0.1
    max_train_puzzles: int | None = None
    shuffle_words: bool = True
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    wandb_project: str | None = "cs238_llmppo"
    wandb_entity: str | None = "qn_cs238"
    wandb_name: str | None = None
    eval_samples: int = 50
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    reward_stage: int = 3  # 1=structure, 2=words, 3=groups+win bonus
    stage2_replay_path: str | None = None
    stage2_replay_fraction: float = 0.0
    resume_adapter_path: str | None = None
    stage3_exact_only: bool = False
    stage3_group_bonus: float = 0.4
    stage3_win_bonus: float = 0.15
    stage3_exact_reward: float = 1.0
    do_eval_only: bool = False
    rlvr_samples: int = 8
    rlvr_temperature: float = 0.7
    rlvr_top_p: float = 0.9
    rlvr_shuffle_words: bool = True
    rlvr_log_path: str | None = None
    rlvr_max_new_tokens: int | None = None


def _build_locked_group_prompt(
    base_prompt: str,
    puzzle: Dict[str, Any],
    locked_indices: List[int],
    locked_words: Set[str],
) -> str:
    if not locked_indices:
        return base_prompt

    answers = puzzle.get("answers") or []
    lines = [
        "You got these groups right. Keep them exactly as written and do NOT reuse their words in other groups:",
    ]

    for idx in locked_indices:
        if idx < 0 or idx >= len(answers):
            continue
        group_data = answers[idx]
        group = group_data if isinstance(group_data, dict) else {}
        category = group.get("category", f"Group {idx + 1}")
        members = [str(word) for word in group.get("members", [])]
        lines.append("{")
        lines.append(f'  "category": "{category}",')
        if members:
            members_fmt = '", "'.join(members)
            lines.append(f'  "members": ["{members_fmt}"]')
        else:
            lines.append('  "members": []')
        lines.append("}")

    remaining_words = [
        word
        for word in puzzle.get("all_words", [])
        if normalize_word(word) not in locked_words
    ]
    if remaining_words:
        lines.append("")
        lines.append("Remaining words you still need to place (do not reuse locked words):")
        lines.append(", ".join(remaining_words))

    lines.append(
        "Re-output all four groups in JSON, keeping the confirmed groups untouched and only moving the remaining words."
    )
    return base_prompt + "\n\n" + "\n".join(lines)


def _record_locked_groups(
    pred_groups: List[List[str]] | None,
    true_group_sets: List[Set[str]],
    locked_group_set: Set[int],
    locked_group_order: List[int],
    locked_words: Set[str],
) -> None:
    if not pred_groups:
        return

    for group in pred_groups:
        pred_set = {normalize_word(word) for word in group}
        if len(pred_set) != 4:
            continue
        for idx, true_set in enumerate(true_group_sets):
            if idx in locked_group_set:
                continue
            if pred_set == true_set:
                locked_group_set.add(idx)
                locked_group_order.append(idx)
                locked_words.update(true_set)
                break


def evaluate_bandit_agent(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    eval_puzzles: List[dict],
    num_samples: int = 50,
    device: str | torch.device = "cpu",
    generation_kwargs: Dict | None = None,
    reward_config: RewardSettings | None = None,
    attempts_per_puzzle: int = 4,
) -> Dict[str, float]:
    """Evaluate the policy with up to ``attempts_per_puzzle`` one-shot completions."""

    model.eval()
    if not eval_puzzles:
        return {
            "success_reward": 0.0,
            "success_nyt": 0.0,
            "avg_mistakes": 0.0,
            "avg_reward": 0.0,
            "avg_correct_words": 0.0,
            "avg_full_groups": 0.0,
        }

    samples = eval_puzzles[:num_samples]
    if len(samples) < num_samples and samples:
        samples = (samples * (num_samples // len(samples) + 1))[:num_samples]

    greedy_config = generation_kwargs or dict(
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    max_attempts = max(1, attempts_per_puzzle)
    system_prompt = "You are a helpful assistant that solves NYT Connections puzzles."

    rewards: List[float] = []
    reward_success_list: List[float] = []
    nyt_success_list: List[float] = []
    mistake_counts: List[int] = []
    correct_word_counts: List[int] = []
    full_group_counts: List[int] = []
    sample_logs: List[Dict[str, Any]] = []

    for puzzle in tqdm(samples, desc="Eval"):
        base_prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
        true_groups = get_true_groups(puzzle)
        all_words = puzzle.get("all_words", shuffled_words)
        true_group_sets = [
            {normalize_word(word) for word in group}
            for group in true_groups
        ]
        locked_group_indices: List[int] = []
        locked_group_set: Set[int] = set()
        locked_words: Set[str] = set()

        best_attempt: Dict[str, Any] | None = None
        attempt_summaries: List[Dict[str, Any]] = []

        for attempt_idx in range(max_attempts):
            prompt_text = _build_locked_group_prompt(base_prompt, puzzle, locked_group_indices, locked_words)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **greedy_config)

            gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

            pred_groups = parse_solution(gen_text, all_words)
            reward = compute_reward(pred_groups, true_groups, reward_config)
            correct_words = count_correct_words(pred_groups, true_groups)
            full_groups = count_full_groups(pred_groups, true_groups)
            game = simulate_nyt_game(pred_groups, true_groups)

            _record_locked_groups(
                pred_groups,
                true_group_sets,
                locked_group_set,
                locked_group_indices,
                locked_words,
            )

            attempt_summary = {
                "attempt_index": attempt_idx,
                "assistant_response": gen_text,
                "parsed_groups": pred_groups,
                "reward": reward,
                "nyt_success": game.success,
                "mistakes": game.mistakes,
            }
            attempt_summaries.append(attempt_summary)

            is_better = False
            if best_attempt is None:
                is_better = True
            elif game.success and not best_attempt["nyt_success"]:
                is_better = True
            elif game.success == best_attempt["nyt_success"] and reward > best_attempt["reward"]:
                is_better = True

            if is_better:
                best_attempt = {
                    "reward": reward,
                    "reward_success": 1.0 if reward >= 0.99 else 0.0,
                    "nyt_success": game.success,
                    "mistakes": game.mistakes,
                    "correct_words": correct_words,
                    "full_groups": full_groups,
                    "parsed_groups": pred_groups,
                    "assistant_response": gen_text,
                    "attempt_index": attempt_idx,
                }

            if game.success:
                break

        if best_attempt is None:
            continue

        rewards.append(best_attempt["reward"])
        reward_success_list.append(best_attempt["reward_success"])
        nyt_success_list.append(1.0 if best_attempt["nyt_success"] else 0.0)
        mistake_counts.append(best_attempt["mistakes"])
        correct_word_counts.append(best_attempt["correct_words"])
        full_group_counts.append(best_attempt["full_groups"])

        if len(sample_logs) < 5:
            sample_logs.append(
                {
                    "puzzle_id": puzzle.get("id", "unknown"),
                    "date": puzzle.get("date"),
                    "best_attempt": best_attempt,
                    "attempts": attempt_summaries,
                }
            )

    model.train()

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success_reward = sum(reward_success_list) / len(reward_success_list) if reward_success_list else 0.0
    success_nyt = sum(nyt_success_list) / len(nyt_success_list) if nyt_success_list else 0.0
    avg_mistakes = sum(mistake_counts) / len(mistake_counts) if mistake_counts else 0.0
    avg_correct_words = sum(correct_word_counts) / len(correct_word_counts) if correct_word_counts else 0.0
    avg_full_groups = sum(full_group_counts) / len(full_group_counts) if full_group_counts else 0.0

    if sample_logs:
        print("\nSample eval responses:")
        for log in sample_logs:
            pid = log.get("puzzle_id")
            date = log.get("date")
            best = log.get("best_attempt", {})
            print(
                f"- Puzzle {pid} ({date}): best_nyt_success={best.get('nyt_success')} "
                f"mistakes={best.get('mistakes')} reward={best.get('reward', 0.0):.3f}"
            )
            print("  Best assistant response:")
            response = best.get("assistant_response", "")
            print("  " + str(response).replace("\n", "\n  "))
            print("  Parsed groups:")
            print(f"    {best.get('parsed_groups')}")
            print("  Attempt summaries:")
            for attempt in log.get("attempts", []):
                print(
                    f"    Attempt {attempt['attempt_index'] + 1}: nyt_success={attempt['nyt_success']} "
                    f"mistakes={attempt['mistakes']} reward={attempt['reward']:.3f}"
                )
            print()

    return {
        "success_reward": success_reward,
        "success_nyt": success_nyt,
        "avg_mistakes": avg_mistakes,
        "avg_reward": avg_reward,
        "avg_correct_words": avg_correct_words,
        "avg_full_groups": avg_full_groups,
    }


def _select_eval_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_run_rlvr_eval(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    eval_puzzles: List[dict],
    script_args: ConnectionsArguments,
    device: torch.device,
    label: str,
) -> None:
    if not eval_puzzles or script_args.rlvr_samples <= 0:
        return

    rlvr_tokens = script_args.rlvr_max_new_tokens or script_args.max_new_tokens
    generation_kwargs = dict(
        max_new_tokens=rlvr_tokens,
        temperature=script_args.rlvr_temperature,
        top_p=script_args.rlvr_top_p,
        do_sample=True,
    )

    log_path = script_args.rlvr_log_path
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    rlvr_puzzles = eval_puzzles
    if script_args.eval_samples and len(eval_puzzles) > script_args.eval_samples:
        rlvr_puzzles = eval_puzzles[: script_args.eval_samples]

    summary = run_rlvr_eval(
        model,
        tokenizer,
        rlvr_puzzles,
        n_samples=script_args.rlvr_samples,
        generation_kwargs=generation_kwargs,
        reward_stage=script_args.reward_stage,
        device=device,
        shuffle_words=script_args.rlvr_shuffle_words,
        log_path=log_path,
    )

    print(
        f"{label} RLVR best-of-{script_args.rlvr_samples}: "
        f"full_solve_rate={summary['full_solve_rate']:.2%}, "
        f"avg_full_groups={summary['avg_full_groups']:.2f}, "
        f"avg_correct_words={summary['avg_correct_words']:.2f}, "
        f"avg_reward={summary['avg_reward']:.3f}"
    )


def _run_eval_suite(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    eval_puzzles: List[dict],
    script_args: ConnectionsArguments,
    reward_config: RewardSettings,
    device: torch.device,
    label: str,
) -> Dict[str, float] | None:
    if not eval_puzzles:
        print("No evaluation puzzles available; skipping eval suite.")
        return None

    eval_kwargs = dict(
        max_new_tokens=script_args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    metrics = evaluate_bandit_agent(
        model,
        tokenizer,
        eval_puzzles,
        num_samples=script_args.eval_samples,
        device=device,
        generation_kwargs=eval_kwargs,
        reward_config=reward_config,
    )

    print(
        f"{label} Eval Success (NYT rules): "
        f"{metrics['success_nyt']:.2%}, Avg Reward: {metrics['avg_reward']:.3f}, "
        f"Avg Mistakes: {metrics['avg_mistakes']:.2f}, Reward-Exact Success: {metrics['success_reward']:.2%}, "
        f"Avg Correct Words: {metrics['avg_correct_words']:.2f}, Avg Full Groups: {metrics['avg_full_groups']:.2f}"
    )

    _maybe_run_rlvr_eval(model, tokenizer, eval_puzzles, script_args, device, label)
    return metrics


class ConnectionsPromptDataset(Dataset):
    def __init__(self, samples: List[Dict[str, List[int]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
        }


@dataclass
class PuzzleMetadata:
    prompt_text: str
    true_groups: List[List[str]]
    original_words: List[str]


def build_prompt_dataset(
    puzzles: List[Dict],
    tokenizer: AutoTokenizer,
    start_index: int,
    shuffle_words: bool,
) -> Tuple[ConnectionsPromptDataset, Dict[str, PuzzleMetadata], int]:
    samples: List[Dict[str, List[int]]] = []
    metadata: Dict[str, PuzzleMetadata] = {}
    next_index = start_index

    iterator = tqdm(puzzles, desc="Tokenizing prompts", leave=False)
    for puzzle in iterator:
        puzzle_id = f"PZ_{next_index:06d}"
        prompt, _ = build_prompt(puzzle, shuffle_words=shuffle_words)
        tagged_prompt = f"[[PUZZLE_ID={puzzle_id}]]\n{prompt}{RESPONSE_PREFIX}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
            {"role": "user", "content": tagged_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        samples.append(
            {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist(),
            }
        )
        metadata[puzzle_id] = PuzzleMetadata(
            prompt_text=prompt_text,
            true_groups=get_true_groups(puzzle),
            original_words=list(puzzle.get("all_words", [])),
        )
        next_index += 1
    return ConnectionsPromptDataset(samples), metadata, next_index


class ConnectionsRewardBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_ids: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:  # pragma: no cover - passthrough
        self.last_input_ids = input_ids.detach().clone()
        hidden = torch.zeros(*input_ids.shape, 1, device=input_ids.device, dtype=torch.float32)
        return SimpleNamespace(hidden_states=(hidden,))


class RewardScorer(nn.Module):
    """Module that converts decoded completions into scalar rewards per token."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        metadata: Dict[str, PuzzleMetadata],
        response_prefix: str,
        backbone: ConnectionsRewardBackbone,
        pad_token_id: int,
        reward_config: RewardSettings,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.response_prefix = response_prefix
        self.backbone = backbone
        self.pad_token_id = pad_token_id
        self.step_count = 0
        self.reward_config = reward_config
        self._metric_callback: Callable[[Dict[str, float]], None] | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pragma: no cover - depends on runtime tensors
        if self.backbone.last_input_ids is None:
            raise RuntimeError("Reward backbone has no cached inputs; ensure forward was called first.")

        batch_tokens = self.backbone.last_input_ids
        rewards: List[float] = []
        correct_counts: List[int] = []
        parse_success: List[int] = []
        full_group_counts: List[int] = []
        solve_hits: List[int] = []

        # Log detailed samples on a cadence to avoid flooding stdout
        should_log = self.step_count % 10 == 0

        for i, tokens in enumerate(batch_tokens):
            trimmed = self._trim_padding(tokens)
            decoded = self.tokenizer.decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            puzzle_id = self._extract_puzzle_id(decoded)
            meta = self.metadata.get(puzzle_id)
            response_text = self._extract_response(decoded, meta.prompt_text if meta else None)
            if meta is None:
                pred_groups = None
                reward = self.reward_config.invalid_penalty
                correct_words = 0
                full_groups = 0
            else:
                pred_groups = parse_solution(response_text, meta.original_words)
                correct_words = count_correct_words(pred_groups, meta.true_groups)
                full_groups = count_full_groups(pred_groups, meta.true_groups)
                reward = compute_reward(pred_groups, meta.true_groups, self.reward_config)

            rewards.append(reward)
            correct_counts.append(correct_words)
            parse_success.append(1 if pred_groups is not None else 0)
            full_group_counts.append(full_groups)
            solve_hits.append(1 if full_groups == 4 else 0)

            if should_log and i == 0:
                print(f"\n[Step {self.step_count}] Sample Generation:\n{response_text}\n")
                print(f"[Step {self.step_count}] Parsed Groups: {pred_groups}")
                print(
                    f"[Step {self.step_count}] Reward: {reward:.3f} "
                    f"(correct_words={correct_words}, full_groups={full_groups})"
                )

        if should_log and rewards:
            avg_reward = statistics.fmean(rewards)
            reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            parse_rate = sum(parse_success) / len(parse_success)
            avg_correct = statistics.fmean(correct_counts) if correct_counts else 0.0
            avg_full_groups = statistics.fmean(full_group_counts) if full_group_counts else 0.0
            solve_rate = sum(solve_hits) / len(solve_hits) if solve_hits else 0.0
            metrics = {
                "mean": avg_reward,
                "std": reward_std,
                "min": min(rewards),
                "max": max(rewards),
                "parse_success": parse_rate,
                "avg_correct_words": avg_correct,
                "avg_full_groups": avg_full_groups,
                "solve_rate": solve_rate,
            }
            print(
                f"[Step {self.step_count}] Reward stats -> mean: {avg_reward:.3f}, std: {reward_std:.3f}, "
                f"min: {metrics['min']:.3f}, max: {metrics['max']:.3f}, parse_success: {parse_rate:.2%}, "
                f"avg_correct_words: {avg_correct:.2f}, avg_full_groups: {avg_full_groups:.2f}, "
                f"solved: {solve_rate:.2%}"
            )
            self._emit_metrics(metrics)

        self.step_count += 1
        reward_tensor = torch.tensor(rewards, device=hidden_states.device, dtype=hidden_states.dtype)
        logits = reward_tensor.view(-1, 1, 1).expand(-1, hidden_states.shape[1], 1)
        return logits

    def set_metric_callback(self, callback: Callable[[Dict[str, float]], None]) -> None:
        self._metric_callback = callback

    def _emit_metrics(self, metrics: Dict[str, float]) -> None:
        if not self._metric_callback:
            return
        try:
            self._metric_callback(metrics)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[RewardScorer] Metric callback failed: {exc}")

    def _trim_padding(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens != self.pad_token_id
        if not torch.any(mask):
            return tokens
        indices = torch.nonzero(mask, as_tuple=False)
        start = int(indices[0].item())
        end = int(indices[-1].item()) + 1
        return tokens[start:end]

    def _extract_puzzle_id(self, decoded: str) -> str:
        match = PUZZLE_TAG_PATTERN.search(decoded)
        return match.group(1) if match else "UNKNOWN"

    def _extract_response(self, decoded: str, prompt_text: str | None) -> str:
        if prompt_text and decoded.startswith(prompt_text):
            return decoded[len(prompt_text):].strip()
        if self.response_prefix in decoded:
            return decoded.split(self.response_prefix, maxsplit=1)[-1].strip()
        return decoded.strip()


class ConnectionsRewardModel(nn.Module):
    """Reward model wrapper compatible with PPOTrainer expectations."""

    base_model_prefix = "backbone"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        metadata: Dict[str, PuzzleMetadata],
        response_prefix: str,
        pad_token_id: int,
        reward_config: RewardSettings,
    ) -> None:
        super().__init__()
        self.backbone = ConnectionsRewardBackbone()
        self.score = RewardScorer(tokenizer, metadata, response_prefix, self.backbone, pad_token_id, reward_config)


class SharedValueModel(nn.Module):
    """Shares the policy backbone/value head with PPOTrainer's value pathway."""

    base_model_prefix = "pretrained_model"

    def __init__(self, policy_model: AutoModelForCausalLMWithValueHead) -> None:
        super().__init__()
        self.pretrained_model = policy_model.pretrained_model
        self.score = policy_model.v_head


def _ensure_dict_forward(model: nn.Module) -> None:
    """Wrap model.forward so PPOTrainer always receives dict-style outputs."""

    if model is None or getattr(model, "_qn_returns_dict", False):
        return

    original_forward = model.forward

    def forward(*args, **kwargs):
        kwargs.setdefault("return_dict", True)
        output = original_forward(*args, **kwargs)
        if isinstance(output, tuple):
            if len(output) >= 2:
                return SimpleNamespace(logits=output[0], value=output[1])
            return SimpleNamespace(logits=output[0])
        return output

    model.forward = forward  # type: ignore[assignment]
    model._qn_returns_dict = True


_patch_value_head_state_dict()


def _load_adapter_state(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors not installed; cannot load .safetensors adapter") from exc
        return load_file(safetensors_path)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(
        f"Could not find adapter weights in '{adapter_dir}'. Expected adapter_model.safetensors or adapter_model.bin."
    )


def _apply_adapter_state(model: nn.Module, adapter_dir: str) -> None:
    state_dict = _load_adapter_state(adapter_dir)
    set_peft_model_state_dict(model, state_dict, adapter_name="default")
    print(f"[LoRA] Loaded adapter weights from {adapter_dir}.")


def maybe_setup_wandb(args: ConnectionsArguments, training_args: PPOConfig) -> None:
    if not args.wandb_project:
        return
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_name)
    if not training_args.report_to:
        training_args.report_to = ["wandb"]


def main() -> None:
    training_parser = HfArgumentParser(PPOConfig)
    *training_args_list, remaining = training_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args = training_args_list[0]

    script_parser = HfArgumentParser(ConnectionsArguments)
    if remaining is None:
        remaining = []
    (script_args,) = script_parser.parse_args_into_dataclasses(args=remaining)
    if not training_args.output_dir:
        training_args.output_dir = "connections-llm-ppo"
    if training_args.seed is None:
        training_args.seed = 42
    if script_args.resume_adapter_path and not script_args.use_lora:
        raise ValueError("--resume_adapter_path requires --use_lora so LoRA layers are initialized.")
    
    training_args.response_length = script_args.max_new_tokens
    if hasattr(training_args, "target_kl") and training_args.target_kl is None:
        training_args.target_kl = 0.05
    if hasattr(training_args, "adap_kl_ctrl"):
        training_args.adap_kl_ctrl = True
    if hasattr(training_args, "kl_penalty") and getattr(training_args, "kl_penalty") is None:
        training_args.kl_penalty = "kl"
    default_lr = 3e-6 if script_args.reward_stage == 3 else 1e-6
    if getattr(training_args, "learning_rate", None) in (None, 0.0):
        training_args.learning_rate = default_lr
    if (
        script_args.reward_stage == 3
        and hasattr(training_args, "entropy_coef")
        and getattr(training_args, "entropy_coef", None) is None
    ):
        training_args.entropy_coef = 0.01
    maybe_setup_wandb(script_args, training_args)

    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    puzzles = load_puzzles(script_args.data_path)
    random.shuffle(puzzles)

    eval_puzzles: List[Dict] = []
    if script_args.eval_path and os.path.exists(script_args.eval_path):
        eval_puzzles = load_puzzles(script_args.eval_path)
    else:
        split_idx = int(len(puzzles) * (1.0 - script_args.eval_ratio))
        eval_puzzles = puzzles[split_idx:] if split_idx < len(puzzles) else puzzles
        puzzles = puzzles[:split_idx]

    train_puzzles = puzzles
    if script_args.max_train_puzzles:
        train_puzzles = train_puzzles[: script_args.max_train_puzzles]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

    dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=dtype,
    )
    if not getattr(policy_model, "generation_config", None):
        policy_model.generation_config = GenerationConfig.from_model_config(policy_model.config)
    gc_enabled = False
    if hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
        gc_enabled = True
    if not hasattr(policy_model, "is_gradient_checkpointing"):
        policy_model.is_gradient_checkpointing = gc_enabled
    else:
        policy_model.is_gradient_checkpointing = gc_enabled or bool(policy_model.is_gradient_checkpointing)
    if hasattr(policy_model, "config"):
        policy_model.config.return_dict = True
    if hasattr(policy_model, "pretrained_model") and hasattr(policy_model.pretrained_model, "config"):
        policy_model.pretrained_model.config.return_dict = True
    if hasattr(policy_model.config, "use_cache"):
        policy_model.config.use_cache = False
    _ensure_dict_forward(policy_model)

    if script_args.use_lora:
        lora_config = LoraConfig(
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        policy_model.pretrained_model = get_peft_model(policy_model.pretrained_model, lora_config)
        if script_args.resume_adapter_path:
            _apply_adapter_state(policy_model.pretrained_model, script_args.resume_adapter_path)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    reward_config = RewardSettings(
        reward_stage=script_args.reward_stage,
        stage3_group_bonus=script_args.stage3_group_bonus,
        stage3_win_bonus=script_args.stage3_win_bonus,
        stage3_exact_reward=script_args.stage3_exact_reward,
        stage3_exact_only=script_args.stage3_exact_only,
    )

    if script_args.do_eval_only:
        device = _select_eval_device()
        policy_model.to(device)
        _run_eval_suite(policy_model, tokenizer, eval_puzzles, script_args, reward_config, device, "Eval-only")
        return

    next_index = 0
    train_dataset, train_meta, next_index = build_prompt_dataset(
        train_puzzles,
        tokenizer,
        next_index,
        script_args.shuffle_words,
    )
    eval_dataset = None
    eval_meta: Dict[str, PuzzleMetadata] = {}
    if eval_puzzles:
        eval_dataset, eval_meta, next_index = build_prompt_dataset(
            eval_puzzles,
            tokenizer,
            next_index,
            False,
        )

    replay_fraction = max(0.0, min(1.0, script_args.stage2_replay_fraction))
    if (
        script_args.reward_stage == 3
        and replay_fraction > 0.0
        and script_args.stage2_replay_path
        and os.path.exists(script_args.stage2_replay_path)
    ):
        stage2_replay_puzzles = load_puzzles(script_args.stage2_replay_path)
        replay_dataset, replay_meta, next_index = build_prompt_dataset(
            stage2_replay_puzzles,
            tokenizer,
            next_index,
            script_args.shuffle_words,
        )
        replay_target = max(1, int(len(train_dataset) * replay_fraction))
        if len(replay_dataset) > 0:
            replay_indices = list(range(len(replay_dataset)))
            random.shuffle(replay_indices)
            selected = replay_indices[: min(replay_target, len(replay_dataset))]
            for idx in selected:
                train_dataset.samples.append(replay_dataset.samples[idx])
            train_meta.update(replay_meta)
            print(
                f"[Curriculum] Added {len(selected)} Stage 2 replay samples (fraction={replay_fraction:.2f})."
            )

    reward_metadata = {**train_meta, **eval_meta}

    value_model = SharedValueModel(policy_model)
    reward_model = ConnectionsRewardModel(tokenizer, reward_metadata, RESPONSE_PREFIX, pad_token_id, reward_config)

    ref_model = None
    if not script_args.use_lora:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=dtype,
        )
        if not getattr(ref_model, "generation_config", None):
            ref_model.generation_config = GenerationConfig.from_model_config(ref_model.config)
        if not hasattr(ref_model, "is_gradient_checkpointing"):
            ref_model.is_gradient_checkpointing = gc_enabled
        if hasattr(ref_model, "config"):
            ref_model.config.return_dict = True
        if hasattr(ref_model, "pretrained_model") and hasattr(ref_model.pretrained_model, "config"):
            ref_model.pretrained_model.config.return_dict = True
        _ensure_dict_forward(ref_model)

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if hasattr(reward_model, "score") and hasattr(reward_model.score, "set_metric_callback"):
        def _log_reward_metrics(metrics: Dict[str, float]) -> None:
            accelerator = getattr(trainer, "accelerator", None)
            if not accelerator:
                return
            prefixed = {f"reward/{key}": value for key, value in metrics.items()}
            accelerator.log(prefixed)

        reward_model.score.set_metric_callback(_log_reward_metrics)

    trainer.generation_kwargs = dict(
        max_new_tokens=script_args.max_new_tokens,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    wrapper = trainer.model
    if not hasattr(wrapper, "gradient_checkpointing_enable"):
        def _wrapper_gc_enable() -> None:
            if hasattr(wrapper.policy, "gradient_checkpointing_enable"):
                wrapper.policy.gradient_checkpointing_enable()
        wrapper.gradient_checkpointing_enable = _wrapper_gc_enable  # type: ignore[attr-defined]
    if not hasattr(wrapper, "gradient_checkpointing_disable"):
        def _wrapper_gc_disable() -> None:
            if hasattr(wrapper.policy, "gradient_checkpointing_disable"):
                wrapper.policy.gradient_checkpointing_disable()
        wrapper.gradient_checkpointing_disable = _wrapper_gc_disable  # type: ignore[attr-defined]

    trainer.train()

    os.makedirs(training_args.output_dir, exist_ok=True)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    device = _select_eval_device()
    unwrapped_policy = trainer.accelerator.unwrap_model(trainer.model).policy.to(device)
    eval_metrics = _run_eval_suite(
        unwrapped_policy,
        tokenizer,
        eval_puzzles,
        script_args,
        reward_config,
        device,
        "Final",
    )
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator and eval_metrics:
        accelerator.log({f"eval/{key}": value for key, value in eval_metrics.items()})


if __name__ == "__main__":
    # Curriculum reference:
    #  Stage 1 (structure only): python train_llm_ppo.py --reward_stage 1 --output_dir outputs/stage1
    #  Stage 2 (word coverage):  python train_llm_ppo.py --model_name_or_path outputs/stage1 --reward_stage 2 --output_dir outputs/stage2
    #  Stage 3 (full solves):    python train_llm_ppo.py --model_name_or_path outputs/stage2 --reward_stage 3 --output_dir outputs/stage3
    main()
