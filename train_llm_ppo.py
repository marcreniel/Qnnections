"""PPO (Bandit) training loop for the Connections LLM using TRL's PPOTrainer."""

from __future__ import annotations

import os
import random
import re
import statistics
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from peft import LoraConfig

from trl import AutoModelForCausalLMWithValueHead
from trl.experimental.ppo import PPOConfig, PPOTrainer

from src.llm.data import build_prompt, get_true_groups, load_puzzles
from src.llm.utils import (
    RewardSettings,
    compute_reward,
    count_correct_words,
    count_full_groups,
    normalize_word,
    parse_group_guess,
    parse_solution,
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


@dataclass
class PuzzleMetadata:
    prompt_text: str
    true_groups: List[List[str]]
    original_words: List[str]


class ConnectionsPromptDataset(Dataset):
    """Simple dataset wrapper compatible with DataCollatorWithPadding."""

    def __init__(self, samples: List[Dict[str, List[int]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


def evaluate_bandit_agent(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    eval_puzzles: List[dict],
    num_samples: int = 50,
    device: str | torch.device = "cpu",
    generation_kwargs: Dict | None = None,
    reward_config: RewardSettings | None = None,
) -> Dict[str, float]:
    """Interactive evaluation that mirrors NYT Connections gameplay."""

    system_prompt = (
        "You are solving an NYT Connections puzzle. Submit exactly one group of four "
        "remaining words per turn, formatted as JSON: {\"guess\": {\"theme\": \"...\", "
        "\"members\": [\"WORD\", ...]}}. Only use remaining words and do not repeat solved groups."
    )
    max_mistakes = 4

    def render_words(word_order: List[str], lookup: Dict[str, str]) -> str:
        if not word_order:
            return "(none)"
        return ", ".join(lookup.get(word, word) for word in word_order)

    def render_solved(true_groups: List[List[str]], solved_flags: List[bool]) -> str:
        lines = [
            f"Group {idx + 1}: {', '.join(true_groups[idx])}"
            for idx, solved in enumerate(solved_flags)
            if solved
        ]
        return "\n".join(lines) if lines else "(none yet)"

    def build_state_prompt(
        base_prompt: str,
        remaining_order: List[str],
        lookup: Dict[str, str],
        true_groups: List[List[str]],
        solved_flags: List[bool],
        mistakes_used: int,
    ) -> str:
        parts = [
            base_prompt.strip(),
            "",
            f"Remaining words ({len(remaining_order)}): {render_words(remaining_order, lookup)}",
            f"Solved groups so far:\n{render_solved(true_groups, solved_flags)}",
            f"Mistakes used: {mistakes_used}/{max_mistakes}",
            "Submit the next group using the JSON schema described in the system prompt.",
        ]
        return "\n".join(parts).strip()

    def build_feedback_message(
        feedback_line: str,
        remaining_order: List[str],
        lookup: Dict[str, str],
        true_groups: List[List[str]],
        solved_flags: List[bool],
        mistakes_used: int,
        needs_next_guess: bool,
    ) -> str:
        parts = [feedback_line]
        parts.append(f"Mistakes used: {mistakes_used}/{max_mistakes}")
        parts.append(f"Solved groups:\n{render_solved(true_groups, solved_flags)}")
        parts.append(f"Remaining words ({len(remaining_order)}): {render_words(remaining_order, lookup)}")
        if needs_next_guess:
            parts.append("Submit another single group guess in the required JSON format.")
        return "\n".join(parts)

    model.eval()
    samples = eval_puzzles[:num_samples]
    if len(samples) < num_samples and samples:
        samples = (samples * (num_samples // len(samples) + 1))[:num_samples]

    rewards: List[float] = []
    reward_success_list: List[float] = []
    nyt_success_list: List[float] = []
    mistake_counts: List[int] = []
    correct_word_counts: List[int] = []
    full_group_counts: List[int] = []
    sample_logs: List[Dict[str, Any]] = []

    greedy_config = generation_kwargs or dict(
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    for puzzle in tqdm(samples, desc="Eval"):
        base_prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
        true_groups = get_true_groups(puzzle)
        true_sets = [set(normalize_word(word) for word in group) for group in true_groups]
        remaining_order = [normalize_word(word) for word in shuffled_words]
        lookup = {normalize_word(word): word for word in shuffled_words}
        remaining_set = set(remaining_order)
        solved_flags = [False] * len(true_groups)
        solved_count = 0
        mistakes = 0
        guess_history: List[List[str]] = []

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_state_prompt(
                    base_prompt, remaining_order, lookup, true_groups, solved_flags, mistakes
                ),
            },
        ]

        attempt = 0
        last_assistant_response = ""
        while True:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **greedy_config)

            gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            last_assistant_response = gen_text
            messages.append({"role": "assistant", "content": gen_text})

            allowed_words = [lookup[word] for word in remaining_order] if remaining_order else shuffled_words
            guess = parse_group_guess(gen_text, allowed_words)
            guess_words = guess.members if guess else []
            padded_guess = guess_words[:4]
            while len(padded_guess) < 4:
                padded_guess.append("")
            guess_history.append(padded_guess)

            if not guess:
                mistakes += 1
                feedback_line = "Could not parse a valid group; counted as a mistake."
            else:
                guess_set = set(guess_words)
                if len(guess_set) != 4:
                    mistakes += 1
                    feedback_line = "Each group must contain four unique words."
                elif not guess_set <= remaining_set:
                    mistakes += 1
                    feedback_line = "One or more words are not available in the remaining bank."
                else:
                    match_idx = -1
                    for idx, true_set in enumerate(true_sets):
                        if not solved_flags[idx] and guess_set == true_set:
                            match_idx = idx
                            break
                    if match_idx != -1:
                        solved_flags[match_idx] = True
                        solved_count += 1
                        for word in guess_set:
                            remaining_set.discard(word)
                            if word in remaining_order:
                                remaining_order.remove(word)
                        theme_note = f" Theme: {guess.theme}." if guess.theme else ""
                        display_words = ", ".join(lookup.get(word, word) for word in guess_words)
                        feedback_line = f"Correct group found: {display_words}.{theme_note}".strip()
                    else:
                        mistakes += 1
                        feedback_line = "That set of words does not match any remaining group."

            attempt += 1
            needs_next_guess = mistakes < max_mistakes and solved_count < len(true_groups)
            feedback_text = build_feedback_message(
                feedback_line,
                remaining_order,
                lookup,
                true_groups,
                solved_flags,
                mistakes,
                needs_next_guess,
            )
            messages.append({"role": "user", "content": feedback_text})

            if not needs_next_guess:
                break

        while len(guess_history) < 4:
            guess_history.append(["", "", "", ""])

        reward = compute_reward(guess_history, true_groups, reward_config)
        rewards.append(reward)
        reward_success_list.append(1.0 if reward == 1.0 else 0.0)
        correct_word_counts.append(count_correct_words(guess_history, true_groups))
        full_group_counts.append(count_full_groups(guess_history, true_groups))

        success = solved_count == len(true_groups) and mistakes < max_mistakes
        nyt_success_list.append(1.0 if success else 0.0)
        mistake_counts.append(min(mistakes, max_mistakes))

        if len(sample_logs) < 5:
            sample_logs.append(
                {
                    "puzzle_id": puzzle.get("id", "unknown"),
                    "date": puzzle.get("date"),
                    "guesses": [list(words) for words in guess_history],
                    "assistant_response": last_assistant_response,
                    "success": success,
                    "mistakes": mistakes,
                    "reward": reward,
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
            print(
                f"- Puzzle {pid} ({date}): success={log['success']} mistakes={log['mistakes']} reward={log['reward']:.3f}"
            )
            print("  Assistant response:")
            print("  " + log["assistant_response"].replace("\n", "\n  "))
            print("  Parsed guesses:")
            for guess in log["guesses"]:
                print(f"    {guess}")
            print()
    return {
        "success_reward": success_reward,
        "success_nyt": success_nyt,
        "avg_mistakes": avg_mistakes,
        "avg_reward": avg_reward,
        "avg_correct_words": avg_correct_words,
        "avg_full_groups": avg_full_groups,
    }


def build_prompt_dataset(
    puzzles: List[Dict],
    tokenizer: AutoTokenizer,
    start_index: int,
    shuffle_words: bool,
) -> Tuple[ConnectionsPromptDataset, Dict[str, PuzzleMetadata], int]:
    samples: List[Dict[str, List[int]]] = []
    metadata: Dict[str, PuzzleMetadata] = {}
    next_index = start_index

    for puzzle in puzzles:
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
            print(
                f"[Step {self.step_count}] Reward stats -> mean: {avg_reward:.3f}, std: {reward_std:.3f}, "
                f"min: {min(rewards):.3f}, max: {max(rewards):.3f}, parse_success: {parse_rate:.2%}, "
                f"avg_correct_words: {avg_correct:.2f}, avg_full_groups: {avg_full_groups:.2f}, "
                f"solved: {solve_rate:.2%}"
            )

        self.step_count += 1
        reward_tensor = torch.tensor(rewards, device=hidden_states.device, dtype=hidden_states.dtype)
        logits = reward_tensor.view(-1, 1, 1).expand(-1, hidden_states.shape[1], 1)
        return logits

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

    next_index = 0
    train_dataset, train_meta, next_index = build_prompt_dataset(train_puzzles, tokenizer, next_index, script_args.shuffle_words)
    eval_dataset = None
    eval_meta: Dict[str, PuzzleMetadata] = {}
    if eval_puzzles:
        eval_dataset, eval_meta, next_index = build_prompt_dataset(eval_puzzles, tokenizer, next_index, False)

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

    value_model = SharedValueModel(policy_model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # Reward stages let us start with "format only" shaping (stage 1) and
    # progressively add word- and group-level credit assignment.
    reward_config = RewardSettings(reward_stage=script_args.reward_stage)
    reward_model = ConnectionsRewardModel(tokenizer, reward_metadata, RESPONSE_PREFIX, pad_token_id, reward_config)

    peft_config = None
    if script_args.use_lora:
        peft_config = LoraConfig(
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    ref_model = None
    if peft_config is None:
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
        peft_config=peft_config,
    )

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    unwrapped_policy = trainer.accelerator.unwrap_model(trainer.model).policy.to(device)
    eval_kwargs = dict(
        max_new_tokens=script_args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    if eval_puzzles:
        eval_metrics = evaluate_bandit_agent(
            unwrapped_policy,
            tokenizer,
            eval_puzzles,
            num_samples=script_args.eval_samples,
            device=device,
            generation_kwargs=eval_kwargs,
            reward_config=reward_config,
        )
        print(
            "Final Eval Success (NYT rules): "
            f"{eval_metrics['success_nyt']:.2%}, Avg Reward: {eval_metrics['avg_reward']:.3f}, "
            f"Avg Mistakes: {eval_metrics['avg_mistakes']:.2f}, Reward-Exact Success: {eval_metrics['success_reward']:.2%}, "
            f"Avg Correct Words: {eval_metrics['avg_correct_words']:.2f}, Avg Full Groups: {eval_metrics['avg_full_groups']:.2f}"
        )


if __name__ == "__main__":
    # Curriculum reference:
    #  Stage 1 (structure only): python train_llm_ppo.py --reward_stage 1 --output_dir outputs/stage1
    #  Stage 2 (word coverage):  python train_llm_ppo.py --model_name_or_path outputs/stage1 --reward_stage 2 --output_dir outputs/stage2
    #  Stage 3 (full solves):    python train_llm_ppo.py --model_name_or_path outputs/stage2 --reward_stage 3 --output_dir outputs/stage3
    main()
