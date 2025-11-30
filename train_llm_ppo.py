"""PPO (Bandit) training loop for the Connections LLM using TRL's PPOTrainer."""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Tuple

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
from src.llm.utils import compute_reward, parse_solution


PUZZLE_TAG_PATTERN = re.compile(r"\[\[PUZZLE_ID=(.+?)\]\]")
RESPONSE_PREFIX = "\n\n### RESPONSE START\n"


@dataclass
class ConnectionsArguments:
    """Script-specific arguments that complement PPOConfig."""

    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    data_path: str = "data/raw/connections.json"
    output_dir: str = "connections-llm-ppo"
    eval_ratio: float = 0.1
    max_train_puzzles: int | None = None
    seed: int = 42
    shuffle_words: bool = True
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    eval_samples: int = 50
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50


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
) -> Dict[str, float]:
    """One-shot greedy evaluation on held-out puzzles."""

    model.eval()
    samples = eval_puzzles[:num_samples]
    if len(samples) < num_samples and samples:
        samples = (samples * (num_samples // len(samples) + 1))[:num_samples]

    rewards: List[float] = []
    success_full_list: List[float] = []

    greedy_config = generation_kwargs or dict(
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    for puzzle in tqdm(samples, desc="Eval"):
        prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **greedy_config)

        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        true_groups = get_true_groups(puzzle)
        pred_groups = parse_solution(gen_text, shuffled_words)
        reward = compute_reward(pred_groups, true_groups)

        rewards.append(reward)
        success_full_list.append(1.0 if reward == 1.0 else 0.0)

    model.train()

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success_rate = sum(success_full_list) / len(success_full_list) if success_full_list else 0.0
    return {"success_full": success_rate, "avg_reward": avg_reward}


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
    """Dummy backbone that stores the latest token ids for reward computation."""

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
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.response_prefix = response_prefix
        self.backbone = backbone
        self.pad_token_id = pad_token_id

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pragma: no cover - depends on runtime tensors
        if self.backbone.last_input_ids is None:
            raise RuntimeError("Reward backbone has no cached inputs; ensure forward was called first.")

        batch_tokens = self.backbone.last_input_ids
        rewards: List[float] = []
        for tokens in batch_tokens:
            trimmed = self._trim_padding(tokens)
            decoded = self.tokenizer.decode(trimmed, skip_special_tokens=False)
            puzzle_id = self._extract_puzzle_id(decoded)
            meta = self.metadata.get(puzzle_id)
            response_text = self._extract_response(decoded, meta.prompt_text if meta else None)
            pred_groups = parse_solution(response_text, meta.original_words if meta else [])
            reward = compute_reward(pred_groups, meta.true_groups) if meta else -1.0
            rewards.append(reward)

        reward_tensor = torch.tensor(rewards, device=hidden_states.device, dtype=hidden_states.dtype)
        logits = torch.zeros_like(hidden_states)
        logits[..., 0] = reward_tensor.unsqueeze(-1).expand_as(logits[..., 0])
        return logits

    def _trim_padding(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens != self.pad_token_id
        if not torch.any(mask):
            return tokens
        first = int(torch.nonzero(mask, as_tuple=False)[0].item())
        return tokens[first:]

    def _extract_puzzle_id(self, decoded: str) -> str:
        match = PUZZLE_TAG_PATTERN.search(decoded)
        return match.group(1) if match else "UNKNOWN"

    def _extract_response(self, decoded: str, prompt_text: str | None) -> str:
        if prompt_text and decoded.startswith(prompt_text):
            return decoded[len(prompt_text):]
        if self.response_prefix in decoded:
            return decoded.split(self.response_prefix, maxsplit=1)[-1]
        return decoded


class ConnectionsRewardModel(nn.Module):
    """Reward model wrapper compatible with PPOTrainer expectations."""

    base_model_prefix = "backbone"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        metadata: Dict[str, PuzzleMetadata],
        response_prefix: str,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.backbone = ConnectionsRewardBackbone()
        self.score = RewardScorer(tokenizer, metadata, response_prefix, self.backbone, pad_token_id)


class SharedValueModel(nn.Module):
    """Shares the policy backbone/value head with PPOTrainer's value pathway."""

    base_model_prefix = "pretrained_model"

    def __init__(self, policy_model: AutoModelForCausalLMWithValueHead) -> None:
        super().__init__()
        self.pretrained_model = policy_model.pretrained_model
        self.score = policy_model.v_head


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
    parser = HfArgumentParser((ConnectionsArguments, PPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = script_args.output_dir
    training_args.response_length = script_args.max_new_tokens
    maybe_setup_wandb(script_args, training_args)

    random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)

    puzzles = load_puzzles(script_args.data_path)
    random.shuffle(puzzles)
    split_idx = int(len(puzzles) * (1.0 - script_args.eval_ratio))
    train_puzzles = puzzles[:split_idx]
    eval_puzzles = puzzles[split_idx:] if split_idx < len(puzzles) else puzzles
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
    reward_metadata = {**train_meta, **eval_meta}

    dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=dtype,
    )
    if hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
    if hasattr(policy_model.config, "use_cache"):
        policy_model.config.use_cache = False

    value_model = SharedValueModel(policy_model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    reward_model = ConnectionsRewardModel(tokenizer, reward_metadata, RESPONSE_PREFIX, pad_token_id)

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

    trainer.train()

    os.makedirs(script_args.output_dir, exist_ok=True)
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)

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
        )
        print(f"Final Eval Success: {eval_metrics['success_full']:.2%}, Avg Reward: {eval_metrics['avg_reward']:.3f}")


if __name__ == "__main__":
    main()
