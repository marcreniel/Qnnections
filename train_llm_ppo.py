"""PPO (Bandit) training loop for Connections LLM."""
from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl import AutoModelForCausalLMWithValueHead

from src.llm.data import build_prompt, get_true_groups, load_puzzles
from src.llm.utils import compute_reward, parse_solution


def compute_logprobs_and_values(
    model: AutoModelForCausalLMWithValueHead,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return summed response log-probs and value predictions at prompt end."""

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if isinstance(outputs, tuple):
        logits = outputs[0]
        values = outputs[-1].squeeze(-1)
    else:
        logits = outputs.logits
        values = outputs.value.squeeze(-1)

    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * shift_mask

    batch_size, seq_len = token_logprobs.shape
    response_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
    for i in range(batch_size):
        start = max(int(prompt_lens[i].item()) - 1, 0)
        end = min(start + int(response_lens[i].item()), seq_len)
        if end > start:
            response_mask[i, start:end] = True

    logprob_sums = (token_logprobs * response_mask.float()).sum(dim=1)

    indices = (prompt_lens - 1).clamp(min=0)
    batch_indices = torch.arange(values.size(0), device=values.device)
    value_preds = values[batch_indices, indices]
    return logprob_sums, value_preds


def evaluate_bandit_agent(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    eval_puzzles: List[dict],
    num_samples: int = 50,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """One-shot greedy evaluation on held-out puzzles."""

    model.eval()

    samples = eval_puzzles[:num_samples]
    if len(samples) < num_samples and samples:
        samples = (samples * (num_samples // len(samples) + 1))[:num_samples]

    rewards: List[float] = []
    success_full_list: List[float] = []

    greedy_config = dict(
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json")
    parser.add_argument("--output_dir", type=str, default="llama-3.2-1b-connections-ppo")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    puzzles = load_puzzles(args.data_path)
    random.shuffle(puzzles)
    split_idx = int(0.9 * len(puzzles))
    train_puzzles = puzzles[:split_idx]
    eval_puzzles = puzzles[split_idx:] if split_idx < len(puzzles) else puzzles

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
            "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ "
            "'<|im_start|>assistant\n' }}{% endif %}"
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not hasattr(model, "generation_config") or model.generation_config is None:
        base_config = getattr(model, "pretrained_model", model).config
        model.generation_config = GenerationConfig.from_model_config(base_config)

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "gradient_checkpointing_enable"):
        model.pretrained_model.gradient_checkpointing_enable()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(f"Using device: {device}")
    print(f"Starting PPO training for {args.steps} steps...")

    for step in tqdm(range(args.steps)):
        batch_puzzles = random.sample(train_puzzles, args.batch_size)

        sequences: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        response_lens: List[int] = []
        rewards: List[float] = []

        for puzzle in batch_puzzles:
            prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
                {"role": "user", "content": prompt},
            ]
            query_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(query_text, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)

            full_sequence = output[0]
            prompt_len = inputs["input_ids"].shape[1]
            response_len = max(full_sequence.shape[0] - prompt_len, 1)
            sequences.append(full_sequence.cpu())
            prompt_lens.append(prompt_len)
            response_lens.append(response_len)

            gen_tokens = full_sequence[prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            true_groups = get_true_groups(puzzle)
            pred_groups = parse_solution(gen_text, shuffled_words)
            rewards.append(compute_reward(pred_groups, true_groups))

        input_ids = pad_sequence(
            sequences, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        prompt_lens_tensor = torch.tensor(prompt_lens, device=device)
        response_lens_tensor = torch.tensor(response_lens, device=device)
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

        with torch.no_grad():
            old_logprobs, old_values = compute_logprobs_and_values(
                model, input_ids, attention_mask, prompt_lens_tensor, response_lens_tensor
            )

        advantages = rewards_tensor - old_values
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        targets = rewards_tensor

        model.train()
        new_logprobs, new_values = compute_logprobs_and_values(
            model, input_ids, attention_mask, prompt_lens_tensor, response_lens_tensor
        )
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        pg_loss = -torch.min(advantages * ratio, advantages * clipped_ratio)

        value_clipped = old_values + torch.clamp(new_values - old_values, -args.value_clip, args.value_clip)
        value_loss_unclipped = (new_values - targets) ** 2
        value_loss_clipped = (value_clipped - targets) ** 2
        vf_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)

        loss = (pg_loss + args.vf_coef * vf_loss).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if (step + 1) % 10 == 0:
            avg_reward = rewards_tensor.mean().item()
            tqdm.write(f"Step {step + 1} | Avg Reward: {avg_reward:.2f} | Loss: {loss.item():.4f}")
            torch.cuda.empty_cache()

        if (step + 1) % args.eval_freq == 0 and eval_puzzles:
            metrics = evaluate_bandit_agent(model, tokenizer, eval_puzzles, device=device)
            tqdm.write(
                f"Eval @ {step + 1}: Success={metrics['success_full']:.2%} | Avg Reward={metrics['avg_reward']:.2f}"
            )

        if (step + 1) % args.save_freq == 0:
            ckpt_dir = f"{args.output_dir}/checkpoint-{step + 1}"
            tqdm.write(f"Saving checkpoint to {ckpt_dir}...")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    print(f"Saving PPO model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
