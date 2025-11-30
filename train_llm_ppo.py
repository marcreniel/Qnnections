"""PPO (Bandit) Training for Connections LLM."""
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from src.llm.data import load_puzzles, build_prompt, get_true_groups
from src.llm.utils import parse_solution, compute_reward

def evaluate_bandit_agent(model, tokenizer, eval_puzzles, num_samples=50, device="cpu"):
    """
    Evaluate model on up to num_samples eval puzzles, one-shot, greedy decode.
    """
    model.eval()
    
    samples = eval_puzzles[:num_samples]
    if len(samples) < num_samples:
        samples = (samples * (num_samples // len(samples) + 1))[:num_samples]
        
    rewards = []
    success_full_list = []
    
    print(f"Evaluating on {len(samples)} puzzles...")
    
    for puzzle in tqdm(samples, desc="Eval"):
        prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
            {"role": "user", "content": prompt},
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Greedy
                temperature=None,
                top_p=None,
            )
            
        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        true_groups = get_true_groups(puzzle)
        pred_groups = parse_solution(gen_text, shuffled_words)
        
        r = compute_reward(pred_groups, true_groups, strict=True)
        
        rewards.append(r)
        success_full_list.append(1.0 if r == 1.0 else 0.0)
        
    model.train()
    
    avg_reward = sum(rewards) / len(rewards)
    success_rate = sum(success_full_list) / len(success_full_list)
    
    return {"success_full": success_rate, "avg_reward": avg_reward}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct") # Use base model directly
    parser.add_argument("--data_path", type=str, default="data/raw/connections.json")
    parser.add_argument("--output_dir", type=str, default="llama-1b-connections-ppo")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=100)
    
    args = parser.parse_args()
    
    # Load Data
    puzzles = load_puzzles(args.data_path)
    random.shuffle(puzzles)
    split_idx = int(0.9 * len(puzzles))
    train_puzzles = puzzles[:split_idx]
    eval_puzzles = puzzles[split_idx:]
    
    # Config
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target_kl=0.1,
        kl_penalty="kl",
        gamma=1.0, # Bandit setting
        lam=0.95,
        steps=args.steps,
    )
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Required for PPO generation
    
    # Set default chat template if missing (e.g. for gpt2)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    
    # Load Model
    # Note: AutoModelForCausalLMWithValueHead adds a value head to the base model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map="auto",
    )
    
    # Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )
    
    device = ppo_trainer.accelerator.device
    print(f"Using device: {device}")
    
    # Training Loop
    print(f"Starting PPO training for {args.steps} steps...")
    
    for step in tqdm(range(args.steps)):
        # Sample batch
        batch_puzzles = random.sample(train_puzzles, args.batch_size)
        
        queries = []
        responses = []
        rewards = []
        
        # Generate and Reward
        for puzzle in batch_puzzles:
            prompt, shuffled_words = build_prompt(puzzle, shuffle_words=True)
            
            # Format prompt as chat
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves Connections puzzles."},
                {"role": "user", "content": prompt},
            ]
            query_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            queries.append(query_text)
            
            # Tokenize
            inputs = tokenizer(query_text, return_tensors="pt", padding=False).to(device)
            
            # Generate
            output = ppo_trainer.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Extract response
            gen_tokens = output[0, inputs["input_ids"].shape[1]:]
            responses.append(gen_tokens)
            
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Reward
            true_groups = get_true_groups(puzzle)
            pred_groups = parse_solution(gen_text, shuffled_words)
            r = compute_reward(pred_groups, true_groups, strict=True)
            rewards.append(torch.tensor(r, dtype=torch.float))
            
        # PPO Update
        # Convert queries to tensors
        query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"][0].to(device) for q in queries]
        
        stats = ppo_trainer.step(query_tensors, responses, rewards)
        
        # Log
        if (step + 1) % 10 == 0:
            avg_reward = torch.stack(rewards).mean().item()
            tqdm.write(f"Step {step+1} | Avg Reward: {avg_reward:.2f}")
            
        # Eval
        if (step + 1) % args.eval_freq == 0:
            metrics = evaluate_bandit_agent(ppo_trainer.model, tokenizer, eval_puzzles, device=device)
            tqdm.write(f"Eval @ {step+1}: Success={metrics['success_full']:.2%} | Avg Reward={metrics['avg_reward']:.2f}")
            
        # Checkpoint
        if (step + 1) % args.save_freq == 0:
            ckpt_dir = f"{args.output_dir}/checkpoint-{step+1}"
            tqdm.write(f"Saving checkpoint to {ckpt_dir}...")
            ppo_trainer.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            
    # Save
    print(f"Saving PPO model to {args.output_dir}...")
    ppo_trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
