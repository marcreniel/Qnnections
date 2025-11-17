"""CLI entry-point for training tabular Q-learning agents on Connections puzzles."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from src.action_selection import candidate_actions
from src.agent import QLearningAgent
from src.embeddings import load_glove_embeddings, load_gemma_embedder
from src.env import ConnectionsEnv
from src.reward import compute_reward
from src.utils import load_puzzles, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-learning agent on Connections puzzles.")
    parser.add_argument("--variant", choices=["baseline", "glove", "gemma"], default="baseline")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument(
        "--lambda-embed",
        "--lambda_embed",
        type=float,
        default=0.1,
        dest="lambda_embed",
        help="Weight applied to embedding cohesion shaping term.",
    )
    parser.add_argument("--mistakes-allowed", type=int, default=3, dest="mistakes_allowed")
    parser.add_argument("--puzzle-path", type=Path, default=Path("data/raw/connections.json"))
    parser.add_argument("--glove-path", type=Path, default=Path("data/embeddings/glove.6B.300d.txt"))
    parser.add_argument("--gemma-model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument(
        "--gemma-device",
        type=str,
        default=None,
        help="Device string understood by PyTorch (e.g., cuda, cpu).",
    )
    parser.add_argument("--q-table-out", type=Path, default=Path("reports/q_tables/latest_q_table.pkl"))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    puzzles = load_puzzles(args.puzzle_path)
    if not puzzles:
        raise RuntimeError("No puzzles loaded; check the dataset path.")

    env = ConnectionsEnv(mistakes_allowed=args.mistakes_allowed)
    agent = QLearningAgent(gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon)

    if args.variant == "glove":
        embed_source = "glove"
        load_glove_embeddings(args.glove_path)
    elif args.variant == "gemma":
        embed_source = "gemma"
        load_gemma_embedder(args.gemma_model, device=args.gemma_device)
    else:
        embed_source = "glove"

    success_count = 0
    failure_count = 0
    total_steps = 0
    total_mistakes = 0

    progress = tqdm(range(1, args.episodes + 1), desc="Training", unit="episode")
    for episode_idx in progress:
        puzzle = random.choice(puzzles)
        state = env.reset(puzzle)
        done = False
        steps = 0

        while not done:
            actions = candidate_actions(state[0])
            action = agent.select_action(state, actions)
            next_state, step_summary, done, _ = env.step(action)

            reward = compute_reward(
                nyt_result=step_summary,
                group_words=step_summary.guess_words,
                variant=args.variant,
                lambda_embed=args.lambda_embed,
                embed_source=embed_source,
            )

            next_actions = candidate_actions(next_state[0]) if not done else []
            agent.update(state, action, reward, next_state, next_actions)
            state = next_state
            steps += 1

        if env.mistakes_left == 0:
            failure_count += 1
            total_mistakes += args.mistakes_allowed
        else:
            success_count += 1
            total_mistakes += args.mistakes_allowed - env.mistakes_left
        total_steps += steps
        agent.decay_epsilon()

        if episode_idx % args.checkpoint_every == 0 or episode_idx == args.episodes:
            agent.save_q_table(args.q_table_out)

        progress.set_postfix(
            success_rate=f"{success_count / episode_idx:.2f}",
            avg_steps=f"{total_steps / episode_idx:.1f}",
        )

    print("Training complete.")
    print(f"Success rate: {success_count / args.episodes:.2%}")
    print(f"Average steps: {total_steps / args.episodes:.2f}")
    print(f"Average mistakes used: {total_mistakes / args.episodes:.2f}")


if __name__ == "__main__":
    main()
