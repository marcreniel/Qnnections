"""CLI entry-point for training tabular Q-learning agents on Connections puzzles."""
from __future__ import annotations

import argparse
import random
import warnings
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
    parser.add_argument(
        "--metrics-plot",
        type=Path,
        default=None,
        help="Optional path to save a matplotlib plot of training metrics.",
    )
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
    total_reward = 0.0
    success_rate_history: list[float] = []
    avg_steps_history: list[float] = []
    avg_mistakes_history: list[float] = []
    avg_reward_history: list[float] = []

    progress = tqdm(range(1, args.episodes + 1), desc="Training", unit="episode")
    for episode_idx in progress:
        puzzle = random.choice(puzzles)
        state = env.reset(puzzle)
        done = False
        steps = 0
        episode_reward = 0.0

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
            episode_reward += reward

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
        total_reward += episode_reward
        success_rate_history.append(success_count / episode_idx)
        avg_steps_history.append(total_steps / episode_idx)
        avg_mistakes_history.append(total_mistakes / episode_idx)
        avg_reward_history.append(total_reward / episode_idx)
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

    if args.metrics_plot:
        _save_metrics_plot(
            plot_path=args.metrics_plot,
            success_rate_history=success_rate_history,
            avg_steps_history=avg_steps_history,
            avg_mistakes_history=avg_mistakes_history,
            avg_reward_history=avg_reward_history,
        )


def _save_metrics_plot(
    plot_path: Path,
    success_rate_history: list[float],
    avg_steps_history: list[float],
    avg_mistakes_history: list[float],
    avg_reward_history: list[float],
) -> None:
    """Persist a simple matplotlib plot for the tracked metrics."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        warnings.warn(f"matplotlib not available, skipping metrics plot: {exc}")
        return

    episodes = range(1, len(success_rate_history) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(episodes, success_rate_history, label="Success rate", color="tab:green")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, avg_steps_history, label="Avg steps", color="tab:blue")
    axes[1].plot(episodes, avg_mistakes_history, label="Avg mistakes", color="tab:red")
    axes[1].set_ylabel("Steps / mistakes")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(episodes, avg_reward_history, label="Avg reward", color="tab:purple", alpha=0.8)
    axes[2].set_ylabel("Average reward")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle("Connections Q-Learning training metrics")
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
