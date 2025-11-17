"""Evaluate a trained Q-learning policy on held-out Connections puzzles."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.action_selection import candidate_actions
from src.agent import QLearningAgent
from src.env import ConnectionsEnv
from src.utils import load_puzzles, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-table on Connections puzzles.")
    parser.add_argument("--q-table", type=Path, required=True)
    parser.add_argument("--puzzle-path", type=Path, default=Path("data/raw/connections.json"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--mistakes-allowed", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    puzzles = load_puzzles(args.puzzle_path)
    if not puzzles:
        raise RuntimeError("No puzzles available for evaluation.")

    env = ConnectionsEnv(mistakes_allowed=args.mistakes_allowed)
    agent = QLearningAgent(epsilon=0.0)
    agent.load_q_table(args.q_table)

    success_count = 0
    total_steps = 0
    total_mistakes = 0
    per_difficulty = defaultdict(lambda: {"episodes": 0, "success": 0})

    num_episodes = min(args.episodes, len(puzzles))
    sample = random.sample(puzzles, num_episodes)

    for puzzle in sample:
        state = env.reset(puzzle)
        done = False
        steps = 0

        while not done:
            actions = candidate_actions(state[0])
            action = agent.select_action(state, actions, explore=False)
            next_state, step_summary, done, _ = env.step(action)
            state = next_state
            steps += 1

        success = bool(env.used_mask.all())
        difficulty = puzzle.get("difficulty", "unknown")
        per_difficulty[difficulty]["episodes"] += 1
        if success:
            per_difficulty[difficulty]["success"] += 1
            success_count += 1
        total_steps += steps
        total_mistakes += args.mistakes_allowed - env.mistakes_left

    success_rate = success_count / num_episodes
    avg_steps = total_steps / num_episodes
    avg_mistakes = total_mistakes / num_episodes

    print("Evaluation complete:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Avg steps: {avg_steps:.2f}")
    print(f"  Avg mistakes used: {avg_mistakes:.2f}")
    print("  Per-difficulty success rates:")
    for difficulty, stats in per_difficulty.items():
        if stats["episodes"] == 0:
            continue
        rate = stats["success"] / stats["episodes"]
        print(f"    {difficulty}: {rate:.2%} ({stats['success']}/{stats['episodes']})")


if __name__ == "__main__":
    main()
