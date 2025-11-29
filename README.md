# Connections DQN Solver

Deep Q-Network (DQN) solver for the NYT-style **Connections** puzzle (16 words split into 4 hidden categories). This implementation uses a neural network to approximate Q-values, enabling generalization across puzzles by leveraging word embeddings.

## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate connections-rl
```

> **Gemma downloads**: Running with Gemma embeddings for the first time pulls `google/embeddinggemma-300m` from Hugging Face. The loader automatically prefers CUDA, then Apple `mps`, then CPU unless you override `--gemma-device`.

## 2. Data requirements

```
data/
├── embeddings/
│   └── glove.6B.300d.txt  # local vectors (optional, if using GloVe)
└── raw/
    └── connections.json   # puzzle dataset: 16 words + 4 solution groups each
```

## 3. Training & Evaluation

The main entry point is `train_dqn.py`. It trains the DQN agent on a set of puzzles and periodically evaluates it on a held-out test set.

### Basic Usage

```bash
python train_dqn.py --episodes 1000 --eval_freq 50
```

### Reward Shaping (Ablations)

You can adjust the weights for different reward components to guide the agent:

*   `--w_correct`: Weight for finding a correct group (default: 2.0).
*   `--w_first`: Weight for first-order semantic cohesion (cosine similarity) (default: 0.1).
*   `--w_second`: Weight for second-order relational cohesion (difference vectors) (default: 0.05).

**Example: Train with high semantic guidance**
```bash
python train_dqn.py --w_first 0.95 --w_second 0.0
```

### Key Flags

*   `--episodes`: Total training episodes.
*   `--eval_freq`: Frequency of evaluation runs.
*   `--batch_size`: Batch size for experience replay (default: 64).
*   `--gamma`: Discount factor (default: 0.99).
*   `--epsilon_start`, `--epsilon_end`, `--epsilon_decay`: Exploration parameters.
*   `--hidden_dim`: Size of hidden layers in the Q-network (default: 256).

## 4. Code Structure

The codebase is organized into:

*   **`src/dqn/`**: Core DQN implementation.
    *   `agent.py`: `DQNAgent` class with PyTorch Q-Network and Replay Buffer.
    *   `env.py`: Gymnasium-compatible `ConnectionsEnv`.
    *   `reward.py`: Reward calculation logic (Correctness + Semantic Cohesion).
    *   `actions.py`: Action space enumeration and masking utilities.
    *   `baseline.py`: Heuristic baseline agent.
*   **`src/common/`**: Shared utilities.
    *   `embeddings.py`: Word embedding generation (Gemma/GloVe).
    *   `data_loader.py`: Puzzle loading and normalization.
    *   `utils.py`: General helpers.
*   **`train_dqn.py`**: Main training and evaluation script.

## 5. Reward Structure

The reward function is designed to incentivize solving the puzzle while optionally using semantic clues as "training wheels":

*   **Correct Group**: `+2.0` (+ semantic bonuses)
*   **One Away**: `0.0`
*   **Wrong Group**: `-1.0` (Strict penalty, ignores semantic bonuses to prevent reward hacking)
*   **Puzzle Solved (Bonus)**: `+2.0`