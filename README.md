# Connections Q-Learning

Tabular Q-learning playground for the NYT-style **Connections** puzzle (16 words split into 4 hidden categories). Three reward variants are supported:

- `baseline`: vanilla NYT rewards
- `glove`: NYT rewards + shaping term computed from local GloVe vectors
- `gemma`: NYT rewards + shaping term computed from google/embeddinggemma-300m vectors (stubbed in `src/embeddings.py` and ready for a real model)

## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate connections-rl
```

The environment includes PyTorch and `transformers` so you can swap in a true Gemma embedding model when available.

> **Gemma downloads**: Running the Gemma variant for the first time pulls `google/embeddinggemma-300m` from Hugging Face. Make sure you have internet access (and `huggingface-cli login` if the repo is gated) before launching training. The loader automatically prefers CUDA, then Apple `mps`, then CPU unless you override `--gemma-device`.

## 2. Data requirements

```
data/
├── embeddings/
│   └── glove.6B.300d.txt  # local vectors loaded by --glove-path
└── raw/
    └── connections.json   # puzzle dataset: 16 words + 4 solution groups each
```

`connections.json` may be a list of puzzles or an object with a `"puzzles"` list. Each puzzle should contain:

```json
{
  "id": "nyt-2023-08-26",
  "words": ["... 16 entries ..."],
  "groups": [
    {"name": "Category", "members": ["word", "word", "word", "word"]},
    {...}
  ],
  "difficulty": "(optional)"
}
```

## 3. Training

`train.py` wires together the environment, agent, reward shaping, and action selection. Common invocations:

```bash
python train.py --variant baseline --episodes 2000
python train.py --variant glove --lambda_embed 0.2
python train.py --variant gemma --lambda_embed 0.2 --episodes 5000 \
  --gemma-model google/embeddinggemma-300m \
  --q-table-out reports/q_tables/gemma.pkl
```

Key flags:

- `--puzzle-path`: dataset location (default `data/raw/connections.json`)
- `--glove-path`: local GloVe file (used automatically for `--variant glove`)
- `--lambda-embed` / `--lambda_embed`: weight for embedding cohesion shaping
- `--gemma-model`, `--gemma-device`: Hugging Face model id + optional PyTorch device for Gemma embeddings
- `--checkpoint-every`: save interval for the Q-table

Training metrics stream through `tqdm`. Q-tables are persisted via `QLearningAgent.save_q_table`.

## 4. Evaluation

```bash
python evaluate.py --q-table reports/q_tables/latest_q_table.pkl --episodes 200
```

Evaluation runs a greedy policy (ε = 0) and prints success rate, step counts, mistakes, and optional per-difficulty summaries.

## 5. Code map

- `src/env.py`: Connections environment + transition logic
- `src/agent.py`: tabular Q-learning agent with ε-greedy policy and optional LR decay
- `src/reward.py`: NYT base rewards + embedding-based shaping
- `src/embeddings.py`: loaders for GloVe and Gemma (via Hugging Face `transformers`), plus cohesion helpers
- `src/action_selection.py`: enumerates every valid 4-word guess from unused indices
- `src/utils.py`: puzzle loading, seeding, and mask helpers
- `train.py` / `evaluate.py`: CLI entry-points for learning and benchmarking

## 6. Plugging in real Gemma embeddings

`src/embeddings.py:get_gemma_embedding` uses Hugging Face `transformers` to load `google/embeddinggemma-300m` (or any compatible embedding ID). The first run will download weights to your HF cache; pass `--gemma-model` / `--gemma-device` to customize. Once loaded, reward shaping automatically consumes the real embeddings.
