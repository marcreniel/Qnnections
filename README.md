# Connections Q-Learning

Tabular Q-learning playground for the NYT-style **Connections** puzzle (16 words split into 4 hidden categories). Three reward variants are supported:

- `baseline`: vanilla NYT rewards
- `glove`: NYT rewards + shaping term computed from local GloVe vectors
- `gemma`: NYT rewards + shaping term computed from google/embeddinggemma-300m vectors 

## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate connections-rl
```

> **Gemma downloads**: Running the Gemma variant for the first time pulls `google/embeddinggemma-300m` from Hugging Face. The loader automatically prefers CUDA, then Apple `mps`, then CPU unless you override `--gemma-device`.

## 2. Data requirements

```
data/
├── embeddings/
│   └── glove.6B.300d.txt  # local vectors loaded by --glove-path
└── raw/
    └── connections.json   # puzzle dataset: 16 words + 4 solution groups each
```


## 3. Training

`train.py` wires together the environment, agent, reward shaping, and action selection. Common invocations:

```bash
python train.py --variant baseline --episodes 5000 \
  --puzzle-path data/raw/connections.json \
  --q-table-out reports/q_tables/baseline.pkl \
  --metrics-plot reports/plots/baseline.png
python train.py --variant glove --lambda_embed 0.8  --episodes 5000 \
  --puzzle-path data/raw/connections.json \
  --glove-path data/embeddings/glove.6B.300d.txt \
  --q-table-out reports/q_tables/glove.pkl \
  --metrics-plot reports/plots/glove.png
python train.py --variant gemma --lambda_embed 0.8 --episodes 5000 \
  --puzzle-path data/raw/connections.json \
  --gemma-model google/embeddinggemma-300m \
  --q-table-out reports/q_tables/gemma.pkl \
  --metrics-plot reports/plots/gemma.png
```

Key flags:

- `--puzzle-path`: dataset location (default `data/raw/connections.json`)
- `--glove-path`: local GloVe file (used automatically for `--variant glove`)
- `--lambda-embed` / `--lambda_embed`: weight for embedding cohesion shaping
- `--gemma-model`, `--gemma-device`: Hugging Face model id + PyTorch device
- `--checkpoint-every`: save interval for the Q-table

Training metrics stream through `tqdm`. Q-tables are persisted via `QLearningAgent.save_q_table`.

## 4. Code map

- `src/env.py`: Connections environment + transition logic
- `src/agent.py`: tabular Q-learning agent with ε-greedy policy and optional LR decay
- `src/reward.py`: NYT base rewards + embedding-based shaping
- `src/embeddings.py`: loaders for GloVe and Gemma (via Hugging Face `transformers`), plus cohesion helpers
- `src/action_selection.py`: enumerates every valid 4-word guess from unused indices
- `src/utils.py`: puzzle loading, seeding, and mask helpers
- `train.py`: CLI entry-point for learning