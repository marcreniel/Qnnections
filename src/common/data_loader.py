"""Data loader for Connections puzzles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .embeddings import get_word_embedding

WORDS_PER_PUZZLE = 16
GROUP_SIZE = 4

class DataLoader:
    """Loads and normalizes Connections puzzles."""

    def __init__(self, data_path: str | Path, embedding_source: str = "gemma"):
        self.data_path = Path(data_path)
        self.embedding_source = embedding_source
        self.puzzles: List[Dict] = []

    def load_data(self) -> None:
        """Loads puzzles from the JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with self.data_path.open("r", encoding="utf-8") as fp:
            raw_data = json.load(fp)

        # Handle different JSON structures if necessary. 
        # Assuming raw_data is a list of puzzles based on previous file inspection (implied).
        if isinstance(raw_data, list):
            self.puzzles = [self._normalize_puzzle(p) for p in raw_data]
        else:
            # If it's a dict with a key like 'puzzles'
            self.puzzles = [self._normalize_puzzle(p) for p in raw_data.get("puzzles", [])]
        
        print(f"Loaded {len(self.puzzles)} puzzles.")

    def _normalize_puzzle(self, raw_puzzle: Dict) -> Dict:
        """Normalizes a raw puzzle dictionary."""
        # Extract words and groups
        # Expected format: 
        # {
        #   "id": ...,
        #   "answers": [ { "level": 0, "members": [...] }, ... ] 
        #   OR "groups": ...
        # }
        # Based on env.py, it seems to handle "groups" or "answers".
        # Let's standardize on a flat list of words and a list of group IDs.
        
        words = []
        group_ids = []
        
        # Try to parse groups
        groups = raw_puzzle.get("groups") or raw_puzzle.get("answers")
        if not groups:
            # Fallback or error
            raise ValueError(f"Puzzle missing groups/answers: {raw_puzzle.keys()}")

        all_words = []
        for group_idx, group in enumerate(groups):
            members = group.get("members") or group.get("words")
            if not members:
                continue
            for word in members:
                all_words.append((word, group_idx))
        
        # Shuffle words to ensure no ordering bias if not already shuffled
        # But for reproducibility, maybe keep them or sort them? 
        # The game presents them shuffled. Let's just take them as is, 
        # but usually they come grouped in the JSON. We MUST shuffle them 
        # or the agent will learn the order.
        
        # For now, let's just store them. The Environment can shuffle if needed, 
        # or we shuffle here. Let's shuffle here deterministically.
        import random
        rng = random.Random(str(raw_puzzle.get("id", "0")))
        rng.shuffle(all_words)
        
        words = [w[0] for w in all_words]
        group_ids = [w[1] for w in all_words]
        
        if len(words) != WORDS_PER_PUZZLE:
             # Some puzzles might be weird, skip or pad? 
             # For now, assume strict 16 words.
             pass

        return {
            "id": raw_puzzle.get("id"),
            "words": words,
            "group_ids": group_ids,
            "raw": raw_puzzle
        }

    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """Generates embeddings for a list of words."""
        embeddings = []
        for word in words:
            vec = get_word_embedding(word, source=self.embedding_source)
            embeddings.append(vec)
        return np.stack(embeddings)

    def get_dataset(self) -> List[Dict]:
        """Returns the processed dataset with embeddings pre-computed (optional)."""
        # To save time during training, we could pre-compute all embeddings.
        # For now, let's just return the normalized puzzles.
        return self.puzzles
