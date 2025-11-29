"""Word embedding utilities for reward shaping."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

try:  # Optional heavy deps, only needed for Gemma variants.
    import torch
except Exception:  # pragma: no cover - torch may be unavailable during docs builds
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import]
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

_GLOVE_VECTORS: Dict[str, np.ndarray] = {}
_GLOVE_DIM = 300
_GEMMA_CACHE: Dict[str, np.ndarray] = {}
_GEMMA_DIM = 768
_GEMMA_MODEL_NAME: str | None = None
_GEMMA_MODEL = None
_GEMMA_TOKENIZER = None
_GEMMA_DEVICE: str | None = None


def load_glove_embeddings(glove_path: str | Path, limit: int | None = None) -> Dict[str, np.ndarray]:
    """Load GloVe vectors (e.g., glove.6B.300d.txt) into memory."""

    global _GLOVE_VECTORS, _GLOVE_DIM
    path = Path(glove_path)
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {path}")

    vectors: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp):
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            values = np.asarray(parts[1:], dtype=float)
            vectors[word] = values
            _GLOVE_DIM = values.shape[0]
            if limit is not None and len(vectors) >= limit:
                break
    _GLOVE_VECTORS = vectors
    return vectors


def _pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def load_gemma_embedder(model_name: str = "google/embeddinggemma-300m", device: str | None = None) -> None:
    """Load the Gemma text embedding model through `transformers`."""

    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required to load Gemma embeddings. Install it or pick another variant."
        )
    if torch is None:
        raise ImportError("PyTorch is required to run the Gemma embedding model.")

    global _GEMMA_MODEL, _GEMMA_TOKENIZER, _GEMMA_MODEL_NAME, _GEMMA_DEVICE, _GEMMA_DIM
    if _GEMMA_MODEL is not None and model_name == _GEMMA_MODEL_NAME:
        if device is None or device == _GEMMA_DEVICE:
            return

    resolved_device = _pick_device(device)
    print(
        f"[Gemma] Loading {model_name} on {resolved_device}. "
        "Override with --gemma-device if needed."
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(resolved_device)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size:
        _GEMMA_DIM = int(hidden_size)

    _GEMMA_MODEL = model
    _GEMMA_TOKENIZER = tokenizer
    _GEMMA_MODEL_NAME = model_name
    _GEMMA_DEVICE = resolved_device
    _GEMMA_CACHE.clear()


def _ensure_gemma_ready() -> None:
    if _GEMMA_MODEL is None:
        load_gemma_embedder()


def _pool_hidden_states(outputs, attention_mask):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output[0]
    last_hidden = outputs.last_hidden_state
    if attention_mask is None:
        return last_hidden.mean(dim=1)[0]
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return (summed / counts)[0]


@torch.inference_mode() if torch is not None else (lambda fn: fn)  # type: ignore[misc]
def get_gemma_embedding(word: str) -> np.ndarray:
    """Embed a word/phrase using google/embeddinggemma-300m via transformers."""

    if word in _GEMMA_CACHE:
        return _GEMMA_CACHE[word]

    _ensure_gemma_ready()
    assert _GEMMA_MODEL is not None and _GEMMA_TOKENIZER is not None and _GEMMA_DEVICE is not None

    inputs = _GEMMA_TOKENIZER(
        word,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )
    inputs = {k: v.to(_GEMMA_DEVICE) for k, v in inputs.items()}
    outputs = _GEMMA_MODEL(**inputs)
    pooled = _pool_hidden_states(outputs, inputs.get("attention_mask"))
    vector = pooled.detach().cpu().numpy().astype(np.float32)
    _GEMMA_CACHE[word] = vector
    return vector


def _fallback_vector(word: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((word, dim))) % (2**32))
    return rng.normal(0, 1, size=dim).astype(np.float32)


def get_word_embedding(
    word: str,
    source: str,
    glove_vectors: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Fetch word embeddings from the requested source with deterministic fallbacks."""

    source = source.lower()
    if source == "glove":
        vectors = glove_vectors if glove_vectors is not None else _GLOVE_VECTORS
        lookup_order = [word, word.lower(), word.upper(), word.title()]
        seen: set[str] = set()
        for candidate in lookup_order:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if candidate in vectors:
                return vectors[candidate]
        dim = next(iter(vectors.values())).shape[0] if vectors else _GLOVE_DIM
        return _fallback_vector(word, dim)
    if source == "gemma":
        return get_gemma_embedding(word)
    raise ValueError(f"Unknown embedding source: {source}")


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def group_cohesion(
    words: Sequence[str],
    source: str,
    glove_vectors: Dict[str, np.ndarray] | None = None,
) -> float:
    """Mean pairwise cosine similarity for a 4-word guess."""

    if len(words) == 0:
        return 0.0
    embeddings = [get_word_embedding(word, source, glove_vectors) for word in words]
    sims: list[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(_cosine_similarity(embeddings[i], embeddings[j]))
    if not sims:
        return 0.0
    return float(np.mean(sims))
