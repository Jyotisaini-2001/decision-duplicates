"""Local embeddings — same default stack as Cairn (fastembed / bge-small).

Turns decision text into a list of numbers (a vector) so we can compare meaning.
"""

from __future__ import annotations

import math
import os
import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Interface every embedder must follow: name, dim, embed / embed_one."""

    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale a vector to length 1 so cosine similarity = plain dot product."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FastEmbedEmbedder:
    """
    Real semantic embedder used by default.
    Runs locally (ONNX) with BAAI/bge-small-en-v1.5 — no API key.
    Same idea / default model family as Cairn.
    """

    name = "fastembed"

    def __init__(self, model: str | None = None) -> None:
        """Load the embedding model once; set self.dim from a probe encode."""
        from fastembed import TextEmbedding

        self.model_name = model or os.environ.get(
            "EMBED_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self._model = TextEmbedding(model_name=self.model_name)
        self.dim = len(self.embed_one("dimension probe"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert many texts → many vectors (batch). Used when ingesting JSON."""
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_one(self, text: str) -> list[float]:
        """Convert a single text → one vector."""
        return self.embed([text])[0]


class HashingEmbedder:
    """
    Cheap fallback if fastembed is missing.
    Hashes words into fixed buckets — NOT true semantics; only for tests / emergency.
    """

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        """Create an empty hashing space of size `dim`."""
        self.dim = dim

    def embed_one(self, text: str) -> list[float]:
        """Map tokens into a fixed-size bag-of-hashes vector, then normalize."""
        import hashlib

        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(
                hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big"
            )
            vec[h % self.dim] += 1.0
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts by calling embed_one on each."""
        return [self.embed_one(t) for t in texts]


def get_embedder(name: str | None = None) -> Embedder:
    """
    Pick which embedder to use.
    Default: fastembed. Set EMBEDDINGS=hashing to force the fallback.
    If fastembed fails to load, falls back to hashing automatically.
    """
    backend = (name or os.environ.get("EMBEDDINGS", "fastembed")).lower()
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return FastEmbedEmbedder()
    except Exception as err:  # noqa: BLE001
        print(f"fastembed unavailable ({err}); falling back to hashing")
        return HashingEmbedder()
