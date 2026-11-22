"""Embedding providers.

- ``fastembed``: local ONNX embedding model (default ``BAAI/bge-small-en-v1.5``, 384-d).
  Runs on CPU, no API key; the model is downloaded once on first use.
- ``hash``: deterministic hashed bag-of-words vectors. Offline and instant, intended
  for tests and CI only; retrieval quality is lexical, not semantic.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from backend.config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models use a query instruction prefix; fastembed handles it here.
        return next(iter(self._model.query_embed(text))).tolist()


class HashEmbedder:
    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in self._token.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider.lower() == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)
