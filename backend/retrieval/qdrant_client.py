"""Qdrant connection and collection management."""

from __future__ import annotations

import warnings
from pathlib import Path

from qdrant_client import QdrantClient, models

from backend.config import Settings


def build_client(settings: Settings) -> QdrantClient:
    """Server mode when QDRANT_URL is set; otherwise embedded on-disk mode.

    ``QDRANT_PATH=:memory:`` gives a throwaway in-memory store (used by tests).
    """
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    if settings.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        existing = getattr(vectors, "size", None)
        if existing is not None and existing != dim:
            raise RuntimeError(
                f"Collection '{name}' has vector size {existing}, but the embedding model "
                f"produces {dim}. Use a new QDRANT_COLLECTION or delete the old one."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    for field in ("doc_id", "document"):
        with warnings.catch_warnings():  # embedded mode ignores payload indexes
            warnings.simplefilter("ignore")
            try:
                client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
            except Exception:
                pass
