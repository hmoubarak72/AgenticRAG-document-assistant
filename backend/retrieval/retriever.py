"""Vector store facade: upsert chunks, semantic search, document registry."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from qdrant_client import QdrantClient, models

from backend.agent.state import Chunk
from backend.retrieval.embeddings import Embedder
from backend.retrieval.qdrant_client import ensure_collection


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self._lock = threading.Lock()  # embedded Qdrant is not thread-safe
        self._docs_cache: list[dict] | None = None
        ensure_collection(client, collection, embedder.dim)

    # ---------------------------------------------------------------- write
    def add_chunks(self, doc_id: str, document: str, chunks: list[dict], meta: dict | None = None) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c["text"] for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
                vector=vec,
                payload={
                    "doc_id": doc_id,
                    "document": document,
                    "chunk_id": f"{doc_id}-{i}",
                    "chunk_index": i,
                    "page": c.get("page"),
                    "text": c["text"],
                    **(meta or {}),
                },
            )
            for i, (c, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        with self._lock:
            for start in range(0, len(points), 256):
                self.client.upsert(self.collection, points=points[start:start + 256], wait=True)
            self._docs_cache = None
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=self._doc_filter([doc_id])),
                wait=True,
            )
            self._docs_cache = None

    # ---------------------------------------------------------------- read
    @staticmethod
    def _doc_filter(doc_ids: list[str]) -> models.Filter:
        return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])

    def search(self, query: str, top_k: int = 6, doc_ids: list[str] | None = None) -> list[Chunk]:
        vector = self.embedder.embed_query(query)
        with self._lock:
            res = self.client.query_points(
                self.collection,
                query=vector,
                limit=top_k,
                query_filter=self._doc_filter(doc_ids) if doc_ids else None,
                with_payload=True,
            )
        return [
            Chunk(chunk_id=p.payload["chunk_id"], doc_id=p.payload["doc_id"], document=p.payload["document"],
                  page=p.payload.get("page"), text=p.payload["text"], score=float(p.score))
            for p in res.points
        ]

    def list_documents(self) -> list[dict]:
        if self._docs_cache is not None:
            return [dict(d) for d in self._docs_cache]
        docs: OrderedDict[str, dict] = OrderedDict()
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(
                    self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False)
                for p in points:
                    pl = p.payload or {}
                    d = docs.setdefault(pl["doc_id"], {
                        "doc_id": pl["doc_id"], "document": pl["document"], "chunks": 0,
                        "file_type": pl.get("file_type"), "size_bytes": pl.get("size_bytes"),
                        "uploaded_at": pl.get("uploaded_at"),
                    })
                    d["chunks"] += 1
                if offset is None:
                    break
            self._docs_cache = list(docs.values())
        return [dict(d) for d in self._docs_cache]
