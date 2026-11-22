"""Ingestion pipeline: extract -> chunk -> embed -> index in Qdrant."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.ingestion.chunker import chunk_sections
from backend.ingestion.loaders import extract_sections
from backend.retrieval.retriever import VectorStore


class EmptyDocument(ValueError):
    pass


def safe_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^\w.\- ()]+", "_", name).strip() or "document"


def ingest_bytes(store: VectorStore, filename: str, data: bytes, *, chunk_size: int, chunk_overlap: int,
                 upload_dir: str | None = None) -> dict:
    filename = safe_filename(filename)
    # Same name + same content -> same id, so re-uploading replaces instead of duplicating.
    doc_id = hashlib.sha1(filename.encode() + b"\0" + data).hexdigest()[:16]

    sections = extract_sections(filename, data)
    chunks = chunk_sections(sections, chunk_size, chunk_overlap)
    if not chunks:
        raise EmptyDocument(f"No extractable text found in '{filename}' (scanned PDFs need OCR first).")

    if upload_dir:
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        (Path(upload_dir) / filename).write_bytes(data)

    store.delete_document(doc_id)
    n = store.add_chunks(doc_id, filename, chunks, meta={
        "file_type": Path(filename).suffix.lower().lstrip("."),
        "size_bytes": len(data),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    pages = sorted({c["page"] for c in chunks if c.get("page")})
    return {"doc_id": doc_id, "document": filename, "chunks": n, "pages": len(pages) or None}


def ingest_path(store: VectorStore, path: str | Path, **kwargs) -> dict:
    p = Path(path)
    return ingest_bytes(store, p.name, p.read_bytes(), **kwargs)
