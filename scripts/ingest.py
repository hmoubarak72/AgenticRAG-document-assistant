"""Bulk-ingest files or folders from the command line.

    python -m scripts.ingest documents/            # every supported file in a folder
    python -m scripts.ingest report.pdf notes.docx  # specific files

Stop the server first when using embedded Qdrant (QDRANT_URL empty): the on-disk
store allows only one process at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.indexer import ingest_path
from backend.ingestion.loaders import LOADERS
from backend.retrieval.embeddings import build_embedder
from backend.retrieval.qdrant_client import build_client
from backend.retrieval.retriever import VectorStore


def iter_files(args: list[str]):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.suffix.lower() in LOADERS)
        elif p.is_file():
            yield p


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    s = get_settings()
    store = VectorStore(build_client(s), s.qdrant_collection, build_embedder(s))
    for f in iter_files(sys.argv[1:]):
        try:
            info = ingest_path(store, f, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
            print(f"✓ {info['document']}: {info['chunks']} chunks")
        except Exception as e:
            print(f"✗ {f.name}: {e}")


if __name__ == "__main__":
    main()
