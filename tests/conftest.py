"""Test fixtures: offline LLM, hash embeddings and in-memory Qdrant (no network, no keys)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "documents" / "sample-refund-policy.md"


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key="",
        embedding_provider="hash",
        qdrant_url="",
        qdrant_path=":memory:",
        qdrant_collection="test_docs",
        upload_dir=str(tmp_path / "uploads"),
        chunk_size=400,
        chunk_overlap=50,
    )


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture()
def loaded_client(client):
    with SAMPLE.open("rb") as fh:
        r = client.post("/upload", files=[("files", (SAMPLE.name, fh, "text/markdown"))])
    assert r.status_code == 200, r.text
    return client
