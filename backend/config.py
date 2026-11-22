"""Application settings, loaded from environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "AgenticRAG-document-assistant"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "*"
    max_upload_mb: int = 30

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    # Smaller/faster model used for routing and grading. Falls back to llm_model.
    router_model: str = ""
    llm_temperature: float = 0.1

    # --- Vector store (Qdrant) ---
    # Set QDRANT_URL for a server (Docker / Qdrant Cloud). If empty, an embedded
    # on-disk Qdrant is used at QDRANT_PATH, so the app runs without Docker.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = str(ROOT_DIR / "data" / "qdrant")
    qdrant_collection: str = "agenticrag_documents"

    # --- Embeddings ---
    # "fastembed" (local ONNX model, no API key) or "hash" (offline, tests only)
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Ingestion ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    upload_dir: str = str(ROOT_DIR / "documents")

    # --- Agent ---
    top_k: int = 6
    max_retrieval_attempts: int = 2  # initial retrieval + (n-1) rewrites
    history_turns: int = 6  # previous turns passed to the planner
    # Before answering directly, search the library once; if the question matches indexed
    # content, retrieve instead. Lets any question about any document be answered from it.
    probe_library: bool = True

    @property
    def effective_router_model(self) -> str:
        return self.router_model or self.llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
