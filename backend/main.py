"""FastAPI application: REST API + static frontend.

Run:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent.graph import build_graph
from backend.agent.llm import build_llms, is_offline
from backend.config import ROOT_DIR, Settings, get_settings
from backend.ingestion.indexer import EmptyDocument, ingest_bytes
from backend.ingestion.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType
from backend.retrieval.embeddings import build_embedder
from backend.retrieval.qdrant_client import build_client
from backend.retrieval.retriever import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agenticrag")

FRONTEND_DIR = ROOT_DIR / "frontend"


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = build_embedder(settings)
        self.client = build_client(settings)
        self.store = VectorStore(self.client, settings.qdrant_collection, self.embedder)
        self.router_llm, self.answer_llm = build_llms(settings)
        self.graph = build_graph(
            self.router_llm, self.answer_llm, self.store,
            top_k=settings.top_k, max_attempts=settings.max_retrieval_attempts,
            history_turns=settings.history_turns, probe_library=settings.probe_library,
        )
        log.info("Services ready (llm=%s, qdrant=%s, embeddings=%s)",
                 "offline" if is_offline(self.answer_llm) else settings.llm_model,
                 settings.qdrant_url or settings.qdrant_path, settings.embedding_provider)


# ---------------------------------------------------------------- schemas
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[ChatTurn] = Field(default_factory=list)
    doc_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    strategy: str
    reasoning: str = ""
    confidence: str = "n/a"
    sources: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    latency_ms: int = 0


def _initial_state(req: ChatRequest) -> dict:
    return {
        "question": req.question.strip(),
        "history": [t.model_dump() for t in req.history],
        "doc_ids": req.doc_ids or None,
        "trace": [],
    }


def _to_response(state: dict, started: float) -> ChatResponse:
    return ChatResponse(
        answer=state.get("answer", ""),
        strategy=state.get("strategy", ""),
        reasoning=state.get("reasoning", ""),
        confidence=state.get("confidence", "n/a"),
        sources=state.get("sources", []),
        trace=state.get("trace", []),
        queries=state.get("queries_tried", []),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# ---------------------------------------------------------------- app factory
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = Services(settings)
        yield
        try:
            app.state.services.client.close()
        except Exception:
            pass

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health")
    def health(request: Request):
        s = svc(request)
        return {
            "status": "ok",
            "llm": "offline" if is_offline(s.answer_llm) else s.settings.llm_model,
            "router_model": None if is_offline(s.router_llm) else s.settings.effective_router_model,
            "vector_store": "qdrant-server" if s.settings.qdrant_url else "qdrant-embedded",
            "embeddings": s.settings.embedding_provider,
            "documents": len(s.store.list_documents()),
            "supported_formats": SUPPORTED_EXTENSIONS,
            "max_upload_mb": s.settings.max_upload_mb,
        }

    @app.get("/documents")
    def list_documents(request: Request):
        return {"documents": svc(request).store.list_documents()}

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request):
        svc(request).store.delete_document(doc_id)
        return {"deleted": doc_id}

    @app.post("/upload")
    def upload(request: Request, files: list[UploadFile] = File(...)):
        s = svc(request)
        limit = s.settings.max_upload_mb * 1024 * 1024
        results = []
        for f in files:
            name = f.filename or "document"
            try:
                data = f.file.read(limit + 1)
                if len(data) > limit:
                    raise ValueError(f"File exceeds {s.settings.max_upload_mb} MB limit")
                started = time.perf_counter()
                info = ingest_bytes(s.store, name, data, chunk_size=s.settings.chunk_size,
                                    chunk_overlap=s.settings.chunk_overlap, upload_dir=s.settings.upload_dir)
                info.update(status="ok", seconds=round(time.perf_counter() - started, 2))
                results.append(info)
            except (UnsupportedFileType, EmptyDocument, ValueError) as e:
                results.append({"document": name, "status": "error", "error": str(e)})
            except Exception as e:  # parser failure on a malformed file
                log.exception("Failed to ingest %s", name)
                results.append({"document": name, "status": "error", "error": f"Could not process file: {e}"})
        if results and all(r["status"] == "error" for r in results):
            raise HTTPException(status_code=400, detail=results)
        return {"results": results}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, request: Request):
        started = time.perf_counter()
        state = svc(request).graph.invoke(_initial_state(req))
        return _to_response(state, started)

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, request: Request):
        """Server-Sent Events: one `step` event per node, then a final `result`."""
        graph = svc(request).graph

        def events() -> Iterator[str]:
            started = time.perf_counter()
            state: dict = _initial_state(req)
            try:
                for update in graph.stream(state, stream_mode="updates"):
                    for delta in update.values():
                        delta = delta or {}
                        for k, v in delta.items():
                            state[k] = state.get(k, []) + v if k == "trace" else v
                        for step in delta.get("trace", []):
                            yield f"event: step\ndata: {json.dumps(step)}\n\n"
                yield f"event: result\ndata: {_to_response(state, started).model_dump_json()}\n\n"
            except Exception as e:
                log.exception("Agent run failed")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- frontend
    if FRONTEND_DIR.exists():
        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(FRONTEND_DIR / "index.html")

        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    return app


app = create_app()


if __name__ == "__main__":  # python -m backend.main
    import uvicorn

    st = get_settings()
    uvicorn.run("backend.main:app", host=st.host, port=st.port, reload=False)
