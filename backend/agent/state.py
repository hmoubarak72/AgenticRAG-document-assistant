"""Shared state passed between LangGraph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Strategy = Literal["DIRECT_ANSWER", "RAG_RETRIEVAL", "CLARIFICATION"]
Confidence = Literal["high", "medium", "low", "n/a"]


class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    document: str
    page: int | None
    text: str
    score: float


class TraceStep(TypedDict):
    node: str
    detail: str


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    doc_ids: list[str] | None  # optional filter to specific documents

    # planner
    strategy: Strategy
    reasoning: str
    search_query: str
    next_query: str

    # retrieval loop
    attempts: int
    queries_tried: list[str]
    retrieved: list[Chunk]
    relevant: list[Chunk]
    sufficient: bool
    library_empty: bool

    # output
    answer: str
    confidence: Confidence
    sources: list[dict]
    # appended to by every node (LangGraph reducer)
    trace: Annotated[list[TraceStep], operator.add]
