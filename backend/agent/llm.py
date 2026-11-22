"""LLM factory.

Uses Groq-hosted models when GROQ_API_KEY is set. Without a key the app runs in
*offline mode*: routing uses keyword heuristics and answers are extractive (the most
relevant passages are returned with citations). Offline mode keeps the UI, ingestion
and retrieval fully testable without any API key, and is what the test-suite uses.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from backend.config import Settings


# ---------------------------------------------------------------- schemas
class PlanDecision(BaseModel):
    strategy: str = Field(description="DIRECT_ANSWER, RAG_RETRIEVAL or CLARIFICATION")
    reasoning: str = Field(description="One short sentence explaining the decision")
    search_query: str = Field(default="", description="Standalone search query for RAG")


class EvidenceGrade(BaseModel):
    relevant_ids: list[int] = Field(default_factory=list)
    sufficient: bool = False
    better_query: str = ""


# ---------------------------------------------------------------- factory
def build_llms(settings: Settings) -> tuple[Any, Any]:
    """Return (router_llm, answer_llm)."""
    if not settings.groq_api_key:
        offline = OfflineLLM()
        return offline, offline

    from langchain_groq import ChatGroq

    answer = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    router = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.effective_router_model,
        temperature=0,
    )
    return router, answer


def is_offline(llm: Any) -> bool:
    return isinstance(llm, OfflineLLM)


# ---------------------------------------------------------------- offline mode
_DOC_CUES = re.compile(
    r"\b(document|doc|docs|file|files|pdf|upload|uploaded|policy|contract|report|paper|"
    r"section|clause|page|chapter|table|according to|based on|summari[sz]e|compare|"
    r"library|attachment|manual|my notes)\b",
    re.I,
)
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\b", re.I)
_VAGUE = re.compile(r"^\s*(what about (it|that|this)|and (it|that)|explain it|more)\s*\??\s*$", re.I)


def _name_terms(names: list[str]) -> set[str]:
    generic = {"sample", "note", "notes", "technical", "report", "data", "doc", "file", "final", "draft"}
    out: set[str] = set()
    for n in names:
        stem = n.rsplit(".", 1)[0].lower()
        out |= {w for w in re.findall(r"[a-z0-9]+", stem) if len(w) > 2 and w not in generic and w not in _STOP}
    return out


def heuristic_plan(question: str, library_size: int, has_history: bool,
                   library: list[str] | None = None) -> PlanDecision:
    q = question.strip()
    if _VAGUE.match(q) and not has_history:
        return PlanDecision(
            strategy="CLARIFICATION",
            reasoning="The request is ambiguous without more context.",
        )
    if _GREETING.match(q) and len(q.split()) <= 6:
        return PlanDecision(strategy="DIRECT_ANSWER", reasoning="This is a greeting; no retrieval needed.")
    # Topic words that match an indexed document's name (e.g. 'elastic', 'fwi') imply retrieval.
    mentions_library = bool(_terms(q) & _name_terms(library or []))
    if _DOC_CUES.search(q) or mentions_library or (library_size and has_history and len(q.split()) <= 6):
        return PlanDecision(
            strategy="RAG_RETRIEVAL",
            reasoning="This requires document context; retrieving relevant sections.",
            search_query=q,
        )
    return PlanDecision(
        strategy="DIRECT_ANSWER",
        reasoning="This is general knowledge; no document retrieval is needed.",
    )


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to in on for and or is are was were what which who how does do did say says "
    "about with from by at as be this that it its uploaded document documents file policy "
    "please tell me according based".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


_OVERVIEW = re.compile(r"\b(summari[sz]e|summary|overview|key points|compare|comparison)\b", re.I)


def heuristic_grade(question: str, chunks: list[str]) -> EvidenceGrade:
    if _OVERVIEW.search(question) and chunks:
        return EvidenceGrade(relevant_ids=list(range(1, len(chunks) + 1)), sufficient=True)
    q_terms = _terms(question)
    relevant = []
    for i, text in enumerate(chunks, start=1):
        overlap = len(q_terms & _terms(text))
        if q_terms and overlap / len(q_terms) >= 0.34:
            relevant.append(i)
    return EvidenceGrade(
        relevant_ids=relevant,
        sufficient=bool(relevant),
        better_query="" if relevant else " ".join(sorted(q_terms)),
    )


class _OfflineStructured:
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def invoke(self, messages: list[BaseMessage], **_: Any) -> BaseModel:
        meta = getattr(messages[-1], "additional_kwargs", {}).get("offline", {})
        if self.schema is PlanDecision:
            return heuristic_plan(meta["question"], meta["library_size"], meta["has_history"], meta.get("library"))
        if self.schema is EvidenceGrade:
            return heuristic_grade(meta["question"], meta["chunks"])
        raise ValueError(f"Unsupported schema {self.schema}")


def _excerpt(text: str, limit: int = 700) -> str:
    """Trim a passage to whole sentences (or lines) within the limit."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("\n"))
    return (cut[: end + 1] if end > limit * 0.5 else cut.rsplit(" ", 1)[0]).rstrip() + " …"


class OfflineLLM:
    """Minimal stand-in for a chat model when no API key is configured."""

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> _OfflineStructured:
        return _OfflineStructured(schema)

    def invoke(self, messages: list[BaseMessage], **_: Any) -> AIMessage:
        meta = getattr(messages[-1], "additional_kwargs", {}).get("offline", {})
        kind = meta.get("kind")
        if kind == "rag":
            chunks: list[str] = meta.get("chunks", [])
            if not chunks:
                return AIMessage(content="The available documents do not contain enough information "
                                         "to answer this question.")
            parts = ["> " + _excerpt(c).replace("\n", "\n> ") + f" [{i}]"
                     for i, c in enumerate(chunks[:3], start=1)]
            return AIMessage(
                content="**Offline mode** (set `GROQ_API_KEY` for generated answers). "
                "Most relevant passages:\n\n" + "\n\n".join(parts)
            )
        if kind == "clarify":
            return AIMessage(content="Could you clarify what you mean? For example, which document "
                                     "or topic are you asking about?")
        return AIMessage(
            content="**Offline mode:** no LLM is configured, so I can't answer general questions yet. "
            "Add `GROQ_API_KEY` to your `.env`, or ask about your uploaded documents."
        )


__all__ = [
    "BaseChatModel",
    "EvidenceGrade",
    "OfflineLLM",
    "PlanDecision",
    "build_llms",
    "heuristic_grade",
    "heuristic_plan",
    "is_offline",
]
