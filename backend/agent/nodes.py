"""LangGraph nodes: planner, retriever, evidence grader, rewriter, generators, responder."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.agent import prompts
from backend.agent.llm import _GREETING, EvidenceGrade, PlanDecision, heuristic_grade, heuristic_plan, is_offline
from backend.agent.state import AgentState, Chunk

log = logging.getLogger(__name__)

VALID_STRATEGIES = {"DIRECT_ANSWER", "RAG_RETRIEVAL", "CLARIFICATION"}


class Retriever(Protocol):
    def search(self, query: str, top_k: int, doc_ids: list[str] | None = None) -> list[Chunk]: ...
    def list_documents(self) -> list[dict]: ...


def _step(node: str, detail: str) -> dict:
    return {"trace": [{"node": node, "detail": detail}]}


class AgentNodes:
    def __init__(self, router_llm: Any, answer_llm: Any, retriever: Retriever, *,
                 top_k: int = 6, max_attempts: int = 2, history_turns: int = 6, probe_library: bool = True):
        self.router_llm = router_llm
        self.answer_llm = answer_llm
        self.retriever = retriever
        self.top_k = top_k
        self.max_attempts = max_attempts
        self.history_turns = history_turns
        self.probe_library = probe_library
        self.offline = is_offline(answer_llm)

    # ------------------------------------------------------------ helpers
    def _human(self, content: str, **offline_meta: Any) -> HumanMessage:
        # Offline metadata is only attached for the offline stand-in model.
        kwargs = {"offline": offline_meta} if self.offline else {}
        return HumanMessage(content=content, additional_kwargs=kwargs)

    def _history_messages(self, state: AgentState) -> list[BaseMessage]:
        msgs: list[BaseMessage] = []
        for turn in (state.get("history") or [])[-self.history_turns * 2:]:
            content = str(turn.get("content", ""))[:2000]
            if turn.get("role") == "assistant":
                msgs.append(AIMessage(content=content))
            else:
                msgs.append(HumanMessage(content=content))
        return msgs

    def _history_text(self, state: AgentState) -> str:
        lines = [f"{m.type}: {m.content}" for m in self._history_messages(state)]
        return "\n".join(lines) if lines else "(none)"

    def _library_names(self) -> list[str]:
        try:
            return [d["document"] for d in self.retriever.list_documents()]
        except Exception:  # pragma: no cover - vector store unavailable
            log.exception("Could not list documents")
            return []

    # ------------------------------------------------------------ 1. planner
    def planner(self, state: AgentState) -> dict:
        question = state["question"]
        library = self._library_names()
        has_history = bool(state.get("history"))
        system = prompts.PLANNER_SYSTEM.format(library=", ".join(library) or "(empty)")
        user = f"Conversation so far:\n{self._history_text(state)}\n\nLatest message: {question}"
        try:
            decision: PlanDecision = self.router_llm.with_structured_output(PlanDecision).invoke(
                [SystemMessage(content=system),
                 self._human(user, question=question, library_size=len(library), has_history=has_history,
                             library=library)]
            )
        except Exception:
            log.exception("Planner LLM failed; using heuristic routing")
            decision = heuristic_plan(question, len(library), has_history, library)

        strategy = decision.strategy.strip().upper()
        if strategy not in VALID_STRATEGIES:
            strategy = "RAG_RETRIEVAL" if library else "DIRECT_ANSWER"
        query = (decision.search_query or question).strip()
        reasoning = decision.reasoning
        if strategy == "DIRECT_ANSWER" and self._matches_library(state):
            strategy = "RAG_RETRIEVAL"
            query = question
            reasoning = "The question matches content in your documents; retrieving relevant sections."
        return {
            "strategy": strategy,
            "reasoning": reasoning,
            "search_query": query,
            "attempts": 0,
            "queries_tried": [],
            **_step("planner", f"Strategy: {strategy}. {reasoning}"),
        }

    def _matches_library(self, state: AgentState) -> bool:
        """Cheap check (one vector search, no LLM call): does the question match indexed content?"""
        question = state["question"]
        if not self.probe_library or _GREETING.match(question) or len(question.split()) < 3:
            return False
        try:
            hits = self.retriever.search(question, 3, state.get("doc_ids"))
        except Exception:  # pragma: no cover - vector store unavailable
            return False
        return bool(hits) and bool(heuristic_grade(question, [h["text"] for h in hits]).relevant_ids)

    # ------------------------------------------------------------ 2. retrieval
    def retrieve(self, state: AgentState) -> dict:
        query = state.get("search_query") or state["question"]
        attempts = state.get("attempts", 0) + 1
        library_empty = not self._library_names()
        chunks = [] if library_empty else self.retriever.search(query, self.top_k, state.get("doc_ids"))
        detail = (
            "Library is empty; nothing to search."
            if library_empty
            else f"Attempt {attempts}: searched Qdrant for \"{query}\" -> {len(chunks)} chunks."
        )
        return {
            "retrieved": chunks,
            "attempts": attempts,
            "queries_tried": [*state.get("queries_tried", []), query],
            "library_empty": library_empty,
            **_step("retrieve", detail),
        }

    # ------------------------------------------------------------ 3. relevance / sufficiency
    def grade(self, state: AgentState) -> dict:
        chunks = state.get("retrieved") or []
        question = state["question"]
        if not chunks:
            return {"relevant": [], "sufficient": False, "next_query": "",
                    **_step("grade", "No chunks retrieved; evidence insufficient.")}

        numbered = "\n\n".join(f"[{i}] ({c['document']}) {c['text'][:1500]}" for i, c in enumerate(chunks, 1))
        try:
            grade: EvidenceGrade = self.router_llm.with_structured_output(EvidenceGrade).invoke(
                [SystemMessage(content=prompts.GRADER_SYSTEM),
                 self._human(f"Question: {question}\n\nChunks:\n{numbered}",
                             question=question, chunks=[c["text"] for c in chunks])]
            )
        except Exception:
            log.exception("Grader LLM failed; using lexical grading")
            grade = heuristic_grade(question, [c["text"] for c in chunks])

        ids = [i for i in dict.fromkeys(grade.relevant_ids) if 1 <= i <= len(chunks)]
        relevant = [chunks[i - 1] for i in ids]
        # Keep the best relevant evidence across retries.
        if not grade.sufficient and state.get("relevant"):
            seen = {c["chunk_id"] for c in relevant}
            relevant += [c for c in state["relevant"] if c["chunk_id"] not in seen]
        sufficient = bool(grade.sufficient and relevant)
        return {
            "relevant": relevant,
            "sufficient": sufficient,
            "next_query": grade.better_query.strip(),
            **_step("grade", f"{len(relevant)}/{len(chunks)} chunks relevant; "
                             f"evidence {'sufficient' if sufficient else 'insufficient'}."),
        }

    def rewrite(self, state: AgentState) -> dict:
        tried = state.get("queries_tried", [])
        candidate = state.get("next_query") or ""
        if not candidate or candidate in tried:
            candidate = f"{state['question']} {state.get('search_query', '')}".strip()
            if candidate in tried:
                candidate = state["question"]
        return {"search_query": candidate,
                **_step("rewrite", f"Rewrote query -> \"{candidate}\"")}

    # ------------------------------------------------------------ 4. generators
    def generate(self, state: AgentState) -> dict:
        question = state["question"]
        relevant = state.get("relevant") or []

        if state.get("library_empty"):
            return {"answer": "No documents are indexed yet. Upload a file in the **Library** panel "
                              "and ask again.",
                    "confidence": "n/a", "sources": [],
                    **_step("generate", "Library empty; asked the user to upload documents.")}
        if not relevant:
            return {"answer": "I searched your documents but couldn't find information that answers "
                              "this question. Try rephrasing, or check that the right file is uploaded.",
                    "confidence": "low", "sources": [],
                    **_step("generate", "No relevant evidence; returned an insufficient-context answer.")}

        context = "\n\n".join(
            f"[{i}] Source: {c['document']}" + (f", page {c['page']}" if c.get("page") else "") + f"\n{c['text']}"
            for i, c in enumerate(relevant, 1)
        )
        msg = self.answer_llm.invoke(
            [SystemMessage(content=prompts.GENERATOR_SYSTEM.format(context=context)),
             *self._history_messages(state),
             self._human(question, kind="rag", chunks=[c["text"] for c in relevant])]
        )
        if state.get("sufficient"):
            confidence = "high" if state.get("attempts", 1) == 1 else "medium"
        else:
            confidence = "low"
        sources = [
            {"index": i, "document": c["document"], "doc_id": c["doc_id"], "page": c.get("page"),
             "chunk_id": c["chunk_id"], "score": round(float(c.get("score", 0.0)), 4),
             "snippet": c["text"][:280]}
            for i, c in enumerate(relevant, 1)
        ]
        return {"answer": str(msg.content), "confidence": confidence, "sources": sources,
                **_step("generate", f"Generated grounded answer from {len(relevant)} chunks.")}

    def direct_answer(self, state: AgentState) -> dict:
        msg = self.answer_llm.invoke(
            [SystemMessage(content=prompts.DIRECT_SYSTEM), *self._history_messages(state),
             self._human(state["question"], kind="direct")]
        )
        return {"answer": str(msg.content), "confidence": "n/a", "sources": [],
                **_step("direct_answer", "Answered from general knowledge (no retrieval).")}

    def clarify(self, state: AgentState) -> dict:
        library = ", ".join(self._library_names()) or "(empty)"
        msg = self.answer_llm.invoke(
            [SystemMessage(content=prompts.CLARIFY_SYSTEM.format(library=library)),
             *self._history_messages(state), self._human(state["question"], kind="clarify")]
        )
        return {"answer": str(msg.content), "confidence": "n/a", "sources": [],
                **_step("clarify", "Asked a clarifying question.")}

    # ------------------------------------------------------------ 5. response
    def respond(self, state: AgentState) -> dict:
        return {"confidence": state.get("confidence", "n/a"),
                **_step("respond", "Formatted final response.")}

    # ------------------------------------------------------------ routers (conditional edges)
    @staticmethod
    def route_after_planner(state: AgentState) -> str:
        return {"DIRECT_ANSWER": "direct_answer", "CLARIFICATION": "clarify"}.get(
            state["strategy"], "retrieve")

    def route_after_grade(self, state: AgentState) -> str:
        if state.get("sufficient") or state.get("library_empty"):
            return "generate"
        if state.get("attempts", 0) < self.max_attempts:
            return "rewrite"
        return "generate"
