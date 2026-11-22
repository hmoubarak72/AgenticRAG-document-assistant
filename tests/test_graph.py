"""Unit tests for graph routing and the retry limit, using scripted fakes."""

from langchain_core.messages import AIMessage

from backend.agent.graph import build_graph
from backend.agent.llm import EvidenceGrade, PlanDecision


class ScriptedLLM:
    """Returns pre-set structured outputs in order; plain invoke returns a fixed answer."""

    def __init__(self, plan: PlanDecision, grades: list[EvidenceGrade] | None = None):
        self.plan = plan
        self.grades = list(grades or [])
        self.grade_calls = 0

    def with_structured_output(self, schema, **_):
        outer = self

        class _S:
            def invoke(self, _messages, **__):
                if schema is PlanDecision:
                    return outer.plan
                outer.grade_calls += 1
                return outer.grades.pop(0) if outer.grades else EvidenceGrade(sufficient=False, better_query="x")

        return _S()

    def invoke(self, _messages, **_):
        return AIMessage(content="answer [1]")


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def search(self, query, top_k, doc_ids=None):
        self.queries.append(query)
        return self.chunks[:top_k]

    def list_documents(self):
        return [{"doc_id": "d1", "document": "policy.pdf"}]


CHUNK = {"chunk_id": "d1-0", "doc_id": "d1", "document": "policy.pdf", "page": 4,
         "text": "Refunds are issued within 7 business days.", "score": 0.9}


def run(llm, retriever, max_attempts=2):
    graph = build_graph(llm, llm, retriever, max_attempts=max_attempts)
    return graph.invoke({"question": "q", "trace": []})


def test_sufficient_on_first_try():
    llm = ScriptedLLM(PlanDecision(strategy="RAG_RETRIEVAL", reasoning="r", search_query="refund"),
                      [EvidenceGrade(relevant_ids=[1], sufficient=True)])
    retr = FakeRetriever([CHUNK])
    out = run(llm, retr)
    assert out["confidence"] == "high"
    assert out["sources"][0]["page"] == 4
    assert retr.queries == ["refund"]


def test_retry_then_success_rewrites_query():
    llm = ScriptedLLM(PlanDecision(strategy="RAG_RETRIEVAL", reasoning="r", search_query="refund"),
                      [EvidenceGrade(sufficient=False, better_query="refund timeline business days"),
                       EvidenceGrade(relevant_ids=[1], sufficient=True)])
    retr = FakeRetriever([CHUNK])
    out = run(llm, retr)
    assert retr.queries == ["refund", "refund timeline business days"]
    assert out["confidence"] == "medium"
    assert any(s["node"] == "rewrite" for s in out["trace"])


def test_retry_limit_prevents_loops():
    llm = ScriptedLLM(PlanDecision(strategy="RAG_RETRIEVAL", reasoning="r", search_query="refund"))
    retr = FakeRetriever([CHUNK])
    out = run(llm, retr, max_attempts=3)
    assert len(retr.queries) == 3
    assert llm.grade_calls == 3
    assert "couldn't find" in out["answer"]
    assert out["confidence"] == "low"


def test_invalid_strategy_falls_back_safely():
    llm = ScriptedLLM(PlanDecision(strategy="SOMETHING_ELSE", reasoning="r"),
                      [EvidenceGrade(relevant_ids=[1], sufficient=True)])
    out = run(llm, FakeRetriever([CHUNK]))
    assert out["strategy"] == "RAG_RETRIEVAL"  # library non-empty -> safe default


def test_direct_answer_never_touches_retriever():
    llm = ScriptedLLM(PlanDecision(strategy="DIRECT_ANSWER", reasoning="general knowledge"))
    retr = FakeRetriever([CHUNK])
    out = run(llm, retr)
    assert retr.queries == []
    assert out["answer"] == "answer [1]"
