import json


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm"] == "offline"
    assert ".pdf" in body["supported_formats"]


def test_upload_and_list(loaded_client):
    docs = loaded_client.get("/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["document"] == "sample-refund-policy.md"
    assert docs[0]["chunks"] > 1


def test_reupload_does_not_duplicate(loaded_client):
    from tests.conftest import SAMPLE

    before = loaded_client.get("/documents").json()["documents"][0]["chunks"]
    with SAMPLE.open("rb") as fh:
        loaded_client.post("/upload", files=[("files", (SAMPLE.name, fh, "text/markdown"))])
    docs = loaded_client.get("/documents").json()["documents"]
    assert len(docs) == 1 and docs[0]["chunks"] == before


def test_unsupported_file_rejected(client):
    r = client.post("/upload", files=[("files", ("malware.exe", b"MZ....", "application/octet-stream"))])
    assert r.status_code == 400


def test_delete_document(loaded_client):
    doc_id = loaded_client.get("/documents").json()["documents"][0]["doc_id"]
    assert loaded_client.delete(f"/documents/{doc_id}").status_code == 200
    assert loaded_client.get("/documents").json()["documents"] == []


def test_direct_answer_skips_retrieval(loaded_client):
    r = loaded_client.post("/chat", json={"question": "What is the capital of France?"})
    body = r.json()
    assert body["strategy"] == "DIRECT_ANSWER"
    assert body["sources"] == []
    assert not any(step["node"] == "retrieve" for step in body["trace"])


def test_rag_answer_cites_refund_section(loaded_client):
    r = loaded_client.post(
        "/chat", json={"question": "What does Section 4.2 of the uploaded policy say about refunds?"})
    body = r.json()
    assert body["strategy"] == "RAG_RETRIEVAL"
    assert body["sources"], body
    assert any("7 business days" in s["snippet"] or "Refunds" in s["snippet"] for s in body["sources"])
    nodes = [s["node"] for s in body["trace"]]
    assert nodes[:3] == ["planner", "retrieve", "grade"]
    assert nodes[-1] == "respond"


def test_rag_with_empty_library(client):
    body = client.post("/chat", json={"question": "Summarize the uploaded document"}).json()
    assert body["strategy"] == "RAG_RETRIEVAL"
    assert "No documents are indexed" in body["answer"]


def test_clarification(client):
    body = client.post("/chat", json={"question": "what about it?"}).json()
    assert body["strategy"] == "CLARIFICATION"


def test_stream_emits_steps_and_result(loaded_client):
    with loaded_client.stream("POST", "/chat/stream",
                              json={"question": "According to the policy, how long does express shipping take?"}) as r:
        raw = "".join(r.iter_text())
    events = [blk for blk in raw.split("\n\n") if blk.strip()]
    kinds = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert "step" in kinds and kinds[-1] == "result"
    result = json.loads(events[-1].split("data: ", 1)[1])
    assert result["strategy"] == "RAG_RETRIEVAL"



def test_any_question_about_documents_is_searched(loaded_client):
    # No mention of a document or file: the library probe still routes it to retrieval.
    body = loaded_client.post("/chat", json={"question": "Can Final Sale items be returned?"}).json()
    assert body["strategy"] == "RAG_RETRIEVAL"
    assert "matches content in your documents" in body["reasoning"]
    assert body["sources"] and body["sources"][0]["document"] == "sample-refund-policy.md"
    general = loaded_client.post("/chat", json={"question": "What is the capital of France?"}).json()
    assert general["strategy"] == "DIRECT_ANSWER"
