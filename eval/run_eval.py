"""Small evaluation harness.

Measures, on your own machine and with your configured models:
  - routing accuracy of the planner against eval/routing_cases.jsonl
  - retrieval hit rate: for RAG cases, whether any retrieved source contains the expected text
  - latency per question

Usage:
    python -m eval.run_eval                       # uses .env settings
    python -m eval.run_eval --offline             # heuristic router + hash embeddings (no keys)

No numbers are pre-filled in this repository; run it to produce your own report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from backend.agent.graph import build_graph
from backend.agent.llm import build_llms
from backend.config import Settings
from backend.ingestion.indexer import ingest_path
from backend.retrieval.embeddings import build_embedder
from backend.retrieval.qdrant_client import build_client
from backend.retrieval.retriever import VectorStore

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "eval" / "routing_cases.jsonl"
SAMPLES = [ROOT / "documents" / n for n in
           ("sample-refund-policy.md", "sample-support-plans.csv")]

# Text that should appear in the retrieved sources for each RAG question.
EXPECTED_EVIDENCE = {
    "Section 4.2": "7 business days",
    "express shipping": "1 to 2 business days",
    "marketing data": "Customer Support",
    "Final Sale": "Final Sale",
    "outdoor equipment": "12-month",
    "order records": "seven years",
    "Enterprise plan": "1 hour",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="No API keys: heuristic routing, hash embeddings")
    args = ap.parse_args()

    overrides = dict(qdrant_url="", qdrant_path=":memory:", qdrant_collection="eval_docs")
    if args.offline:
        overrides.update(groq_api_key="", embedding_provider="hash")
    settings = Settings(**overrides)

    store = VectorStore(build_client(settings), settings.qdrant_collection, build_embedder(settings))
    for sample in SAMPLES:
        ingest_path(store, sample, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    router, answer = build_llms(settings)
    graph = build_graph(router, answer, store, top_k=settings.top_k,
                        max_attempts=settings.max_retrieval_attempts)

    rows, latencies = [], []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        t0 = time.perf_counter()
        out = graph.invoke({"question": case["question"], "trace": []})
        dt = time.perf_counter() - t0
        latencies.append(dt)
        hit = None
        for key, needle in EXPECTED_EVIDENCE.items():
            if key.lower() in case["question"].lower():
                hit = any(needle.lower() in c["text"].lower() for c in out.get("relevant", []))
        rows.append({**case, "got": out["strategy"], "correct": out["strategy"] == case["expected"],
                     "evidence_hit": hit, "attempts": out.get("attempts", 0), "seconds": round(dt, 2)})

    print(f"{'expected':<15}{'got':<15}{'ok':<4}{'evid':<6}{'att':<4}{'sec':<6}question")
    for r in rows:
        ev = "-" if r["evidence_hit"] is None else ("yes" if r["evidence_hit"] else "NO")
        print(f"{r['expected']:<15}{r['got']:<15}{'✓' if r['correct'] else '✗':<4}{ev:<6}{r['attempts']:<4}"
              f"{r['seconds']:<6}{r['question']}")

    acc = sum(r["correct"] for r in rows) / len(rows)
    ev_rows = [r for r in rows if r["evidence_hit"] is not None]
    ev_rate = sum(r["evidence_hit"] for r in ev_rows) / len(ev_rows) if ev_rows else 0.0
    retries = sum(1 for r in rows if r["attempts"] > 1)
    print(f"\nRouting accuracy : {acc:.0%} ({sum(r['correct'] for r in rows)}/{len(rows)})")
    print(f"Evidence hit rate: {ev_rate:.0%} ({sum(r['evidence_hit'] for r in ev_rows)}/{len(ev_rows)})")
    print(f"Retry rate       : {retries}/{len(rows)} questions needed a query rewrite")
    print(f"Median latency   : {statistics.median(latencies):.2f}s")

    out_path = ROOT / "eval" / "last_report.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
