# Architecture

AgenticRAG-document-assistant is a FastAPI service wrapping a LangGraph state machine. The graph decides, per
request, whether retrieval is needed, runs a bounded retrieve–grade–rewrite loop when it is,
and returns a grounded answer with a full trace of the decisions it made.

## Request lifecycle

```
Browser ──POST /chat/stream──► FastAPI ──graph.stream()──► LangGraph
   ▲                                                          │
   └──── SSE: step, step, step, …, result ◄───────────────────┘
```

1. The frontend sends the question, the last few turns of history and an optional list of
   document IDs to scope the search.
2. FastAPI builds the initial `AgentState` and streams node updates as `step` events.
3. The final `result` event carries the answer, strategy, confidence, sources and trace.

## Agent state

`backend/agent/state.py` defines a `TypedDict` shared by all nodes. The fields fall into
four groups:

| Group | Fields |
|---|---|
| Input | `question`, `history`, `doc_ids` |
| Plan | `strategy`, `reasoning`, `search_query` |
| Retrieval loop | `attempts`, `queries_tried`, `retrieved`, `relevant`, `sufficient`, `next_query`, `library_empty` |
| Output | `answer`, `confidence`, `sources`, `trace` |

`trace` uses an `operator.add` reducer, so every node appends its own step without
overwriting others. This is what makes the streamed trace and the final trace identical.

## Nodes

**Planner.** Receives the question, recent history and the names of indexed documents, and
returns a structured `PlanDecision` (strategy, one-line reasoning, standalone search query)
via the model's structured-output mode. Including the library contents lets it route "what
does the contract say…" to retrieval and "what's a contract?" to a direct answer. If the
model returns an invalid strategy, the node falls back to retrieval when documents exist and
to a direct answer otherwise. If the call fails entirely, keyword heuristics take over so the
request still completes.

**Library check.** When the planner chooses a direct answer and documents are indexed, it runs
one quick vector search (no LLM call). If the top chunks share enough terms with the question, the
strategy switches to retrieval with the reasoning "The question matches content in your
documents". This makes any question about any uploaded document answerable from it, even when
no file is named. Disable it with `PROBE_LIBRARY=false`.

**Retrieve.** Embeds `search_query` with FastEmbed and queries Qdrant (cosine similarity,
`TOP_K` results), applying a `doc_id` filter when the user scoped the question. An empty
library short-circuits to a helpful "upload a document" response.

**Grade.** A single LLM call receives all retrieved chunks, numbered, and returns
`relevant_ids`, `sufficient`, and a `better_query`. Grading the batch in one call keeps
latency roughly constant regardless of `TOP_K`. Relevant chunks from earlier attempts are
kept, so a retry can only add evidence.

**Rewrite.** Uses the grader's `better_query`, skipping any query already in
`queries_tried`, then loops back to Retrieve.

**Generate.** Builds a numbered context block from the relevant chunks only and instructs the
model to cite `[n]`, answer only what the context supports, and say what it doesn't cover.
With no relevant chunks it returns a fixed insufficient-evidence message without calling the
model, which removes one path for hallucination.

**Direct answer / Clarify.** Answer from general knowledge, or ask one targeted question.

**Respond.** Finalizes confidence and closes the trace.

## Conditional edges

```python
planner  → {direct_answer | clarify | retrieve}           # route_after_planner
grade    → generate  if sufficient or library is empty
         → rewrite   if insufficient and attempts < MAX_RETRIEVAL_ATTEMPTS
         → generate  otherwise (answer with what was found, low confidence)
rewrite  → retrieve
```

The retry counter lives in state, so the loop is bounded by construction; it can't run
away even if the grader never reports sufficiency.

## Confidence

Confidence is a simple, explainable signal rather than a model-reported score:

| Value | Condition |
|---|---|
| `high` | Evidence sufficient on the first retrieval |
| `medium` | Evidence sufficient after at least one rewrite |
| `low` | Retry limit reached without sufficient evidence |
| `n/a` | Direct answer or clarification |

## Ingestion

`loaders.py` extracts text per format (PDF pages and PPTX slides keep their numbers),
`chunker.py` splits with LangChain's `RecursiveCharacterTextSplitter`, and `indexer.py`
embeds and upserts into Qdrant. Document IDs are a hash of file name and content, and point
IDs are derived from the document ID and chunk index, so re-uploading the same file replaces
its vectors instead of duplicating them.

## Storage modes

| Mode | Setting | Use |
|---|---|---|
| Embedded on-disk | `QDRANT_URL` empty, `QDRANT_PATH=./data/qdrant` | Local development, single process |
| In-memory | `QDRANT_PATH=:memory:` | Tests and evaluation |
| Server / Cloud | `QDRANT_URL=…` (+ `QDRANT_API_KEY`) | Docker Compose, deployment, multiple workers |

## Offline mode

When `GROQ_API_KEY` is empty, an `OfflineLLM` stands in for the chat model: routing uses
keyword heuristics, grading uses term overlap, and answers quote the top passages with
citations. Combined with the hash embedder and in-memory Qdrant, this lets the whole graph,
API and UI run in CI with no network access or secrets.

## Extending

- **New node:** add a method to `AgentNodes`, register it in `graph.py`, and wire edges.
- **New tool:** let the planner emit a new strategy and route it to a tool node.
- **New file type:** add a loader function to `LOADERS` in `loaders.py`.
- **Different embeddings:** implement `embed_documents` / `embed_query` / `dim` and select it in `build_embedder`.
