# AgenticRAG-document-assistant — Project Presentation

*An agentic RAG system that decides how to answer a question before it answers it.*
Dr. Hesham Moubarak · github.com/hmoubarak72/AgenticRAG-document-assistant

This document collects the five deliverables for presenting the project: a LinkedIn post, a
GitHub README introduction, a node-by-node architecture explanation, a "Why this matters"
section, and a list of future improvements. It describes what the code in this repository
does; it contains no benchmark results, user counts or production claims.

---

## 1. LinkedIn post

Most RAG systems follow a script:
Question → search the vector database → generate an answer.

Some questions are general knowledge. Some need private documents. Sometimes the retrieved context doesn't answer it. And sometimes the right response is a clarifying question.

So I built AgenticRAG-document-assistant, an agentic RAG system that decides how to answer before it answers.

How it works (LangGraph):

1️⃣ Planner: classifies intent as a direct answer, document retrieval, or clarification
2️⃣ Retrieval: turns the request into a semantic query and searches Qdrant
3️⃣ Relevance check: grades whether the chunks are relevant and sufficient, and retries with a rewritten query (up to a limit) when they aren't
4️⃣ Generator: answers only from that evidence, with citations, and says plainly when the documents don't cover it

User: "What is the capital of France?"
Assistant: "This is general knowledge; no document retrieval is needed."
User: "What does Section 4.2 of the uploaded policy say about refunds?"
Assistant: "This requires document context; retrieving relevant policy sections."

LangGraph's conditional edges and shared agent state make every routing decision explicit.

🛠️ FastAPI · LangGraph + LangChain · Groq (Llama 3) · Qdrant · local embeddings · PDF, DOCX, XLSX and PPTX ingestion · HTML/CSS/JS

Ask any question: the agent checks your library first, so you never need to name a file.

The next generation of AI applications won't come only from bigger models. It will come from better architecture: systems that plan, check their evidence, and know when not to answer.

🔗 github.com/hmoubarak72/AgenticRAG-document-assistant

#AI #LangGraph #RAG #AgenticAI #LLM #Qdrant #FastAPI #MachineLearning

---

## 2. GitHub README introduction

> **AgenticRAG-document-assistant is an Agentic RAG system that decides *how* to answer a question before answering it.**
>
> A traditional RAG pipeline sends every question to a vector database, whether or not it needs
> one. AgenticRAG-document-assistant first reasons about the user's intent and picks a strategy:
> answer directly, retrieve from documents, or ask a clarifying question. When it retrieves, it
> grades the evidence for relevance and sufficiency, rewrites the query if the evidence is weak,
> and only then generates a grounded answer with citations. When the documents don't contain the
> answer, it says so instead of guessing.
>
> Built with **FastAPI, LangGraph, LangChain, Groq and Qdrant**. Documents (PDF, DOCX, XLSX, PPTX,
> CSV, TXT, Markdown, JSON, HTML) are parsed, chunked, embedded locally and indexed in Qdrant.
> LangGraph manages the stateful workflow and its conditional edges; Groq-hosted models (Llama 3
> by default, or GPT-OSS) handle planning, grading and generation; and a lightweight HTML/CSS/JS
> interface streams each step of the agent's reasoning as it runs.

---

## 3. Architecture: node-by-node workflow

```
                    ┌───────────── DIRECT_ANSWER ─────────────► Direct answer ──┐
User question ─► Planner ─── CLARIFICATION ─────────────────────► Clarify ───────┤
                    └── RAG_RETRIEVAL ─► Retrieval ─► Relevance / Sufficiency    │
                                            ▲            │                       │
                                            │  insufficient & attempts < max     │
                                            └── Rewrite ◄┘                       │
                                                         │ sufficient, or        │
                                                         │ retry limit reached   │
                                                         ▼                       ▼
                                                     Generator ─────────► Response ─► Answer
```

**1. Planner node.** Reads the latest message, recent conversation history and the list of
indexed documents, and returns a structured decision:
`DIRECT_ANSWER` for greetings, general knowledge and simple explanations;
`RAG_RETRIEVAL` for anything that needs document-specific context;
`CLARIFICATION` for requests too ambiguous to act on.
It also writes a one-sentence reasoning trace and, for retrieval, a standalone search query
that resolves follow-ups such as "what about international orders?". Before committing to a
direct answer, the planner runs one quick search of the library (no LLM call); if the question
matches indexed content, it switches to retrieval, so any question about any uploaded document
is answered from that document even when no file is named.

**2. Retrieval node.** Embeds the search query (FastEmbed, `BAAI/bge-small-en-v1.5`) and queries
Qdrant for the top-k most similar chunks, optionally filtered to documents the user selected.

**3. Relevance / Sufficiency node.** One LLM call grades all retrieved chunks together: which
are relevant, whether together they are sufficient to answer confidently, and, if not, a better
search query. Insufficient evidence routes to a rewrite step and another retrieval attempt. A
retry counter in the agent state caps the loop (`MAX_RETRIEVAL_ATTEMPTS`, default 2), so it
cannot run indefinitely.

**4. Generator node.** Builds a numbered context from the relevant chunks only and produces a
grounded answer with inline `[n]` citations. If the evidence is partial, it answers what is
supported and states what is missing. If nothing relevant was found, it returns a plain
insufficient-context message without calling the model.

**5. Response node.** Returns the answer with the chosen strategy, the planner's reasoning,
the retrieved sources (document, page, similarity score, snippet), a confidence level and the
full step-by-step trace.

**LangGraph's role.** The workflow is a `StateGraph`, not a fixed chain. Conditional edges
implement the branching: after the planner (direct, retrieve or clarify) and after grading
(generate, or rewrite and retry). A shared `AgentState` carries the strategy, retry counter,
queries tried, retrieved and graded chunks and an append-only trace between nodes. That state
is what the API streams to the UI as Server-Sent Events, which makes the agent's decisions
visible to the user.

**Confidence** comes from the grader, not from the model rating itself: `high` when the first
retrieval was sufficient, `medium` when a rewrite was needed, `low` when the retry limit was
reached, and `n/a` for direct answers and clarifications.

**Example behavior**

| User | AgenticRAG-document-assistant |
|---|---|
| "What is the capital of France?" | "This is general knowledge; no document retrieval is needed." |
| "What does Section 4.2 of the uploaded policy say about refunds?" | "This requires document context; retrieving relevant policy sections." |
| "Can Final Sale items be returned?" (no file named) | The library check finds matching content; retrieves the policy section and cites it |
| "what about it?" (no prior context) | Asks a clarifying question |
| A question the documents don't cover | States that the available documents don't contain enough information |

---

## 4. Why this matters: standard RAG vs. agentic RAG

A standard RAG pipeline runs the same path for every request:

```
Question ──► Vector search ──► LLM answer
```

That works for many questions, but the fixed path has predictable failure points:

| | Standard RAG | AgenticRAG-document-assistant |
|---|---|---|
| General questions | Still searches the vector database | Answers directly; no retrieval call |
| Weak or off-topic chunks | Generates from them anyway | Grades evidence; rewrites the query and retries |
| Ambiguous requests | Guesses | Asks a clarifying question |
| Missing information | Risk of a confident, unsupported answer | Says the documents don't cover it |
| Transparency | Answer only | Strategy, reasoning, sources, confidence and trace |
| Control flow | Linear chain | Stateful graph with conditional edges and a retry limit |

The practical result is fewer unnecessary LLM and vector-database calls, answers that stay
grounded in retrieved evidence, and behavior that users can inspect and trust in real document
workflows such as technical reports, policies, contracts, manuals and research papers.

---

## 5. Future improvements

- **Hybrid retrieval and reranking:** combine dense and sparse (BM25) search, then rerank before grading.
- **Token streaming:** stream generated answers word by word, alongside the existing step trace.
- **Multi-document comparison:** structured comparison tables with citations to each source document.
- **OCR for scanned PDFs:** extract text from image-only pages.
- **Evaluation suite:** a larger test set measuring routing accuracy, retrieval precision, answer groundedness and citation accuracy.
- **Tool-using planner:** let the planner choose tools such as summarize, extract tables or compare documents.
- **Accounts and collections:** authentication and per-user document libraries.

---

Agentic architecture is where retrieval-augmented systems go next: the gains come less from
larger models and more from giving models the ability to plan, check their own evidence and
recognize when not to answer. AgenticRAG-document-assistant is built to grow in that direction,
with new tools, data sources, memory and evaluation added as nodes in the same graph.
