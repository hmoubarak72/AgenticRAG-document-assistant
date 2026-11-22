# LinkedIn post

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
