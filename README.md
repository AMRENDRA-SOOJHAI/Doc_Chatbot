# RAG Document Chatbot API

A FastAPI-based Retrieval-Augmented Generation (RAG) service that loads a local PDF/TXT corpus, retrieves the most relevant chunks, and generates answers with an LLM.

**Features**
- PDF and TXT ingestion (`data/doc.pdf`, `data/input.txt`).
- In-memory semantic retrieval with OpenAI embeddings.
- RAG answer generation with GPT-4o via LangChain.
- Confidence scoring on retrieved context.
- Health and stats endpoints for monitoring.

**Project Structure**
- `main.py` FastAPI app and API routes.
- `app/ingest.py` document loaders and optional Milvus ingestion.
- `app/retriever.py` embeddings + cosine similarity retrieval.
- `app/generator.py` LLM prompt and answer generation.
- `app/rag_graph.py` RAG orchestration and confidence scoring.
- `data/` local corpus files.

**Requirements**
- Python 3.12
- Poetry
- OpenAI API key
- Optional: Milvus/Zilliz credentials if you use `ingest_to_milvus()`

**Setup**
1. Install dependencies:

```bash
poetry install --with dev
```

2. Create `.env` in the project root:

```bash
OPENAI_API_KEY="your_openai_api_key"
MILVUS_URI="your_milvus_uri"        # optional
MILVUS_TOKEN="your_milvus_token"    # optional
MILVUS_DB_NAME="rag_db"             # optional
MILVUS_COLLECTION="rag_collection"  # optional
```

3. Add your documents:
- `data/input.txt` for text input (one line per chunk)
- `data/doc.pdf` for PDF input

4. Run the API:

```bash
uvicorn main:app --reload
```

**API Endpoints**
- `GET /` basic status
- `GET /health` readiness and document/embedding status
- `GET /stats` corpus and embedding stats
- `POST /ask` ask a question

**Example Request**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "k": 2}'
```

**Response Shape**

```json
{
  "question": "...",
  "k": 2,
  "answer": "...",
  "contexts": ["..."],
  "confidence": 0.72
}
```

**Tests**

```bash
poetry run pytest
```

**Notes**
- Documents and embeddings are loaded at startup. Restart the server after changing files in `data/`.
- The confidence threshold is set in `main.py` via `Confidence_threshold`.
- CORS is currently open for all origins. Tighten for production use.
