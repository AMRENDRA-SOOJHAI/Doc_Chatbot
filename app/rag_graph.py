# app/rag_graph.py

from pathlib import Path

from app.generator import generate_answer
from app.ingest import load_pdf, load_txt
from app.retriever import embed_texts, retrieve


def compute_confidence(sim_scores: list[float]) -> float:
    """Calculate confidence score from similarity scores"""
    if not sim_scores:
        return 0.0

    avg_score = sum(sim_scores) / len(sim_scores)

    # cosine similarity is already between [-1, 1], mostly [0,1] for embeddings
    confidence = max(0.0, min(1.0, avg_score))
    return round(confidence, 2)


_DEFAULT_DOCUMENTS: list[str] | None = None
_DEFAULT_EMBEDDINGS = None


def _load_default_documents() -> list[str]:
    """
    Best-effort local loader for notebook/CLI usage.

    The FastAPI app preloads documents on startup, but interactive callers often
    import `DOCUMENTS` / `DOC_EMBEDDINGS` from `main` without running startup.
    """
    docs: list[str] = []
    # Resolve from repo root (.. from app/), not current working directory.
    repo_root = Path(__file__).resolve().parent.parent
    txt_path = repo_root / "data" / "input.txt"
    pdf_path = repo_root / "data" / "doc.pdf"
    if txt_path.exists():
        docs.extend(load_txt(str(txt_path)))
    if pdf_path.exists():
        docs.extend(load_pdf(str(pdf_path)))
    return docs


def _get_default_index():
    global _DEFAULT_DOCUMENTS, _DEFAULT_EMBEDDINGS

    if _DEFAULT_DOCUMENTS is None or _DEFAULT_EMBEDDINGS is None:
        _DEFAULT_DOCUMENTS = _load_default_documents()
        _DEFAULT_EMBEDDINGS = (
            embed_texts(_DEFAULT_DOCUMENTS) if _DEFAULT_DOCUMENTS else None
        )
    return _DEFAULT_DOCUMENTS, _DEFAULT_EMBEDDINGS


def rag(question, documents=None, doc_embeddings=None, k=2):
    """
    RAG pipeline using LCEL chains

    Args:
        question: User's question
        documents: List of documents
        doc_embeddings: Pre-computed embeddings for documents
        k: Number of documents to retrieve

    Returns:
        answer: Generated answer from LLM
        retrieved_docs: List of retrieved documents
        confidence: Confidence score (0-1)
    """
    # Notebook/CLI fallback: if nothing is provided, load from local `data/`.
    if not documents:
        documents, doc_embeddings = _get_default_index()

    if not documents:
        raise ValueError(
            "No documents loaded. Provide `documents` or ensure `data/input.txt` "
            "and/or `data/doc.pdf` exist."
        )

    if doc_embeddings is None or len(documents) != len(doc_embeddings):
        doc_embeddings = embed_texts(documents)

    # Retrieve relevant documents
    retrieved_docs, sim_scores = retrieve(question, documents, doc_embeddings, k=k)

    # Combine retrieved documents into context
    context = "\n".join(retrieved_docs)

    # Generate answer using LCEL chain
    answer = generate_answer(question, context)

    # Compute confidence
    confidence = compute_confidence(sim_scores)

    return answer, retrieved_docs, confidence
