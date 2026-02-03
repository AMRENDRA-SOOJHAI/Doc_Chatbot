# app/rag_graph.py

from app.generator import generate_answer
from app.retriever import retrieve


def compute_confidence(sim_scores: list[float]) -> float:
    """Calculate confidence score from similarity scores"""
    if not sim_scores:
        return 0.0

    avg_score = sum(sim_scores) / len(sim_scores)

    # cosine similarity is already between [-1, 1], mostly [0,1] for embeddings
    confidence = max(0.0, min(1.0, avg_score))
    return round(confidence, 2)


def rag(question, documents, doc_embeddings, k=2):
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
    # Retrieve relevant documents
    retrieved_docs, sim_scores = retrieve(question, documents, doc_embeddings, k=k)

    # Combine retrieved documents into context
    context = "\n".join(retrieved_docs)

    # Generate answer using LCEL chain
    answer = generate_answer(question, context)

    # Compute confidence
    confidence = compute_confidence(sim_scores)

    return answer, retrieved_docs, confidence
