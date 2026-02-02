# app/retriever.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from app.ingest import load_documents

_embeddings_model = None

def get_embeddings_model():
    """Get or create embeddings model lazily"""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings_model


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings_model = get_embeddings_model()
    vectors = embeddings_model.embed_documents(texts)
    return np.array(vectors)


def build_index(path="input.txt"):
    documents = load_documents(path)
    doc_embeddings = embed_texts(documents)
    return documents, doc_embeddings


def retrieve(query: str, documents, doc_embeddings, k=2):
    embeddings_model = get_embeddings_model()
    query_vector = np.array(embeddings_model.embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_vector, doc_embeddings)[0]

    top_idx = scores.argsort()[-k:][::-1]

    top_docs = [documents[i] for i in top_idx]
    top_scores = [float(scores[i]) for i in top_idx]

    return top_docs, top_scores
