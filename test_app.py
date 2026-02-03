"""
Test suite for RAG Chatbot with comprehensive mocking
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.generator import generate_answer
from app.ingest import load_documents, load_pdf, load_txt
from app.rag_graph import compute_confidence, rag
from app.retriever import build_index, embed_texts, retrieve


# Fixtures
@pytest.fixture
def mock_embeddings_model():
    """Mock embeddings model"""
    model = MagicMock()
    model.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    model.embed_query.return_value = [0.9, 0.1]
    return model


@pytest.fixture
def test_documents():
    """Test documents and embeddings"""
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    return docs, embeddings


# Confidence Tests
class TestConfidence:
    """Confidence score computation tests"""

    @pytest.mark.parametrize(
        "scores,expected",
        [
            ([], 0.0),
            ([0.8], 0.8),
            ([0.6, 0.8, 1.0], 0.8),
            ([1.5], 1.0),
            ([-0.5], 0.0),
        ],
    )
    def test_compute_confidence(self, scores, expected):
        assert compute_confidence(scores) == expected


# RAG Pipeline Tests
class TestRAG:
    """RAG pipeline tests"""

    @patch("app.rag_graph.retrieve")
    @patch("app.rag_graph.generate_answer")
    def test_rag_success(self, mock_gen, mock_ret):
        mock_ret.return_value = (["Doc A", "Doc B"], [0.9, 0.7])
        mock_gen.return_value = "Answer"

        answer, contexts, conf = rag(
            "Q", ["D1", "D2"], np.array([[0.1, 0.2], [0.3, 0.4]]), k=2
        )

        assert answer == "Answer"
        assert contexts == ["Doc A", "Doc B"]
        assert conf == 0.8

    @patch("app.rag_graph.retrieve")
    @patch("app.rag_graph.generate_answer")
    def test_rag_no_results(self, mock_gen, mock_ret):
        mock_ret.return_value = ([], [])
        mock_gen.return_value = "No docs"

        _, contexts, conf = rag("Q", ["D1"], np.array([[0.1, 0.2]]), k=1)

        assert contexts == []
        assert conf == 0.0


# Retriever Tests
class TestRetriever:
    """Document retrieval tests"""

    @patch("app.retriever.get_embeddings_model")
    def test_retrieve(self, mock_get):
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.9, 0.1]
        mock_get.return_value = mock_model

        docs, scores = retrieve(
            "Q", ["D1", "D2"], np.array([[0.9, 0.1], [0.1, 0.9]]), k=2
        )

        assert len(docs) == 2
        assert len(scores) == 2

    @patch("app.retriever.embed_texts")
    @patch("app.retriever.load_documents")
    def test_build_index(self, mock_load, mock_embed):
        mock_load.return_value = ["Doc"]
        mock_embed.return_value = np.array([[0.1, 0.2]])

        docs, emb = build_index("test.txt")

        assert docs == ["Doc"]
        assert emb.shape == (1, 2)


# Embeddings Tests
class TestEmbeddings:
    """Text embedding tests"""

    @patch("app.retriever.get_embeddings_model")
    def test_embed_texts(self, mock_get):
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get.return_value = mock_model

        result = embed_texts(["Text1", "Text2"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)


# File Loading Tests
class TestFileLoading:
    """File loading tests"""

    def test_load_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1\nLine 2\n\nLine 3\n")
            temp_path = f.name

        try:
            result = load_txt(temp_path)
            assert result == ["Line 1", "Line 2", "Line 3"]
        finally:
            os.unlink(temp_path)

    def test_load_documents_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content\n")
            temp_path = f.name

        try:
            result = load_documents(temp_path)
            assert "Content" in result
        finally:
            os.unlink(temp_path)

    def test_load_documents_invalid(self):
        with pytest.raises(ValueError):
            load_documents("file.xyz")

    @patch("app.ingest.PyPDFLoader")
    def test_load_pdf(self, mock_pdf):
        doc1, doc2 = MagicMock(), MagicMock()
        doc1.page_content = "Page 1"
        doc2.page_content = "Page 2"

        mock_loader = MagicMock()
        mock_loader.load.return_value = [doc1, doc2]
        mock_pdf.return_value = mock_loader

        result = load_pdf("test.pdf")
        assert result == ["Page 1", "Page 2"]


# API Endpoint Tests
class TestEndpoints:
    """FastAPI endpoint tests"""

    @patch("main.rag")
    def test_ask_success(self, mock_rag):
        from fastapi.testclient import TestClient

        from main import app

        mock_rag.return_value = ("Answer", ["Doc"], 0.85)
        client = TestClient(app)

        resp = client.post("/ask", json={"question": "Q?", "k": 2})
        data = resp.json()

        assert resp.status_code == 200
        assert data["answer"] == "Answer"
        assert data["confidence"] == 0.85

    @patch("main.rag")
    def test_ask_low_confidence(self, mock_rag):
        from fastapi.testclient import TestClient

        from main import app

        mock_rag.return_value = ("Answer", [], 0.1)
        client = TestClient(app)

        resp = client.post("/ask", json={"question": "Q?"})
        assert "not related" in resp.json()["answer"].lower()

    def test_ask_empty_question(self):
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)
        resp = client.post("/ask", json={"question": "   "})
        assert "valid question" in resp.json()["answer"].lower()

    def test_home(self):
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "running" in resp.json()["status"].lower()

    def test_health(self):
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()

        assert "status" in data
        assert "documents_loaded" in data

    def test_stats(self):
        from fastapi.testclient import TestClient

        from main import app

        client = TestClient(app)
        resp = client.get("/stats")
        data = resp.json()

        assert "total_documents" in data
        assert "embedding_dimension" in data


# Error Handling Tests
class TestErrorHandling:
    """Error handling tests"""

    @patch("main.rag")
    def test_ask_exception(self, mock_rag):
        from fastapi.testclient import TestClient

        from main import app

        mock_rag.side_effect = Exception("Error")
        client = TestClient(app)

        resp = client.post("/ask", json={"question": "Q?"})
        assert "Error processing" in resp.json()["answer"]

    @patch("main.rag")
    def test_ask_no_documents(self, mock_rag):
        from fastapi.testclient import TestClient

        import main as main_module
        from main import app

        original = main_module.DOCUMENTS
        try:
            main_module.DOCUMENTS = None
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "Q?"})
            assert "Documents not loaded" in resp.json()["answer"]
        finally:
            main_module.DOCUMENTS = original

    def test_stats_no_embeddings(self):
        from fastapi.testclient import TestClient

        import main as main_module
        from main import app

        original = main_module.DOC_EMBEDDINGS
        try:
            main_module.DOC_EMBEDDINGS = None
            client = TestClient(app)
            data = client.get("/stats").json()
            assert data["embedding_dimension"] == 0
        finally:
            main_module.DOC_EMBEDDINGS = original


# Generator Tests
class TestGenerator:
    """Generator function tests"""

    @patch("app.generator.ChatOpenAI")
    def test_get_llm(self, mock_llm):
        import app.generator
        from app.generator import get_llm

        app.generator.llm = None
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance

        get_llm()
        mock_llm.assert_called_with(model="gpt-4o")

    def test_generate_answer_signature(self):
        import inspect

        sig = inspect.signature(generate_answer)
        assert "question" in sig.parameters
        assert "context" in sig.parameters

    @patch("app.rag_graph.retrieve")
    @patch("app.rag_graph.generate_answer")
    def test_generate_answer_called_in_rag(self, mock_gen, mock_ret):
        mock_ret.return_value = (["D"], [0.8])
        mock_gen.return_value = "Answer"

        answer, _, _ = rag("Q", ["D"], np.array([[0.1]]), k=1)
        mock_gen.assert_called()


# Request/Response Model Tests
class TestModels:
    """Request/response model tests"""

    def test_question_request(self):
        from main import QuestionRequest

        req = QuestionRequest(question="Test?", k=3)
        assert req.question == "Test?"
        assert req.k == 3

        req2 = QuestionRequest(question="Test?")
        assert req2.k == 2  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
