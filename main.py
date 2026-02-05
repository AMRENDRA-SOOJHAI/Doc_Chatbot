"""
Main FastAPI application for RAG Chatbot
Imports all necessary components and sets up the Uvicorn server
"""

import asyncio
import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.ingest import load_pdf, load_txt
from app.rag_graph import rag
from app.retriever import embed_texts
from app.validator import QuestionRequest, QuestionResponse

# Load environment variables
load_dotenv()


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation Chatbot using LangChain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOGGER MIDDLEWARE (LOGS EVERY API CALL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ragchatbot-api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    content_length = request.headers.get("content-length", "0")

    #  Log EVERYTHING
    logger.info(
        f"{request.method} {request.url.path} | "
        f"Status={response.status_code} | "
        f"Time={process_time:.3f}s | "
        f"Body={content_length}"
    )

    return response


# Document store initialized on startup
DOCUMENTS: list[str] = []
DOC_EMBEDDINGS = None
Confidence_threshold = 0.25


def load_documents_and_embeddings():
    """
    Load documents from PDF/TXT and create embeddings once at startup
    """
    docs_txt = load_txt("data/input.txt")
    docs_pdf = load_pdf("data/doc.pdf")
    DOCUMENTS = docs_txt + docs_pdf
    DOC_EMBEDDINGS = embed_texts(DOCUMENTS)

    return DOCUMENTS, DOC_EMBEDDINGS


@app.on_event("startup")
async def startup_event():
    """Startup event (documents already loaded at module level)"""
    global DOCUMENTS, DOC_EMBEDDINGS
    logger.info("Loading documents and building index...")
    DOCUMENTS, DOC_EMBEDDINGS = await asyncio.to_thread(load_documents_and_embeddings)
    logger.info("Documents loaded: %s", len(DOCUMENTS))


# Routes
@app.get("/", tags=["Health"])
def home():
    """Root endpoint - Returns API status"""
    return {
        "status": "✓ RAG Chatbot API is running",
        "docs": "/docs",
        "message": "Go to /docs for Swagger UI to test the API",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    ready = bool(DOCUMENTS) and DOC_EMBEDDINGS is not None
    return {
        "status": "healthy" if DOCUMENTS else "documents_not_loaded",
        "documents_loaded": len(DOCUMENTS) if DOCUMENTS else 0,
        "embeddings_ready": DOC_EMBEDDINGS is not None,
        "message": "Ready" if ready else "Documents or embeddings not loaded",
    }


@app.post("/ask", response_model=QuestionResponse, tags=["Chat"])
def ask_question(payload: QuestionRequest):
    """
    Ask a question to the RAG chatbot

    Parameters:
    - question: The question to ask
    - k: Number of context documents to retrieve (default: 2)

    Returns:
    - question: The user asked question
    - k: Number of retrieved contexts
    - answer: The generated answer from the LLM based on retrieved contexts
    - contexts: List of retrieved relevant documents
    - confidence: Confidence score based on similarity scores (0-1)
    """
    # Validate documents are loaded
    if not DOCUMENTS or DOC_EMBEDDINGS is None:
        raise HTTPException(
            status_code=503,
            detail="Documents or embeddings not loaded. Check your configuration.",
        )

    # Validate question
    if not payload.question or not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid question.",
        )

    try:
        # Get RAG response (retrieve documents and generate answer)
        answer, contexts, confidence = rag(
            payload.question, DOCUMENTS, DOC_EMBEDDINGS, k=payload.k
        )

        # Check if confidence is too low
        if confidence < Confidence_threshold or not contexts:
            raise HTTPException(
                status_code=400,
                detail="Sorry, your question is not related to the uploaded document.",
            )

        return QuestionResponse(
            question=payload.question,
            k=payload.k,
            answer=answer,
            contexts=contexts,
            confidence=float(confidence),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing question: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Error processing question. Please try again later.",
        )


@app.get("/stats", tags=["Info"])
def get_stats():
    """Get statistics about the loaded documents"""
    return {
        "total_documents": len(DOCUMENTS) if DOCUMENTS else 0,
        "embedding_dimension": (
            DOC_EMBEDDINGS.shape[1] if DOC_EMBEDDINGS is not None else 0
        ),
        "total_embeddings": len(DOC_EMBEDDINGS) if DOC_EMBEDDINGS is not None else 0,
    }


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("Starting RAG Chatbot Server...")
    logger.info("Documents loaded: %s", len(DOCUMENTS))
    logger.info("Swagger UI: http://localhost:8000/docs")
    logger.info("ReDoc: http://localhost:8000/redoc")
    logger.info("=" * 60)

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
