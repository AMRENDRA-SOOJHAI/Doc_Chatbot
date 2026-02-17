from pydantic import BaseModel


# Pydantic models
class QuestionRequest(BaseModel):
    """Request model for asking questions"""

    question: str
    k: int = 2  # Number of context documents to retrieve


class QuestionResponse(BaseModel):
    """Response model for question answers"""

    question: str
    k: int
    answer: str
    contexts: list[str]
    confidence: float
