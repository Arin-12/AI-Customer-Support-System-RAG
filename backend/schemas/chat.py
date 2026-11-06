
from pydantic import BaseModel
from uuid import UUID


class ChatRequest(BaseModel):
    user_id: int
    message: str
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    escalated: bool
    ticket_id: str | None = None
