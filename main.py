from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chat import router as chat_router
from backend.routes.tickets import router as ticket_router


app = FastAPI(
    title="AI Customer Support System",
    description="AI-powered customer support using RAG, PostgreSQL, and agentic AI",
    version="1.0.0"
)


# Allow the frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(ticket_router)