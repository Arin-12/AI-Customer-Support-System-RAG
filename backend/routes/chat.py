from fastapi import APIRouter, HTTPException

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.ticket_service import TicketService

from ai.agent.orchestrator import (
    AgentOrchestrator,
    rag_tool,
    sql_tool,
)

from ai.agent.router import Route

from ai.llm.llm import GroqLLM


router = APIRouter()

ticket_service = TicketService()


def normalize_user_id(user_id) -> str:
    """
    Convert API user IDs to the database's string ID format.

    Examples:
        1 -> USR001
        5 -> USR005
    """

    if isinstance(user_id, int):
        return f"USR{user_id:03d}"

    user_id = str(user_id)

    if user_id.isdigit():
        return f"USR{int(user_id):03d}"

    return user_id


def ticket_tool(
    query: str,
    user_id: str,
) -> str:
    """
    Create a support ticket for the requesting customer.
    """

    result = ticket_service.create_ticket(
        user_id=user_id,
        subject="Human support request",
        description=query,
    )

    return result["ticket_id"]


def build_fallback_response(state) -> str:
    """
    Convert an internal escalation/failure state into
    a customer-friendly response.

    The orchestrator intentionally keeps response=None
    for some failure states so existing agent tests and
    internal behavior remain unchanged.
    """

    if state.route == Route.UNKNOWN:
        return (
            "I'm not able to confidently answer that question "
            "from the information available. Please provide "
            "more details, such as your order ID or product "
            "name. If you prefer, I can also connect you with "
            "a human support representative."
        )

    if state.route == Route.SQL:
        return (
            "I couldn't verify the requested order information. "
            "Please check your order ID and try again. If you "
            "still need help, I can connect you with a human "
            "support representative."
        )

    if state.route == Route.SQL_AND_RAG:
        return (
            "I couldn't verify the order details from the "
            "information provided. Please provide your order "
            "ID so I can check the order and give you the "
            "correct policy information."
        )

    if state.route == Route.RAG:
        return (
            "I couldn't retrieve enough verified information "
            "to answer that question safely. Please try "
            "rephrasing your question, or I can connect you "
            "with a human support representative."
        )

    if state.route == Route.TICKET:
        return (
            "I've created a support ticket for you. A human "
            "support representative can review your request "
            "and assist you further."
        )

    return (
        "I'm sorry, but I couldn't generate a verified answer "
        "right now. Please try again or request human support."
    )


orchestrator = AgentOrchestrator(
    llm=GroqLLM(),
    rag_tool=rag_tool,
    sql_tool=sql_tool,
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    normalized_user_id = normalize_user_id(
        request.user_id
    )

    # print(
    # f"DEBUG API conversation_id={request.conversation_id}, "
    # f"message={request.message!r}"
    # )

    state = orchestrator.handle(
        request.message,
        user_id=normalized_user_id,
        conversation_id=request.conversation_id,
    )

    ticket_id = state.ticket_id

    # -------------------------------------------------
    # Explicit human-support request
    # -------------------------------------------------
    if (
        state.requires_escalation
        and state.route == Route.TICKET
    ):
        try:

            ticket_id = ticket_tool(
                query=request.message,
                user_id=normalized_user_id,
            )

            state.ticket_id = ticket_id

            state.response = (
                "I've created a support ticket for you.\n\n"
                f"**Ticket ID: {ticket_id}**\n\n"
                "A human support representative can review "
                "your request and assist you further."
            )

        except ValueError as exc:

            raise HTTPException(
                status_code=404,
                detail=str(exc),
            )
    # -------------------------------------------------
    # Customer-facing fallback
    # -------------------------------------------------
    answer = state.response

    if not answer:
        answer = build_fallback_response(state)

    # -------------------------------------------------
    # Response
    # -------------------------------------------------
    return ChatResponse(
        answer=answer,
        confidence=state.confidence or 0.0,
        escalated=state.requires_escalation,
        ticket_id=ticket_id,
    )