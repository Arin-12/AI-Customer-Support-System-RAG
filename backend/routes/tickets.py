from fastapi import APIRouter, HTTPException

from backend.schemas.ticket import TicketCreate, TicketResponse
from backend.services.ticket_service import TicketService


router = APIRouter()

ticket_service = TicketService()


@router.post("/tickets", response_model=TicketResponse)
def create_ticket(request: TicketCreate):

    try:
        result = ticket_service.create_ticket(
            user_id=request.user_id,
            subject=request.subject,
            description=request.description,
        )

        return TicketResponse(
            ticket_id=result["ticket_id"],
            user_id=result["user_id"],
            subject=result["subject"],
            status=result["status"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )