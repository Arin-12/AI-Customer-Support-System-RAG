from pydantic import BaseModel


class TicketCreate(BaseModel):
    user_id: str
    subject: str
    description: str


class TicketResponse(BaseModel):
    ticket_id: str
    user_id: str
    subject: str
    status: str