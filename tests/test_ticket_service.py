from ai.sql.database import SessionLocal
from sqlalchemy import text

from backend.services.ticket_service import TicketService


def test_ticket_service_create_get_message_update():
    service = TicketService()

    result = service.create_ticket(
        user_id="USR001",
        subject="Test damaged product ticket",
        description="Testing the ticket service.",
    )

    assert result["ticket_id"].startswith("TKT")
    assert result["user_id"] == "USR001"
    assert result["status"] == "open"

    ticket_id = result["ticket_id"]

    ticket = service.get_ticket(ticket_id)

    assert ticket is not None
    assert ticket["ticket_id"] == ticket_id
    assert ticket["user_id"] == "USR001"

    message = service.add_message(
        ticket_id=ticket_id,
        sender="agent",
        message="This is a test agent response.",
    )

    assert message["ticket_id"] == ticket_id
    assert message["sender_type"] == "agent"

    messages = service.get_messages(ticket_id)

    assert len(messages) >= 2
    assert any(
        item["message"] == "This is a test agent response."
        for item in messages
    )

    updated = service.update_status(
        ticket_id=ticket_id,
        status="closed",
    )

    assert updated["ticket_id"] == ticket_id
    assert updated["status"] == "closed"

    final_ticket = service.get_ticket(ticket_id)

    assert final_ticket["status"] == "closed"

    # Verify directly in PostgreSQL too.
    with SessionLocal() as session:
        db_ticket = session.execute(
            text("""
                SELECT ticket_id, user_id, status
                FROM support_tickets
                WHERE ticket_id = :ticket_id
            """),
            {"ticket_id": ticket_id},
        ).fetchone()

    assert db_ticket is not None
    assert db_ticket[0] == ticket_id
    assert db_ticket[1] == "USR001"
    assert db_ticket[2] == "closed"