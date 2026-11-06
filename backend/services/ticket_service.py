from datetime import datetime

from sqlalchemy import text

from ai.sql.database import SessionLocal


class TicketService:

    def _generate_ticket_id(self, session):
        result = session.execute(
            text("""
                SELECT ticket_id
                FROM support_tickets
                WHERE ticket_id LIKE 'TKT%'
                ORDER BY ticket_id DESC
                LIMIT 1
            """)
        ).fetchone()

        if result is None:
            return "TKT001"

        number = int(result[0].replace("TKT", ""))
        return f"TKT{number + 1:03d}"

    def _generate_message_id(self, session):
        result = session.execute(
            text("""
                SELECT message_id
                FROM ticket_messages
                WHERE message_id LIKE 'MSG%'
                ORDER BY message_id DESC
                LIMIT 1
            """)
        ).fetchone()

        if result is None:
            return "MSG001"

        number = int(result[0].replace("MSG", ""))
        return f"MSG{number + 1:03d}"

    def create_ticket(self, user_id, subject, description):

        with SessionLocal() as session:

            user = session.execute(
                text("""
                    SELECT user_id
                    FROM users
                    WHERE user_id = :user_id
                """),
                {"user_id": user_id},
            ).fetchone()

            if user is None:
                raise ValueError(
                    f"User '{user_id}' does not exist."
                )

            ticket_id = self._generate_ticket_id(session)
            message_id = self._generate_message_id(session)
            now = datetime.now()

            session.execute(
                text("""
                    INSERT INTO support_tickets (
                        ticket_id,
                        user_id,
                        category,
                        priority,
                        status,
                        customer_message,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :ticket_id,
                        :user_id,
                        :category,
                        :priority,
                        :status,
                        :customer_message,
                        :created_at,
                        :updated_at
                    )
                """),
                {
                    "ticket_id": ticket_id,
                    "user_id": user_id,
                    "category": subject,
                    "priority": "medium",
                    "status": "open",
                    "customer_message": description,
                    "created_at": now,
                    "updated_at": now,
                },
            )

            session.execute(
                text("""
                    INSERT INTO ticket_messages (
                        message_id,
                        ticket_id,
                        sender_type,
                        message,
                        timestamp
                    )
                    VALUES (
                        :message_id,
                        :ticket_id,
                        :sender_type,
                        :message,
                        :timestamp
                    )
                """),
                {
                    "message_id": message_id,
                    "ticket_id": ticket_id,
                    "sender_type": "customer",
                    "message": description,
                    "timestamp": now,
                },
            )

            session.commit()

            return {
                "ticket_id": ticket_id,
                "user_id": user_id,
                "subject": subject,
                "status": "open",
            }

    def get_ticket(self, ticket_id):

        with SessionLocal() as session:
            result = session.execute(
                text("""
                    SELECT
                        ticket_id,
                        user_id,
                        category,
                        status,
                        customer_message,
                        ai_response,
                        created_at,
                        updated_at
                    FROM support_tickets
                    WHERE ticket_id = :ticket_id
                """),
                {"ticket_id": ticket_id},
            ).mappings().fetchone()

            return dict(result) if result else None

    def add_message(self, ticket_id, sender, message):

        with SessionLocal() as session:

            ticket = session.execute(
                text("""
                    SELECT ticket_id
                    FROM support_tickets
                    WHERE ticket_id = :ticket_id
                """),
                {"ticket_id": ticket_id},
            ).fetchone()

            if ticket is None:
                raise ValueError(
                    f"Ticket '{ticket_id}' does not exist."
                )

            message_id = self._generate_message_id(session)
            now = datetime.now()

            session.execute(
                text("""
                    INSERT INTO ticket_messages (
                        message_id,
                        ticket_id,
                        sender_type,
                        message,
                        timestamp
                    )
                    VALUES (
                        :message_id,
                        :ticket_id,
                        :sender_type,
                        :message,
                        :timestamp
                    )
                """),
                {
                    "message_id": message_id,
                    "ticket_id": ticket_id,
                    "sender_type": sender,
                    "message": message,
                    "timestamp": now,
                },
            )

            session.execute(
                text("""
                    UPDATE support_tickets
                    SET updated_at = :updated_at
                    WHERE ticket_id = :ticket_id
                """),
                {
                    "updated_at": now,
                    "ticket_id": ticket_id,
                },
            )

            session.commit()

            return {
                "message_id": message_id,
                "ticket_id": ticket_id,
                "sender_type": sender,
                "message": message,
            }

    def get_messages(self, ticket_id):

        with SessionLocal() as session:
            results = session.execute(
                text("""
                    SELECT
                        message_id,
                        ticket_id,
                        sender_type,
                        message,
                        timestamp
                    FROM ticket_messages
                    WHERE ticket_id = :ticket_id
                    ORDER BY timestamp ASC
                """),
                {"ticket_id": ticket_id},
            ).mappings().all()

            return [dict(row) for row in results]

    def update_status(self, ticket_id, status):

        with SessionLocal() as session:

            result = session.execute(
                text("""
                    UPDATE support_tickets
                    SET
                        status = :status,
                        updated_at = :updated_at
                    WHERE ticket_id = :ticket_id
                """),
                {
                    "ticket_id": ticket_id,
                    "status": status,
                    "updated_at": datetime.now(),
                },
            )

            if result.rowcount == 0:
                raise ValueError(
                    f"Ticket '{ticket_id}' does not exist."
                )

            session.commit()

            return {
                "ticket_id": ticket_id,
                "status": status,
            }