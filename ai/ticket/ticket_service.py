class TicketService:

    def create_ticket(self, user_id, subject, description):
        """
        Create a new support ticket.
        """
        raise NotImplementedError

    def get_ticket(self, ticket_id):
        """
        Retrieve a support ticket.
        """
        raise NotImplementedError

    def add_message(self, ticket_id, sender, message):
        """
        Add a message to an existing ticket.
        """
        raise NotImplementedError

    def get_messages(self, ticket_id):
        """
        Retrieve all messages for a ticket.
        """
        raise NotImplementedError

    def update_status(self, ticket_id, status):
        """
        Update ticket status.
        """
        raise NotImplementedError