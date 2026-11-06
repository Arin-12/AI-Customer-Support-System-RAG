from ai.verification.response_verifier import verify_response
from ai.verification.confidence import calculate_confidence

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.schemas.ticket import TicketCreate, TicketResponse

from fastapi.testclient import TestClient
from main import app
from backend.routes import chat as chat_route
from ai.agent.router import Route
from ai.agent.state import AgentState


def test_verified_response():
    result = verify_response(
        answer="You can return the product within 7 days.",
        context="Products can be returned within 7 days."
    )

    assert result.verified is True
    assert result.confidence > 0


def test_empty_response():
    result = verify_response(
        answer="",
        context="Some policy information."
    )

    assert result.verified is False


def test_sql_and_rag_confidence():
    confidence = calculate_confidence(
        context_available=True,
        sql_verified=True,
        rag_supported=True
    )

    assert confidence == 0.95


def test_sql_confidence():
    confidence = calculate_confidence(
        context_available=True,
        sql_verified=True,
        rag_supported=False
    )

    assert confidence == 0.90


def test_rag_confidence():
    confidence = calculate_confidence(
        context_available=True,
        sql_verified=False,
        rag_supported=True
    )

    assert confidence == 0.85


def test_no_context_confidence():
    confidence = calculate_confidence(
        context_available=False,
        sql_verified=False,
        rag_supported=False
    )

    assert confidence == 0.20


def test_chat_request_schema():
    request = ChatRequest(
        user_id=1,
        message="What is your return policy?"
    )

    assert request.user_id == 1
    assert request.message == "What is your return policy?"


def test_chat_response_schema():
    response = ChatResponse(
        answer="You can return the product within the allowed period.",
        confidence=0.90,
        escalated=False
    )

    assert response.confidence == 0.90
    assert response.escalated is False
    assert response.ticket_id is None


def test_ticket_create_schema():
    ticket = TicketCreate(
        user_id="USR001",
        subject="Damaged product",
        description="The product arrived damaged."
    )

    assert ticket.user_id == "USR001"
    assert ticket.subject == "Damaged product"


def test_ticket_response_schema():
    ticket = TicketResponse(
        ticket_id="TKT001",
        user_id="USR001",
        subject="Damaged product",
        status="open"
    )

    assert ticket.ticket_id == "TKT001"
    assert ticket.user_id == "USR001"
    assert ticket.status == "open"


    
def test_chat_api_does_not_return_unverified_answer(monkeypatch):
    class FakeOrchestrator:
        def handle(self, query, user_id=None, conversation_id=None):
            return AgentState(
                query=query,
                user_id=user_id,
                route=Route.RAG,
                response=None,
                requires_escalation=True,
                confidence=0.20,
            )

    monkeypatch.setattr(
        chat_route,
        "orchestrator",
        FakeOrchestrator(),
    )

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "user_id": 1,
            "message": "What is your return policy?",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["escalated"] is True
    assert "14-day return policy" not in body["answer"]
    assert "couldn't retrieve enough verified information" in (
        body["answer"].lower()
    )



def test_chat_api_does_not_disclose_another_customers_order(
    monkeypatch,
):
    from ai.agent.orchestrator import AgentOrchestrator
    from backend.routes import chat as chat_route

    class FakeLLM:
        def generate(self, prompt):
            return "Here are the order details."

    def fake_sql_tool(query, user_id=None):
        # Customer USR001 is attempting to access an
        # order belonging to USR005.
        assert user_id == "USR001"
        return {
            "found": False,
            "access_denied": True,
        }

    protected_orchestrator = AgentOrchestrator(
        llm=FakeLLM(),
        sql_tool=fake_sql_tool,
    )

    monkeypatch.setattr(
        chat_route,
        "orchestrator",
        protected_orchestrator,
    )

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "user_id": 1,
            "message": "Where is order ORD1005?",
        },
    )

    assert response.status_code == 200

    body = response.json()
    answer = body["answer"].lower()

    assert "private laptop" not in answer
    assert "here are the order details" not in answer
    assert body["escalated"] is True



def test_live_chat_api_denies_another_customers_order():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # ORD1005 belongs to USR005; API user_id 1 maps to USR001.
    response = client.post(
        "/chat",
        json={
            "user_id": 1,
            "message": "Where is my order ORD1005?",
        },
    )

    assert response.status_code == 200

    body = response.json()
    answer = body["answer"].lower()

    # The response must not reveal private order details.
    assert "ord1005" not in answer
    assert "private laptop" not in answer
    assert "139,980" not in answer
    assert "access" in answer or "verify" in answer or "couldn't" in answer



def test_live_chat_api_allows_owners_order_lookup():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "user_id": 5,
            "message": "Where is my order ORD1005?",
        },
    )

    assert response.status_code == 200

    body = response.json()
    answer = body["answer"].lower()

    assert "ord1005" in answer
    assert "delivered" in answer
    assert body["escalated"] is False
