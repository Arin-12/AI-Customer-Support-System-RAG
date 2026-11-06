from ai.agent.router import Route
from ai.agent.state import AgentState
from ai.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from ai.llm.llm import MockLLM
from ai.agent.orchestrator import AgentOrchestrator
from ai.llm.llm import GroqLLM


def test_initial_state():
    state = AgentState(
        query="Where is order ORD1002?"
    )

    assert state.query == "Where is order ORD1002?"
    assert state.route is None
    assert state.rag_context == []
    assert state.sql_context == {}
    assert state.response is None
    assert state.requires_escalation is False


def test_state_with_route():
    state = AgentState(
        query="Where is order ORD1002?",
        route=Route.SQL
    )

    assert state.route == Route.SQL


def test_state_with_sql_result():
    state = AgentState(
        query="Where is order ORD1002?",
        route=Route.SQL,
        sql_context={
            "order_id": "ORD1002",
            "status": "Shipped"
        }
    )

    assert state.sql_context["order_id"] == "ORD1002"
    assert state.sql_context["status"] == "Shipped"


def test_system_prompt():
    assert "customer support assistant" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT


def test_build_user_prompt():
    prompt = build_user_prompt(
        query="Where is order ORD1002?",
        sql_context={
            "order_id": "ORD1002",
            "status": "Shipped"
        }
    )

    assert "ORD1002" in prompt
    assert "Shipped" in prompt
    assert "Where is order ORD1002?" in prompt


def test_mock_llm():
    llm = MockLLM()

    response = llm.generate(
        "Where is order ORD1002?"
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert response == "This is a test response from the mock LLM."



def fake_rag(query: str) -> list[str]:
    return [
        "Customers may return eligible products according to the return policy."
    ]


def fake_sql(query: str) -> dict:
    return {
        "order_id": "ORD1002",
        "status": "Shipped"
    }


def fake_ticket(query: str) -> str:
    return "TICKET-001"


def test_orchestrator_rag():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        rag_tool=fake_rag,
    )

    state = orchestrator.handle(
        "What is your return policy?"
    )

    assert state.route == Route.RAG
    assert len(state.rag_context) == 1
    assert state.response is not None


def test_orchestrator_sql():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        sql_tool=fake_sql,
    )

    state = orchestrator.handle(
        "Where is order ORD1002?"
    )

    assert state.route == Route.SQL
    assert state.sql_context["order_id"] == "ORD1002"
    assert state.sql_context["status"] == "Shipped"
    assert state.response is not None


def test_orchestrator_sql_and_rag():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        rag_tool=fake_rag,
        sql_tool=fake_sql,
    )

    state = orchestrator.handle(
        "My order ORD1002 has a damaged product. Can I get a refund?"
    )

    assert state.route == Route.SQL_AND_RAG
    assert len(state.rag_context) == 1
    assert state.sql_context["order_id"] == "ORD1002"
    assert state.response is not None


def test_orchestrator_ticket():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        ticket_tool=fake_ticket,
    )

    state = orchestrator.handle(
        "I want to talk to a human agent"
    )

    assert state.route == Route.TICKET
    assert state.ticket_id == "TICKET-001"
    assert state.requires_escalation is True
    assert state.response is None


def test_unknown_query_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM()
    )

    state = orchestrator.handle(
        "Tell me something completely unrelated."
    )

    assert state.route == Route.UNKNOWN
    assert state.requires_escalation is True
    assert state.response is None


def test_missing_rag_tool_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM()
    )

    state = orchestrator.handle(
        "What is your return policy?"
    )

    assert state.route == Route.RAG
    assert state.requires_escalation is True
    assert state.response is None


def test_missing_sql_tool_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM()
    )

    state = orchestrator.handle(
        "Where is order ORD1002?"
    )

    assert state.route == Route.SQL
    assert state.requires_escalation is True
    assert state.response is None


def test_missing_tool_for_sql_and_rag_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        sql_tool=fake_sql,
    )

    state = orchestrator.handle(
        "My order ORD1002 has a damaged product. Can I get a refund?"
    )

    assert state.route == Route.SQL_AND_RAG
    assert state.requires_escalation is True
    assert state.response is None


def test_ticket_without_ticket_tool_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM()
    )

    state = orchestrator.handle(
        "I want to talk to a human agent"
    )

    assert state.route == Route.TICKET
    assert state.requires_escalation is True
    assert state.ticket_id is None
    assert state.response is None

def failing_rag(query: str) -> list[str]:
    raise RuntimeError("RAG service unavailable")


def failing_sql(query: str) -> dict:
    raise RuntimeError("Database unavailable")


def test_rag_failure_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        rag_tool=failing_rag,
    )

    state = orchestrator.handle(
        "What is your return policy?"
    )

    assert state.route == Route.RAG
    assert state.requires_escalation is True
    assert state.response is None


def test_sql_failure_escalates():
    orchestrator = AgentOrchestrator(
        llm=MockLLM(),
        sql_tool=failing_sql,
    )

    state = orchestrator.handle(
        "Where is order ORD1002?"
    )

    assert state.route == Route.SQL
    assert state.requires_escalation is True
    assert state.response is None


def test_groq_llm():
    llm = GroqLLM()

    response = llm.generate(
        "Reply with exactly: Real Groq LLM test successful"
    )

    assert isinstance(response, str)
    assert len(response) > 0

def test_real_groq_orchestrator():
    orchestrator = AgentOrchestrator(
        llm=GroqLLM(),
        rag_tool=fake_rag,
    )

    state = orchestrator.handle(
        "What is your return policy?"
    )

    assert state.route == Route.RAG
    assert state.requires_escalation is False
    assert state.response is not None
    assert len(state.response) > 0

    print("\n--- Real Groq Response ---")
    print(state.response)

def test_real_rag_groq_orchestrator():
    from ai.rag.pipeline import retrieve_context

    def real_rag_tool(query: str) -> list[str]:
        result = retrieve_context(query, k=5)

        if not result["context"]:
            return []

        return [result["context"]]

    orchestrator = AgentOrchestrator(
        llm=GroqLLM(),
        rag_tool=real_rag_tool,
    )

    state = orchestrator.handle(
        "My product arrived damaged. What should I do?"
    )

    assert state.route == Route.RAG
    assert state.requires_escalation is False
    assert len(state.rag_context) > 0
    assert state.response is not None
    assert len(state.response) > 0

    print("\n--- Real RAG + Groq Response ---")
    print(state.response)