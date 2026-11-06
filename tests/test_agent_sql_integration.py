from ai.agent.orchestrator import AgentOrchestrator, sql_tool
from ai.agent.router import Route
from ai.llm.llm import GroqLLM


def test_real_sql_groq_orchestrator():

    orchestrator = AgentOrchestrator(
        llm=GroqLLM(),
        sql_tool=sql_tool,
    )

    state = orchestrator.handle(
        "Where is my order ORD1005?"
    )

    assert state.route == Route.SQL
    assert state.requires_escalation is False

    assert state.sql_context["found"] is True
    assert state.sql_context["order"]["order_id"] == "ORD1005"

    assert state.response is not None
    assert len(state.response) > 0

    print("\n--- SQL Context ---")
    print(state.sql_context)

    print("\n--- Real SQL + Groq Response ---")
    print(state.response)