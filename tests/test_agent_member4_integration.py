from ai.agent.orchestrator import (
    AgentOrchestrator,
    rag_tool,
    sql_tool,
)
from ai.agent.router import Route
from ai.llm.llm import GroqLLM


def test_real_agent_member4_integration():

    orchestrator = AgentOrchestrator(
        llm=GroqLLM(),
        rag_tool=rag_tool,
        sql_tool=sql_tool,
    )

    state = orchestrator.handle(
        "Where is my order ORD1005?"
    )

    assert state.route == Route.SQL

    assert state.sql_context["found"] is True

    assert state.sql_context["order"]["order_id"] == "ORD1005"

    assert state.response is not None
    assert len(state.response) > 0

    assert state.confidence == 0.90

    assert state.requires_escalation is False

    print("\n--- Route ---")
    print(state.route)

    print("\n--- SQL Context ---")
    print(state.sql_context)

    print("\n--- Response ---")
    print(state.response)

    print("\n--- Confidence ---")
    print(state.confidence)

    print("\n--- Escalated ---")
    print(state.requires_escalation)