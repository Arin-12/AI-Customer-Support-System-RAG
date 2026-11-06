from ai.agent.orchestrator import (
    AgentOrchestrator,
    rag_tool,
    sql_tool,
)
from ai.agent.router import Route
from ai.llm.llm import GroqLLM


def test_real_agent_member4_hybrid_integration():

    orchestrator = AgentOrchestrator(
        llm=GroqLLM(),
        rag_tool=rag_tool,
        sql_tool=sql_tool,
    )

    state = orchestrator.handle(
        "My order ORD1005 contains a damaged product. "
        "Can I get a refund?"
    )

    assert state.route == Route.SQL_AND_RAG

    assert state.sql_context["found"] is True
    assert state.sql_context["order"]["order_id"] == "ORD1005"

    assert len(state.sql_context["items"]) > 0

    assert len(state.rag_context) > 0

    assert state.response is not None
    assert len(state.response) > 0

    assert state.confidence == 0.95

    assert state.requires_escalation is False

    print("\n--- Route ---")
    print(state.route)

    print("\n--- SQL Context ---")
    print(state.sql_context)

    print("\n--- RAG Context ---")
    print(state.rag_context[0])

    print("\n--- Response ---")
    print(state.response)

    print("\n--- Confidence ---")
    print(state.confidence)

    print("\n--- Escalated ---")
    print(state.requires_escalation)