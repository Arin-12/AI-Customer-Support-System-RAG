from ai.agent.orchestrator import (
    AgentOrchestrator,
    rag_tool,
    sql_tool,
)
from ai.agent.router import Route
from ai.llm.llm import GroqLLM


def test_real_sql_rag_groq_orchestrator():

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
    assert state.requires_escalation is False

    # Verify SQL information
    assert state.sql_context["found"] is True
    assert state.sql_context["order"]["order_id"] == "ORD1005"

    # Verify order product
    assert len(state.sql_context["items"]) > 0

    # Verify RAG information
    assert len(state.rag_context) > 0

    # Verify final LLM response
    assert state.response is not None
    assert len(state.response) > 0

    print("\n--- SQL Context ---")
    print(state.sql_context)

    print("\n--- RAG Context ---")
    print(state.rag_context[0])

    print("\n--- Hybrid SQL + RAG + Groq Response ---")
    print(state.response)