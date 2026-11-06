from ai.rag.pipeline import retrieve_context


def test_rag_handoff():

    customer_query = "My product arrived damaged. What should I do?"

    result = retrieve_context(
        customer_query,
        k=5
    )

    context = result["context"]

    print("\n===== RAG CONTEXT =====")
    print(context)

    assert context
    assert len(context) > 0