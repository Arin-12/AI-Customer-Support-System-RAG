from ai.rag.pipeline import retrieve_context


def test_retrieve_context():

    result = retrieve_context(
        "My product arrived damaged. What should I do?",
        k=5,
    )

    assert result["query"]
    assert len(result["documents"]) == 5
    assert result["context"].strip()

    assert "damaged_product_policy.txt" in result["context"]


    