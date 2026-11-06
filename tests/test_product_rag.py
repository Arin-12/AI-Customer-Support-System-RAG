from ai.rag.pipeline import retrieve_context


def test_product_rag_query():
    query = "Tell me about the iPhone 16 Pro"

    result = retrieve_context(query, k=5)

    assert result["documents"]
    assert result["context"]

    print("\n--- Query ---")
    print(query)

    print("\n--- Retrieved Product Context ---")
    print(result["context"])

    assert "iPhone 16 Pro" in result["context"]


def test_samsung_product_rag_query():
    query = "Tell me about Samsung Galaxy S25"

    result = retrieve_context(query, k=5)

    assert result["documents"]
    assert result["context"]

    print("\n--- Query ---")
    print(query)

    print("\n--- Retrieved Product Context ---")
    print(result["context"])

    assert "Galaxy S25" in result["context"]