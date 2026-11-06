from ai.rag.embeddings import create_embeddings


def test_create_embeddings():
    embeddings = create_embeddings()

    vector = embeddings.embed_query(
        "What is the return policy?"
    )

    assert len(vector) == 768
    assert all(isinstance(value, float) for value in vector)