from ai.rag.retriever import retrieve_documents


def test_retrieve_documents():

    query = "What is the return policy for damaged electronics?"

    documents = retrieve_documents(query, k=5)

    assert len(documents) == 5

    assert all(document.page_content.strip() for document in documents)

    assert all("source" in document.metadata for document in documents)