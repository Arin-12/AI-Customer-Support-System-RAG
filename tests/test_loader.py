from ai.rag.loader import load_knowledge_base


def test_load_knowledge_base():
    documents = load_knowledge_base()

    assert len(documents) == 107

    # Make sure documents contain actual text
    assert all(doc.page_content.strip() for doc in documents)

    # Make sure LangChain metadata exists
    assert all("source" in doc.metadata for doc in documents)
