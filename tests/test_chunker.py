from ai.rag.loader import load_knowledge_base
from ai.rag.chunker import chunk_documents


def test_chunk_documents():
    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    assert len(documents) == 107
    assert len(chunks) > len(documents)

    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)