from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split LangChain Documents into smaller chunks.

    Metadata from the original documents is preserved.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    from ai.rag.loader import load_knowledge_base

    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print("\n--- Chunk", i + 1, "---")
        print("Source:", chunk.metadata["source"])
        print("Length:", len(chunk.page_content))
        print(chunk.page_content[:300])