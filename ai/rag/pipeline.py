from ai.rag.retriever import retrieve_documents


def build_context(documents):
    """
    Combine retrieved documents into a single context string.
    """

    context_parts = []

    for i, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")

        context_parts.append(
            f"[Source {i}: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def retrieve_context(query: str, k: int = 5):
    """
    Retrieve relevant documents and build the context
    that will later be sent to the LLM.
    """

    documents = retrieve_documents(query, k=k)

    context = build_context(documents)

    return {
        "query": query,
        "documents": documents,
        "context": context,
    }


if __name__ == "__main__":

    query = "My product arrived damaged. What should I do?"

    result = retrieve_context(query, k=5)

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(result["query"])

    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXT")
    print("=" * 80)
    print(result["context"])