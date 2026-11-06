
from ai.rag.vector_store import load_vector_store


WARRANTY_KEYWORDS = (
    "warranty",
    "warranties",
    "guarantee",
)

CANCELLATION_KEYWORDS = (
    "cancel",
    "cancellation",
    "cancelled",
    "canceled",
    "cancelling",
)


def retrieve_documents(query: str, k: int = 5):
    """
    Retrieve relevant documents from FAISS.

    Warranty queries prioritize documents containing warranty information.
    Cancellation queries prioritize cancellation policy information.
    Other queries use the original similarity search.
    """

    vector_store = load_vector_store()
    query_lower = query.lower()

    is_warranty_query = any(
        keyword in query_lower
        for keyword in WARRANTY_KEYWORDS
    )

    is_cancellation_query = any(
        keyword in query_lower
        for keyword in CANCELLATION_KEYWORDS
    )

    if not is_warranty_query and not is_cancellation_query:
        return vector_store.similarity_search(query, k=k)

    candidates = vector_store.similarity_search(
        query,
        k=max(k * 4, 20),
    )

    if is_warranty_query:

        def score_document(document):
            content = document.page_content.lower()
            score = 0

            if "warranty" in content or "warranties" in content:
                score += 3

            if "guarantee" in content:
                score += 2

            if "warranty terms" in content:
                score += 2

            if "warranty handling" in content:
                score += 2

            if "warranty support" in content:
                score += 2

            return score

    else:

        def score_document(document):
            content = document.page_content.lower()
            source = str(
                document.metadata.get("source", "")
            ).lower()
            score = 0

            # Prefer the dedicated cancellation policy.
            if "cancellation_policy.txt" in source:
                score += 10

            # Also consider cancellation-related FAQ content.
            if "cancellation" in source:
                score += 4

            if "cancellation" in content:
                score += 3

            if "cancel" in content:
                score += 3

            if "eligible processing stage" in content:
                score += 3

            if "shipped" in content:
                score += 1

            return score

    ranked_documents = sorted(
        candidates,
        key=score_document,
        reverse=True,
    )

    return ranked_documents[:k]


if __name__ == "__main__":

    queries = [
        "What is the return policy?",
        "My product arrived damaged. What should I do?",
        "How can I cancel my order?",
        "How do refunds work?",
        "What payment methods are supported?",
        "What is the warranty policy for electronics?",
        "Can I get warranty support?",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        documents = retrieve_documents(query, k=3)

        for i, document in enumerate(documents, start=1):
            print(f"\n--- RESULT {i} ---")
            print("Source:", document.metadata.get("source"))
            print("Content:")
            print(document.page_content[:500])
