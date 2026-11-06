
"""
Experimental cross-encoder reranker.

Keeps the production retriever unchanged.
Retrieves a wider candidate pool, then reranks it
using query-document relevance scores.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from ai.rag.retriever import retrieve_documents


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def load_reranker():
    """Load the reranker once and reuse it."""
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def rerank_documents(query: str, documents: list, k: int = 5) -> list:
    """
    Rerank candidates returned by the existing retriever.

    Args:
        query: User's question.
        documents: Candidate documents to rerank.
        k: Number of final documents to return.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not documents:
        return []

    model = load_reranker()

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    # Stable sorting preserves candidate order when scores tie.
    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [document for document, _ in ranked[:k]]


def retrieve_with_reranking(query: str, k: int = 5) -> list:
    """
    Experimental retrieval pipeline:
    existing retrieval -> cross-encoder reranking.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    candidates = retrieve_documents(query, k=max(k * 4, 20))
    return rerank_documents(query, candidates, k=k)


if __name__ == "__main__":
    queries = [
        "A device arrived with a cracked screen. What should I do?",
        "Can I send back an item if I changed my mind?",
        "The model I bought is unsuitable. Can I swap it for another one?",
        "I placed an order by mistake. Is there a way to stop it?",
        "My package has not arrived yet. Where can I check its status?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = retrieve_with_reranking(query, k=5)

        for rank, document in enumerate(results, start=1):
            print(f"\n--- RESULT {rank} ---")
            print("Source:", document.metadata.get("source"))
            print(document.page_content[:350])
