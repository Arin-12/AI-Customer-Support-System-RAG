
"""
Experimental product-aware retriever.

Keeps the existing FAISS + topic retriever unchanged.
Applies exact product-name matching only to product queries.
"""

import re

from ai.rag.retriever import retrieve_documents
from ai.rag.vector_store import load_vector_store


def _normalize(text: str) -> str:
    """Normalize text for product-name matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def retrieve_product_experiment(query: str, k: int = 5):
    """
    Retrieve documents using the existing retriever, then prioritize
    exact product-name matches when the query names a known product.

    Non-product queries use the existing retriever unchanged.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    query_normalized = _normalize(query)

    # Read product names from the actual indexed product documents.
    vector_store = load_vector_store()
    product_docs = [
        doc
        for doc in vector_store.docstore._dict.values()
        if doc.metadata.get("source") == "product_information.csv"
    ]

    matching_products = []
    for doc in product_docs:
        content = doc.page_content
        match = re.search(
            r"product_name:\s*([^\n]+)",
            content,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        product_name = _normalize(match.group(1))
        if product_name and product_name in query_normalized:
            matching_products.append((product_name, doc))

    # If no exact product name appears in the query, preserve current behavior.
    if not matching_products:
        return retrieve_documents(query, k=k)

    # Start with a wider candidate pool, then place the exact product first.
    candidates = retrieve_documents(query, k=max(k * 4, 20))

    exact_names = {name for name, _ in matching_products}

    def is_exact_product(document):
        content = _normalize(document.page_content)
        return any(
            re.search(
                rf"product_name:\s*{re.escape(name)}(?:\n|$)",
                content,
            )
            for name in exact_names
        )

    exact_results = [doc for doc in candidates if is_exact_product(doc)]
    other_results = [doc for doc in candidates if not is_exact_product(doc)]

    # If the exact product was not in the candidate pool, add its indexed record.
    for _, product_doc in matching_products:
        if not any(
            _normalize(doc.page_content) == _normalize(product_doc.page_content)
            for doc in exact_results
        ):
            exact_results.insert(0, product_doc)

    # Remove duplicate document contents while preserving order.
    ordered = []
    seen = set()
    for doc in exact_results + other_results:
        key = _normalize(doc.page_content)
        if key not in seen:
            seen.add(key)
            ordered.append(doc)

    return ordered[:k]


if __name__ == "__main__":
    test_queries = [
        "What are the specifications of the Apple MacBook Air M2?",
        "Compare MacBook Air M2 and MacBook Air M3",
        "What is the return policy for damaged electronics?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = retrieve_product_experiment(query, k=5)
        for rank, doc in enumerate(results, start=1):
            print(f"\n--- RESULT {rank} ---")
            print("Source:", doc.metadata.get("source"))
            print(doc.page_content[:300])
