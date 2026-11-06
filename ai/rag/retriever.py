
from ai.rag.vector_store import load_vector_store


# Topic-specific ranking hints.
# These are used to rerank FAISS candidates, not to replace retrieval.
TOPIC_RULES = {
    "damaged": {
        "query_terms": (
            "damaged", "damage", "broken", "defective", "arrived damaged",
        ),
        "source": "damaged_product_policy.txt",
        "content_terms": (
            "damaged", "damage", "defective", "broken", "replacement",
        ),
    },
    "return": {
        "query_terms": ("return", "returns", "returning"),
        "source": "return_policy.txt",
        "content_terms": ("return", "returned", "eligible", "return window"),
    },
    "refund": {
        "query_terms": ("refund", "refunds", "refunded", "money back"),
        "source": "refund_policy.txt",
        "content_terms": ("refund", "refunded", "reimbursement", "payment"),
    },
    "exchange": {
        "query_terms": ("exchange", "exchanges", "exchanging"),
        "source": "exchange_policy.txt",
        "content_terms": ("exchange", "exchanged", "replacement"),
    },
    "cancel": {
        "query_terms": ("cancel", "cancellation", "cancelled", "canceled"),
        "source": "cancellation_policy.txt",
        "content_terms": ("cancel", "cancellation", "eligible processing stage"),
    },
    "shipping": {
        "query_terms": (
            "shipping", "shipment", "delivery", "deliver", "tracking", "track",
        ),
        "source": "shipping_policy.txt",
        "content_terms": ("shipping", "shipment", "delivery", "tracking", "shipped"),
    },
    "payment": {
        "query_terms": (
            "payment", "payments", "pay", "paid", "credit card", "debit card",
        ),
        "source": "payment_policy.txt",
        "content_terms": ("payment", "payments", "credit card", "debit card"),
    },
    "warranty": {
        "query_terms": ("warranty", "warranties", "guarantee"),
        "source": None,
        "content_terms": ("warranty", "warranties", "guarantee", "warranty terms"),
    },
}


def _matching_topic_rules(query: str):
    """Find applicable topic rules using query wording."""
    query_lower = query.lower()
    matched = []

    for topic, rule in TOPIC_RULES.items():
        if any(term in query_lower for term in rule["query_terms"]):
            matched.append((topic, rule))

    # Avoid applying the generic return rule to a more specific
    # damaged-product or refund question.
    specific_topics = {"damaged", "refund", "exchange"}
    if any(topic in specific_topics for topic, _ in matched):
        matched = [
            (topic, rule)
            for topic, rule in matched
            if topic != "return"
        ]

    return matched


def retrieve_documents(query: str, k: int = 5):
    """
    Retrieve FAISS candidates and rerank them using topic relevance.

    Original FAISS order breaks ties, so documents with equal topic
    scores retain their relative retrieval order.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    vector_store = load_vector_store()
    rules = _matching_topic_rules(query)

    # Preserve the original FAISS behaviour when no topic rule matches.
    if not rules:
        return vector_store.similarity_search(query, k=k)

    candidates = vector_store.similarity_search(
        query,
        k=max(k * 4, 20),
    )

    def score_document(document):
        content = document.page_content.lower()
        source = str(document.metadata.get("source", "")).lower()
        score = 0

        for topic, rule in rules:
            # Give the dedicated policy source a moderate boost.
            # Content evidence is also considered.
            expected_source = rule["source"]
            if expected_source and expected_source in source:
                score += 4

            for term in rule["content_terms"]:
                if term in content:
                    score += 1

        return score

    # Python's sort is stable: original FAISS ordering is retained
    # when two candidates receive the same score.
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

        for rank, document in enumerate(documents, start=1):
            print(f"\n--- RESULT {rank} ---")
            print("Source:", document.metadata.get("source"))
            print("Content:")
            print(document.page_content[:500])
