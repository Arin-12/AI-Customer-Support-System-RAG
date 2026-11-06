from ai.rag.retriever import retrieve_documents


# Initial manually labelled evaluation set.
# Each query has one expected relevant source based on its topic.
EVALUATION_CASES = [
    {
        "query": "What is the return policy for damaged electronics?",
        "relevant_sources": {"damaged_product_policy.txt"},
    },
    {
        "query": "What is the general return policy?",
        "relevant_sources": {"return_policy.txt"},
    },
    {
        "query": "How do refunds work after a return?",
        "relevant_sources": {"refund_policy.txt"},
    },
    {
        "query": "Can I exchange a product I purchased?",
        "relevant_sources": {"exchange_policy.txt"},
    },
    {
        "query": "How can I cancel my order?",
        "relevant_sources": {"cancellation_policy.txt"},
    },
    {
        "query": "How can I track my shipment and delivery?",
        "relevant_sources": {"shipping_policy.txt"},
    },
    {
        "query": "What payment methods and payment rules are supported?",
        "relevant_sources": {"payment_policy.txt"},
    },
    {
        "query": "What are the specifications of the Apple MacBook Air M2?",
        "relevant_sources": {"product_information.csv"},
    },
]


def recall_at_k(retrieved_sources, relevant_sources):
    """Fraction of labelled relevant sources retrieved in the top K."""
    if not relevant_sources:
        raise ValueError("relevant_sources must not be empty")

    found = set(retrieved_sources) & set(relevant_sources)
    return len(found) / len(relevant_sources)


def test_recall_at_5_evaluation():
    scores = []

    for case in EVALUATION_CASES:
        documents = retrieve_documents(case["query"], k=5)

        retrieved_sources = {
            document.metadata["source"]
            for document in documents
            if document.metadata.get("source")
        }

        score = recall_at_k(
            retrieved_sources,
            case["relevant_sources"],
        )
        scores.append(score)

        print(f"\nQuery: {case['query']}")
        print(f"Expected source(s): {case['relevant_sources']}")
        print(f"Retrieved source(s): {retrieved_sources}")
        print(f"Recall@5: {score:.2f}")

    mean_recall = sum(scores) / len(scores)
    print(f"\nMean Recall@5: {mean_recall:.2%}")
    print(f"Queries evaluated: {len(scores)}")
    print(f"Queries retrieving expected source: {sum(s == 1.0 for s in scores)}")

    assert len(scores) == len(EVALUATION_CASES)
