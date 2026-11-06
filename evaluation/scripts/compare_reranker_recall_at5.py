
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.reranker_experiment import retrieve_with_reranking


CASES = [
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

K = 5


def evaluate(label, retrieval_function):
    recall_scores = []
    precision_scores = []

    print(f"\n--- {label} ---")

    for case in CASES:
        documents = retrieval_function(case["query"], k=K)

        retrieved_sources = {
            Path(str(doc.metadata.get("source", ""))).name
            for doc in documents
            if doc.metadata.get("source")
        }

        relevant_sources = case["relevant_sources"]
        relevant_found = retrieved_sources & relevant_sources

        precision = (
            len(relevant_found) / len(retrieved_sources)
            if retrieved_sources else 0.0
        )
        recall = len(relevant_found) / len(relevant_sources)

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(
            f"Recall@5: {recall:.2%} | "
            f"Precision@5: {precision:.2%} | "
            f"{case['query']}"
        )

    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)

    print(f"Mean source-level Precision@5: {mean_precision:.2%}")
    print(f"Mean source-level Recall@5: {mean_recall:.2%}")

    return mean_precision, mean_recall


if __name__ == "__main__":
    baseline = evaluate("EXISTING RETRIEVER", retrieve_documents)
    reranked = evaluate("CROSS-ENCODER RERANKER", retrieve_with_reranking)

    print("\n--- COMPARISON ---")
    print(f"Precision@5 change: {(reranked[0] - baseline[0]) * 100:+.2f} percentage points")
    print(f"Recall@5 change: {(reranked[1] - baseline[1]) * 100:+.2f} percentage points")
