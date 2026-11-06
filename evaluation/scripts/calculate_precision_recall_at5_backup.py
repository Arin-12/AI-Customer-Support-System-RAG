
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents

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


def main():
    precision_scores = []
    recall_scores = []

    for case in CASES:
        documents = retrieve_documents(case["query"], k=K)

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

        print(f"\nQuery: {case['query']}")
        print(f"Unique sources retrieved: {len(retrieved_sources)}")
        print(f"Relevant sources found: {len(relevant_found)}")
        print(f"Precision@5 (source-level): {precision:.2%}")
        print(f"Recall@5 (source-level): {recall:.2%}")

    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)

    print("\n--- Mean results across 8 queries ---")
    print(f"Mean Precision@5 (source-level): {mean_precision:.2%}")
    print(f"Mean Recall@5 (source-level): {mean_recall:.2%}")
    print(f"Queries evaluated: {len(CASES)}")


if __name__ == "__main__":
    main()
