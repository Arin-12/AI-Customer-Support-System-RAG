
import csv
import sys
from pathlib import Path

# Allow imports from the project root when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents


# Keep these queries aligned with tests/test_recall_evaluation.py.
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

K = 5
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def reciprocal_rank(retrieved_sources, relevant_sources):
    """Return 1/rank for the first relevant source, or 0 if not found."""
    if not relevant_sources:
        raise ValueError("relevant_sources must not be empty")

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_sources:
            return 1.0 / rank, rank

    return 0.0, None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    case_results = []

    for case in EVALUATION_CASES:
        documents = retrieve_documents(case["query"], k=K)

        # Preserve the retrieval order: rank 1 is the first result.
        retrieved_sources = [
            document.metadata.get("source")
            for document in documents
            if document.metadata.get("source")
        ]

        score, first_relevant_rank = reciprocal_rank(
            retrieved_sources,
            case["relevant_sources"],
        )

        case_results.append({
            "query": case["query"],
            "expected_sources": ";".join(sorted(case["relevant_sources"])),
            "retrieved_sources": ";".join(retrieved_sources),
            "first_relevant_rank": first_relevant_rank or "",
            "reciprocal_rank": score,
            "k": K,
        })

        rank_text = (
            str(first_relevant_rank)
            if first_relevant_rank is not None
            else "Not found in top 5"
        )
        print(f"\nQuery: {case['query']}")
        print(f"Expected source(s): {case['relevant_sources']}")
        print(f"Retrieved sources in rank order: {retrieved_sources}")
        print(f"First relevant rank: {rank_text}")
        print(f"Reciprocal rank: {score:.4f}")

    mrr = sum(row["reciprocal_rank"] for row in case_results) / len(case_results)
    found_count = sum(
        bool(row["first_relevant_rank"]) for row in case_results
    )

    case_path = RESULTS_DIR / "mrr_case_results.csv"
    with case_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=case_results[0].keys())
        writer.writeheader()
        writer.writerows(case_results)

    summary = {
        "metric": "MRR@5",
        "value": round(mrr, 4),
        "queries_evaluated": len(case_results),
        "queries_with_relevant_source_in_top_5": found_count,
        "k": K,
    }

    summary_path = RESULTS_DIR / "mrr_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print("\n========== MRR EVALUATION SUMMARY ==========")
    print(f"MRR@{K}: {mrr:.4f} ({mrr:.2%})")
    print(f"Queries evaluated: {len(case_results)}")
    print(f"Queries with a relevant source in top {K}: {found_count}")
    print(f"Per-query results saved to: {case_path}")
    print(f"Summary metrics saved to: {summary_path}")


if __name__ == "__main__":
    main()
