
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents

K = 5

# Manual chunk-level relevance judgments:
# 1.0 = directly relevant
# 0.5 = partially relevant
# 0.0 = irrelevant
#
# These judgments correspond to the top-five results we inspected.
# Each query is checked against the current retrieval output.

CASES = [
    {
        "query": "What is the return policy for damaged electronics?",
        "scores": [1.0, 0.5, 1.0, 0.5, 0.5],
    },
    {
        "query": "What is the general return policy?",
        "scores": [1.0, 0.0, 0.5, 1.0, 1.0],
    },
    {
        "query": "How do refunds work after a return?",
        "scores": [1.0, 1.0, 0.5, 1.0, 0.5],
    },
    {
        "query": "Can I exchange a product I purchased?",
        "scores": [1.0, 1.0, 0.5, 0.5, 0.0],
    },
    {
        "query": "How can I cancel my order?",
        "scores": [1.0, 1.0, 1.0, 1.0, 0.0],
    },
    {
        "query": "How can I track my shipment and delivery?",
        "scores": [1.0, 1.0, 0.5, 0.0, 1.0],
    },
    {
        "query": "What payment methods and payment rules are supported?",
        "scores": [1.0, 1.0, 1.0, 1.0, 0.0],
    },
    {
        "query": "What are the specifications of the Apple MacBook Air M2?",
        "scores": [1.0, 0.5, 0.0, 0.5, 0.0],
    },
]


def main():
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for case in CASES:
        query = case["query"]
        judgments = case["scores"]
        documents = retrieve_documents(query, k=K)

        if len(documents) != len(judgments):
            raise ValueError(
                f"Expected {len(judgments)} results for query: {query}; "
                f"received {len(documents)}. Review the judgments."
            )

        # Binary relevance: only fully relevant results count as relevant.
        relevant_count = sum(score == 1.0 for score in judgments)
        precision = relevant_count / K

        # Graded relevance: partial results contribute 0.5.
        graded_precision = sum(judgments) / K

        # Recall needs the total number of relevant items for each query,
        # including relevant items that were not retrieved.
        # These judgments cover only the top five, so true Recall@5
        # cannot be calculated reliably from these scores alone.

        first_relevant_rank = next(
            (
                rank
                for rank, score in enumerate(judgments, start=1)
                if score == 1.0
            ),
            None,
        )
        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        precision_scores.append(precision)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuery: {query}")
        for rank, (document, score) in enumerate(
            zip(documents, judgments), start=1
        ):
            label = (
                "DIRECT"
                if score == 1.0
                else "PARTIAL"
                if score == 0.5
                else "IRRELEVANT"
            )
            source = document.metadata.get("source", "unknown")
            preview = " ".join(document.page_content.split())[:160]
            print(f"  {rank}. [{label}] {source}")
            print(f"     {preview}")

        print(f"Binary Precision@5: {precision:.2%}")
        print(f"Graded Precision@5: {graded_precision:.2%}")
        print(f"First directly relevant result: {first_relevant_rank}")
        print(f"Reciprocal rank: {reciprocal_rank:.4f}")

    print("\n--- Mean results ---")
    print(f"Queries evaluated: {len(CASES)}")
    print(
        "Mean Binary Precision@5: "
        f"{sum(precision_scores) / len(precision_scores):.2%}"
    )
    print(
        "Mean Graded Precision@5: "
        f"{sum(sum(case['scores']) / K for case in CASES) / len(CASES):.2%}"
    )
    print(
        "MRR@5 (manual judgments): "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.2%}"
    )
    print(
        "\nNote: These are preliminary manual judgments. "
        "Recall@5 requires a complete list of relevant items per query."
    )


if __name__ == "__main__":
    main()
