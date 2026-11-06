
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.reranker_experiment import retrieve_with_reranking


EVALUATION_CASES = [
    {
        "query": "A device arrived with a cracked screen. What should I do?",
        "relevant_sources": {"damaged_product_policy.txt"},
    },
    {
        "query": "Can I send back an item if I changed my mind?",
        "relevant_sources": {"return_policy.txt"},
    },
    {
        "query": "When should I expect my money back once a return is approved?",
        "relevant_sources": {"refund_policy.txt"},
    },
    {
        "query": "The model I bought is unsuitable. Can I swap it for another one?",
        "relevant_sources": {"exchange_policy.txt"},
    },
    {
        "query": "I placed an order by mistake. Is there a way to stop it?",
        "relevant_sources": {"cancellation_policy.txt"},
    },
    {
        "query": "My package has not arrived yet. Where can I check its status?",
        "relevant_sources": {"shipping_policy.txt"},
    },
    {
        "query": "My card payment failed. Which payment options can I use?",
        "relevant_sources": {"payment_policy.txt"},
    },
    {
        "query": "What does the warranty cover if my gadget stops working?",
        "relevant_sources": {"warranty_policy.txt"},
    },
    {
        "query": "What features and specifications does the MacBook Air M2 have?",
        "relevant_sources": {"product_information.csv"},
    },
    {
        "query": "Is there a deadline for returning an opened electronics item?",
        "relevant_sources": {"return_policy.txt"},
    },
]

K = 20


def evaluate_retriever(retrieval_function, label):
    reciprocal_ranks = []
    found_count = 0

    print(f"\n--- {label} ---")

    for index, case in enumerate(EVALUATION_CASES, start=1):
        documents = retrieval_function(case["query"], k=K)

        first_relevant_rank = next(
            (
                rank
                for rank, document in enumerate(documents, start=1)
                if document.metadata.get("source", "")
                in case["relevant_sources"]
            ),
            None,
        )

        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        reciprocal_ranks.append(reciprocal_rank)
        found_count += first_relevant_rank is not None

        print(
            f"{index:02}. Rank: {first_relevant_rank or 'NOT_FOUND'}"
            f" | RR: {reciprocal_rank:.4f}"
            f" | {case['query']}"
        )

    mrr = sum(reciprocal_ranks) / len(EVALUATION_CASES)

    print(f"MRR@20: {mrr:.4f} ({mrr * 100:.2f}%)")
    print(f"Relevant source found: {found_count}/10")

    return mrr, found_count


if __name__ == "__main__":
    baseline_mrr, baseline_found = evaluate_retriever(
        retrieve_documents,
        "EXISTING RETRIEVER",
    )

    reranked_mrr, reranked_found = evaluate_retriever(
        retrieve_with_reranking,
        "CROSS-ENCODER RERANKER",
    )

    print("\n--- COMPARISON ---")
    print(f"Baseline MRR@20: {baseline_mrr * 100:.2f}%")
    print(f"Reranked MRR@20: {reranked_mrr * 100:.2f}%")
    print(
        "MRR change: "
        f"{(reranked_mrr - baseline_mrr) * 100:+.2f} percentage points"
    )
    print(f"Baseline source recall: {baseline_found}/10")
    print(f"Reranked source recall: {reranked_found}/10")
