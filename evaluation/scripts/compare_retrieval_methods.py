
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.vector_store import load_vector_store

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
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def first_relevant_rank(documents, relevant_sources):
    for rank, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "")
        if source in relevant_sources:
            return rank
    return None


vector_store = load_vector_store()
results = []
baseline_rrs = []
reranked_rrs = []

for index, case in enumerate(EVALUATION_CASES, start=1):
    query = case["query"]
    expected = case["relevant_sources"]

    # Baseline: original FAISS similarity order.
    baseline_docs = vector_store.similarity_search(query, k=K)

    # Current method: FAISS candidates plus topic-based reranking.
    reranked_docs = retrieve_documents(query, k=K)

    baseline_rank = first_relevant_rank(baseline_docs, expected)
    reranked_rank = first_relevant_rank(reranked_docs, expected)

    baseline_rr = 1 / baseline_rank if baseline_rank else 0.0
    reranked_rr = 1 / reranked_rank if reranked_rank else 0.0

    baseline_rrs.append(baseline_rr)
    reranked_rrs.append(reranked_rr)

    results.append({
        "case_id": index,
        "query": query,
        "expected_sources": ";".join(sorted(expected)),
        "baseline_rank": baseline_rank or "NOT_FOUND",
        "baseline_rr": round(baseline_rr, 4),
        "reranked_rank": reranked_rank or "NOT_FOUND",
        "reranked_rr": round(reranked_rr, 4),
        "reranked_rank_change": (
            baseline_rank - reranked_rank
            if baseline_rank and reranked_rank
            else "N/A"
        ),
    })

    print(
        f"{index:02}. Baseline rank: {baseline_rank or 'NOT_FOUND':>8}"
        f" | Reranked rank: {reranked_rank or 'NOT_FOUND':>8}"
        f" | {query}"
    )

baseline_mrr = sum(baseline_rrs) / len(baseline_rrs)
reranked_mrr = sum(reranked_rrs) / len(reranked_rrs)

summary = [
    {
        "method": "Plain FAISS",
        "mrr_at_20": round(baseline_mrr, 4),
        "percentage": round(baseline_mrr * 100, 2),
        "queries_with_relevant_source": sum(
            rank > 0 for rank in baseline_rrs
        ),
        "queries_evaluated": len(EVALUATION_CASES),
        "k": K,
    },
    {
        "method": "FAISS + topic reranking",
        "mrr_at_20": round(reranked_mrr, 4),
        "percentage": round(reranked_mrr * 100, 2),
        "queries_with_relevant_source": sum(
            rank > 0 for rank in reranked_rrs
        ),
        "queries_evaluated": len(EVALUATION_CASES),
        "k": K,
    },
]

case_file = RESULTS_DIR / "retrieval_comparison_cases.csv"
summary_file = RESULTS_DIR / "retrieval_comparison_summary.csv"

with case_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with summary_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\n--- Retrieval Comparison ---")
for item in summary:
    print(
        f"{item['method']}: MRR@20 = {item['percentage']:.2f}%"
        f" | Relevant source found: "
        f"{item['queries_with_relevant_source']}/"
        f"{item['queries_evaluated']}"
    )

print(
    f"\nMRR change: {(reranked_mrr - baseline_mrr) * 100:+.2f}"
    " percentage points"
)
print(f"Per-query comparison: {case_file}")
print(f"Summary: {summary_file}")
