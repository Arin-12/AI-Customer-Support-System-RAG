import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
# These are NEW questions, different from the original MRR@5 questions.
# Each question has an expected relevant source file.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

case_results = []
reciprocal_ranks = []

for index, case in enumerate(EVALUATION_CASES, start=1):
    documents = retrieve_documents(case["query"], k=K)

    relevant_sources = case["relevant_sources"]
    first_relevant_rank = None
    retrieved_sources = []

    # Rank documents in the exact order returned by the retriever.
    # Do not apply a second keyword-scoring step here.
    for rank, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "")
        retrieved_sources.append(source)

        if source in relevant_sources and first_relevant_rank is None:
            first_relevant_rank = rank

    reciprocal_rank = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    reciprocal_ranks.append(reciprocal_rank)

    case_results.append({
        "case_id": index,
        "query": case["query"],
        "expected_sources": ";".join(sorted(relevant_sources)),
        "first_relevant_rank": first_relevant_rank or "NOT_FOUND",
        "reciprocal_rank": round(reciprocal_rank, 4),
        "retrieved_document_count": len(documents),
        "relevant_source_found": first_relevant_rank is not None,
    })

    print(
        f"{index:02}. Rank: {first_relevant_rank or 'NOT_FOUND'}"
        f" | RR: {reciprocal_rank:.4f}"
        f" | {case['query']}"
    )

mrr_at_20 = sum(reciprocal_ranks) / len(reciprocal_ranks)
found_count = sum(rank > 0 for rank in reciprocal_ranks)

case_file = RESULTS_DIR / "mrr20_case_results.csv"
metrics_file = RESULTS_DIR / "mrr20_metrics.csv"

with case_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=case_results[0].keys())
    writer.writeheader()
    writer.writerows(case_results)

summary = [{
    "metric": "MRR@20",
    "value": round(mrr_at_20, 4),
    "percentage": round(mrr_at_20 * 100, 2),
    "queries_evaluated": len(EVALUATION_CASES),
    "queries_with_relevant_source": found_count,
    "k": K,
    "evaluation_set": "new_questions",
}]

with metrics_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\n--- MRR@20 Results ---")
print(f"MRR@20: {mrr_at_20:.4f} ({mrr_at_20 * 100:.2f}%)")
print(f"Queries evaluated: {len(EVALUATION_CASES)}")
print(f"Relevant source found in top 20: {found_count}/{len(EVALUATION_CASES)}")
print(f"Per-query results: {case_file}")
print(f"Summary metrics: {metrics_file}")
