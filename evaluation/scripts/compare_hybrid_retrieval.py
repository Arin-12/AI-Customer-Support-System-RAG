
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.hybrid_retriever import retrieve_hybrid
from ai.rag.vector_store import load_vector_store

QUERIES = [
    ("A device arrived with a cracked screen. What should I do?",
     "damaged_product_policy.txt"),
    ("Can I send back an item if I changed my mind?",
     "return_policy.txt"),
    ("When should I expect my money back once a return is approved?",
     "refund_policy.txt"),
    ("The model I bought is unsuitable. Can I swap it for another one?",
     "exchange_policy.txt"),
    ("I placed an order by mistake. Is there a way to stop it?",
     "cancellation_policy.txt"),
    ("My package has not arrived yet. Where can I check its status?",
     "shipping_policy.txt"),
    ("My card payment failed. Which payment options can I use?",
     "payment_policy.txt"),
    ("What does the warranty cover if my gadget stops working?",
     "warranty_policy.txt"),
    ("What features and specifications does the MacBook Air M2 have?",
     "product_information.csv"),
    ("Is there a deadline for returning an opened electronics item?",
     "return_policy.txt"),
]

K = 20


def reciprocal_rank(documents, expected_source):
    for rank, doc in enumerate(documents, start=1):
        source = Path(str(doc.metadata.get("source", ""))).name
        if source.lower() == expected_source.lower():
            return rank
    return None


def main():
    vector_store = load_vector_store()
    rows = []

    for number, (query, expected_source) in enumerate(QUERIES, start=1):
        baseline_docs = vector_store.similarity_search(query, k=K)
        current_docs = retrieve_documents(query, k=K)
        hybrid_docs = retrieve_hybrid(query, k=K)

        baseline_rank = reciprocal_rank(baseline_docs, expected_source)
        current_rank = reciprocal_rank(current_docs, expected_source)
        hybrid_rank = reciprocal_rank(hybrid_docs, expected_source)

        rows.append({
            "query_number": number,
            "query": query,
            "expected_source": expected_source,
            "baseline_rank": baseline_rank or "",
            "current_retriever_rank": current_rank or "",
            "hybrid_rank": hybrid_rank or "",
            "baseline_reciprocal_rank": 1 / baseline_rank if baseline_rank else 0,
            "current_reciprocal_rank": 1 / current_rank if current_rank else 0,
            "hybrid_reciprocal_rank": 1 / hybrid_rank if hybrid_rank else 0,
        })

        print(f"{number:02}. Baseline: {str(baseline_rank):>4} | "
              f"Current: {str(current_rank):>4} | "
              f"Hybrid: {str(hybrid_rank):>4} | {query}")

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = output_dir / "hybrid_retrieval_comparison_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- MRR@20 comparison ---")
    methods = [
        ("Plain FAISS", "baseline_rank", "baseline_reciprocal_rank"),
        ("Current retriever", "current_retriever_rank",
         "current_reciprocal_rank"),
        ("BM25 + FAISS hybrid", "hybrid_rank", "hybrid_reciprocal_rank"),
    ]

    for name, rank_key, rr_key in methods:
        mrr = sum(row[rr_key] for row in rows) / len(rows)
        found = sum(bool(row[rank_key]) for row in rows)
        print(f"{name}: MRR@20 = {mrr:.2%} | "
              f"Relevant source found: {found}/{len(rows)}")

    print(f"\nDetailed results saved to: {cases_path.resolve()}")


if __name__ == "__main__":
    main()
