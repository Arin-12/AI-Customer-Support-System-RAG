

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
CONSTANTS = [10, 20, 60]


def find_rank(documents, expected_source):
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
        baseline_rank = find_rank(baseline_docs, expected_source)

        row = {
            "query_number": number,
            "query": query,
            "expected_source": expected_source,
            "faiss_rank": baseline_rank or "",
            "faiss_rr": 1 / baseline_rank if baseline_rank else 0,
        }

        print(f"\n{number:02}. {query}")
        print(f"    FAISS rank: {baseline_rank}")

        for constant in CONSTANTS:
            docs = retrieve_hybrid(
                query=query,
                k=K,
                rrf_constant=constant,
            )
            rank = find_rank(docs, expected_source)

            row[f"rrf_{constant}_rank"] = rank or ""
            row[f"rrf_{constant}_rr"] = 1 / rank if rank else 0

            print(f"    RRF constant {constant}: rank {rank}")

        rows.append(row)

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rrf_constant_comparison.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- MRR@20 summary ---")
    methods = [
        ("FAISS only", "faiss_rr"),
        ("RRF constant 10", "rrf_10_rr"),
        ("RRF constant 20", "rrf_20_rr"),
        ("RRF constant 60", "rrf_60_rr"),
    ]

    for name, key in methods:
        mrr = sum(row[key] for row in rows) / len(rows)
        found = sum(bool(row[
            "faiss_rank" if key == "faiss_rr"
            else key.replace("_rr", "_rank")
        ]) for row in rows)
        print(f"{name}: MRR@20 = {mrr:.2%} | Found {found}/10")

    print(f"\nDetailed results: {output_path.resolve()}")


if __name__ == "__main__":
    main()
