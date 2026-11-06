
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.hybrid_retriever import retrieve_hybrid
from ai.rag.vector_store import load_vector_store

# These labels identify FAQ answers that directly address each query.
# Policy documents are tracked separately from FAQ relevance.
CASES = [
    {
        "query": "I placed an order by mistake. Is there a way to stop it?",
        "relevant_faq_ids": {"FAQ007", "FAQ008", "FAQ024"},
        "relevant_policy": "cancellation_policy.txt",
    },
    {
        "query": "My package has not arrived yet. Where can I check its status?",
        "relevant_faq_ids": {"FAQ009", "FAQ013"},
        "relevant_policy": "shipping_policy.txt",
    },
]

K = 20


def get_faq_id(document):
    text = document.page_content
    if "id: FAQ" not in text:
        return None

    start = text.find("id: FAQ")
    return text[start + len("id: "):].split()[0].strip()


def is_relevant(document, case):
    faq_id = get_faq_id(document)

    if faq_id in case["relevant_faq_ids"]:
        return True

    source = Path(str(document.metadata.get("source", ""))).name
    return source.lower() == case["relevant_policy"].lower()


def first_relevant_rank(documents, case):
    for rank, document in enumerate(documents, start=1):
        if is_relevant(document, case):
            return rank
    return None


def main():
    store = load_vector_store()
    results = []

    for case in CASES:
        query = case["query"]

        methods = {
            "FAISS": store.similarity_search(query, k=K),
            "Current retriever": retrieve_documents(query, k=K),
            "Hybrid RRF 10": retrieve_hybrid(
                query, k=K, rrf_constant=10
            ),
        }

        print("\nQUERY:", query)

        for method_name, documents in methods.items():
            rank = first_relevant_rank(documents, case)
            reciprocal_rank = 1 / rank if rank else 0

            print(
                f"{method_name}: first relevant rank = "
                f"{rank if rank else 'NOT_FOUND'}; "
                f"reciprocal rank = {reciprocal_rank:.4f}"
            )

            results.append({
                "query": query,
                "method": method_name,
                "first_relevant_rank": rank or "",
                "reciprocal_rank": reciprocal_rank,
            })

            if method_name == "Hybrid RRF 10":
                print("Top 5 hybrid results:")
                for i, document in enumerate(documents[:5], start=1):
                    print(
                        f"  {i}. FAQ={get_faq_id(document)} | "
                        f"{document.metadata.get('source')} | "
                        f"{document.page_content[:130].replace(chr(10), ' ')}"
                    )

    output_path = PROJECT_ROOT / "evaluation" / "results" / (
        "faq_relevance_comparison_v2.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
