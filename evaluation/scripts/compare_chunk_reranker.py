
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.reranker_experiment import retrieve_with_reranking

QUERIES = [
    "What is the return policy for damaged electronics?",
    "What is the general return policy?",
    "How do refunds work after a return?",
    "Can I exchange a product I purchased?",
    "How can I cancel my order?",
    "How can I track my shipment and delivery?",
    "What payment methods and payment rules are supported?",
    "What are the specifications of the Apple MacBook Air M2?",
]

K = 5


def show_results(label, retrieval_function, query):
    print(f"\n--- {label} ---")

    documents = retrieval_function(query, k=K)

    for rank, document in enumerate(documents, start=1):
        source = Path(
            str(document.metadata.get("source", "unknown"))
        ).name

        preview = " ".join(document.page_content.split())[:220]

        print(f"{rank}. Source: {source}")
        print(f"   Chunk: {preview}")


def main():
    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")

        show_results(
            "EXISTING RETRIEVER",
            retrieve_documents,
            query,
        )

        show_results(
            "CROSS-ENCODER RERANKER",
            retrieve_with_reranking,
            query,
        )


if __name__ == "__main__":
    main()
