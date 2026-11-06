
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents
from ai.rag.vector_store import load_vector_store

CASES = [
    {
        "query": "I placed an order by mistake. Is there a way to stop it?",
        "relevant_sources": {"cancellation_policy.txt", "faq.csv"},
    },
    {
        "query": "My package has not arrived yet. Where can I check its status?",
        "relevant_sources": {"shipping_policy.txt", "faq.csv"},
    },
]

K = 20
store = load_vector_store()

for case in CASES:
    query = case["query"]
    relevant_sources = case["relevant_sources"]

    baseline = store.similarity_search(query, k=K)
    reranked = retrieve_documents(query, k=K)

    def first_relevant_rank(documents):
        for rank, doc in enumerate(documents, start=1):
            if doc.metadata.get("source") in relevant_sources:
                return rank
        return None

    baseline_rank = first_relevant_rank(baseline)
    reranked_rank = first_relevant_rank(reranked)

    print("\nQUERY:", query)
    print("Relevant sources:", ", ".join(sorted(relevant_sources)))
    print("Plain FAISS first relevant rank:", baseline_rank or "NOT_FOUND")
    print("Current reranker first relevant rank:", reranked_rank or "NOT_FOUND")

    print("\nTop 5 reranked results:")
    for rank, doc in enumerate(reranked[:5], start=1):
        print(
            f"{rank}. {doc.metadata.get('source')} | "
            f"{doc.page_content[:160].replace(chr(10), ' ')}"
        )
