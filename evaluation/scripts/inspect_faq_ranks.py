
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag.retriever import retrieve_documents

CASES = [
    (
        "I placed an order by mistake. Is there a way to stop it?",
        {"FAQ007", "FAQ008"},
    ),
    (
        "My package has not arrived yet. Where can I check its status?",
        {"FAQ009", "FAQ013"},
    ),
]

for query, expected_ids in CASES:
    print("\nQUERY:", query)
    print("Expected FAQ IDs:", ", ".join(sorted(expected_ids)))

    documents = retrieve_documents(query, k=20)

    for rank, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "")
        content = document.page_content.replace("\n", " ")

        if source == "faq.csv":
            faq_id = content.split()[1] if content.startswith("id: ") else "unknown"
            if faq_id in expected_ids or rank <= 5:
                print(f"Rank {rank}: {faq_id} | {content[:150]}")
        elif rank <= 5 or source in {
            "cancellation_policy.txt",
            "shipping_policy.txt",
        }:
            print(f"Rank {rank}: {source} | {content[:150]}")
