from ai.rag.retriever import retrieve_documents

queries = [
    "Can I exchange a product I purchased?",
    "How can I cancel my order?",
]

for query in queries:
    print("\nQUESTION:", query)
    documents = retrieve_documents(query, k=3)

    for rank, doc in enumerate(documents, start=1):
        print(f"Rank: {rank} | Source: {doc.metadata.get('source')}")
        print(doc.page_content[:250].replace("\n", " "))
