from ai.rag.retriever import retrieve_documents


def test_macbook_air_m2_retrieval():
    query = "What are the specifications of the Apple MacBook Air M2?"
    docs = retrieve_documents(query, k=5)

    print("\n--- MacBook Air M2 retrieval results ---")

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- RESULT {i} ---")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:350])