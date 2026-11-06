from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-base-en-v1.5"


def create_embeddings():
    """
    Create the BGE embedding model used by the RAG pipeline.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


if __name__ == "__main__":
    embeddings = create_embeddings()

    test_text = "What is the return policy for damaged electronics?"

    vector = embeddings.embed_query(test_text)

    print("Model:", MODEL_NAME)
    print("Embedding dimensions:", len(vector))
    print("First 5 values:", vector[:5])