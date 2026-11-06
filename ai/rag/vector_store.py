
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS

from ai.rag.loader import load_knowledge_base
from ai.rag.chunker import chunk_documents
from ai.rag.embeddings import create_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db" / "faiss_index"


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Create the embedding model once and reuse it.
    """
    return create_embeddings()


def create_vector_store():
    """
    Build and save a fresh FAISS index from the current knowledge base.
    """
    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_DB_PATH))

    # The saved index has changed. Clear the cached copy so the next
    # load_vector_store() call loads the newly built index.
    load_vector_store.cache_clear()

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")

    return vector_store


@lru_cache(maxsize=1)
def load_vector_store():
    """
    Load the FAISS index once and reuse it between queries.
    """
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {VECTOR_DB_PATH}. "
            "Build it first by running create_vector_store()."
        )

    index_file = VECTOR_DB_PATH / "index.faiss"
    metadata_file = VECTOR_DB_PATH / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"The FAISS index is incomplete at {VECTOR_DB_PATH}. "
            "Rebuild it using create_vector_store()."
        )

    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def clear_vector_store_cache():
    """
    Clear cached index and embedding objects when needed.
    """
    load_vector_store.cache_clear()
    get_embeddings.cache_clear()


if __name__ == "__main__":
    vector_store = create_vector_store()

    print("FAISS vector store created successfully.")
    print("Location:", VECTOR_DB_PATH)
    print("Indexed vectors:", vector_store.index.ntotal)
