from pathlib import Path

from ai.rag.vector_store import (
    VECTOR_DB_PATH,
    create_vector_store,
    load_vector_store,
)


def test_vector_store_creation_and_loading():

    create_vector_store()

    assert VECTOR_DB_PATH.exists()
    assert (VECTOR_DB_PATH / "index.faiss").exists()
    assert (VECTOR_DB_PATH / "index.pkl").exists()

    vector_store = load_vector_store()

    assert vector_store.index.ntotal > 0