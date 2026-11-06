
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from ai.rag.chunker import chunk_documents
from ai.rag.loader import load_knowledge_base
from ai.rag.vector_store import load_vector_store


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=1)
def load_bm25_index():
    """Build a BM25 index over the current knowledge-base chunks."""
    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("Cannot build BM25 index: no document chunks found.")

    tokenized_chunks = [tokenize(doc.page_content) for doc in chunks]
    index = BM25Okapi(tokenized_chunks)

    return chunks, index


def retrieve_hybrid(
    query: str,
    k: int = 5,
    candidate_k: int = 60,
    rrf_constant: int = 60,
):
    """
    Combine BM25 and FAISS rankings with Reciprocal Rank Fusion.

    The original retriever.py is not modified.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than zero")

    chunks, bm25_index = load_bm25_index()
    vector_store = load_vector_store()

    # BM25 candidates are indices into the same chunk list.
    bm25_scores = bm25_index.get_scores(tokenize(query))
    bm25_order = sorted(
        range(len(chunks)),
        key=lambda i: float(bm25_scores[i]),
        reverse=True,
    )[:candidate_k]

    # Build a stable mapping from document content + metadata to index.
    # Lists handle duplicate chunks without collapsing their identities.
    chunk_indices = {}
    for i, doc in enumerate(chunks):
        key = (doc.page_content, repr(sorted(doc.metadata.items())))
        chunk_indices.setdefault(key, []).append(i)

    # Retrieve semantic candidates from FAISS.
    faiss_order = vector_store.similarity_search(
        query,
        k=min(candidate_k, vector_store.index.ntotal),
    )

    # Map FAISS results to chunks using their content and metadata.
    # Consume each duplicate match in its original order.
    used_faiss_indices = set()
    faiss_indices = []

    for doc in faiss_order:
        key = (doc.page_content, repr(sorted(doc.metadata.items())))
        matches = chunk_indices.get(key, [])

        matched_index = next(
            (i for i in matches if i not in used_faiss_indices),
            None,
        )

        if matched_index is not None:
            used_faiss_indices.add(matched_index)
            faiss_indices.append(matched_index)

    # Reciprocal Rank Fusion combines rank positions, not raw scores.
    fused_scores = {}

    for rank, index in enumerate(bm25_order, start=1):
        fused_scores[index] = (
            fused_scores.get(index, 0.0)
            + 1.0 / (rrf_constant + rank)
        )

    for rank, index in enumerate(faiss_indices, start=1):
        fused_scores[index] = (
            fused_scores.get(index, 0.0)
            + 1.0 / (rrf_constant + rank)
        )

    ranked_indices = sorted(
        fused_scores,
        key=lambda i: fused_scores[i],
        reverse=True,
    )

    return [chunks[i] for i in ranked_indices[:k]]



def diagnose_retrieval(query: str, k: int = 5):
    """Print BM25 and FAISS rankings separately for debugging."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    chunks, bm25_index = load_bm25_index()
    vector_store = load_vector_store()

    # BM25's top results
    bm25_scores = bm25_index.get_scores(tokenize(query))
    bm25_order = sorted(
        range(len(chunks)),
        key=lambda i: float(bm25_scores[i]),
        reverse=True,
    )[:k]

    print("\nBM25 RESULTS")
    for rank, index in enumerate(bm25_order, start=1):
        doc = chunks[index]
        print(
            f"{rank}. {doc.metadata.get('source')} | "
            f"{doc.page_content[:160].replace(chr(10), ' ')}"
        )

    # FAISS's top results
    faiss_docs = vector_store.similarity_search(query, k=k)

    print("\nFAISS RESULTS")
    for rank, doc in enumerate(faiss_docs, start=1):
        print(
            f"{rank}. {doc.metadata.get('source')} | "
            f"{doc.page_content[:160].replace(chr(10), ' ')}"
        )



def retrieve_hybrid_with_constant(
    query: str,
    k: int = 20,
    candidate_k: int = 60,
    rrf_constant: int = 10,
):
    """Experimental hybrid retrieval with a configurable RRF constant."""
    return retrieve_hybrid(
        query=query,
        k=k,
        candidate_k=candidate_k,
        rrf_constant=rrf_constant,
    )
