from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


EMBED_MODEL = "all-MiniLM-L6-v2"

@staticmethod
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks):
    """Build a FAISS vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def search_documents(vector_store, query: str, k: int = 4):
    """
    Search vector store and return (results, top_score).
    Score is cosine similarity (0–1); higher = more relevant.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)

    if not results_with_scores:
        return [], 0.0

    
    results = [r for r, _ in results_with_scores]
    top_distance = results_with_scores[0][1]
    similarity = float(1 - (top_distance ** 2) / 2)
    similarity = max(0.0, min(1.0, similarity))

    return results, similarity
