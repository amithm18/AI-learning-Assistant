"""
Step 3 of the pipeline: the "R" in RAG (Retrieval).

An embedding model turns a piece of text into a list of 384 numbers (a vector).
Texts with similar meaning get vectors that point in a similar direction, so
"How do you add an element to a stack?" lands close to the paragraph about push().

- When a PDF is uploaded, every chunk is embedded once and saved (vectors.npy).
- When we need context, the query is embedded and compared with every chunk
  vector using cosine similarity. The highest-scoring chunks are sent to the LLM.

The model (BAAI/bge-small-en-v1.5) runs locally through fastembed, so it is free
and needs no extra API key. It is downloaded once (~130MB) on first use.
"""
import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def _get_model():
    # Loading the model takes a few seconds, so do it once and reuse it
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_chunks(texts):
    """Returns a (number_of_chunks x 384) matrix of unit-length vectors."""
    vectors = np.array(list(_get_model().embed(texts)), dtype=np.float32)
    return _normalize(vectors)


def embed_query(text):
    # query_embed adds the instruction prefix bge models expect for search queries
    vector = np.array(next(iter(_get_model().query_embed(text))), dtype=np.float32)
    return _normalize(vector)


def top_k(query_vector, chunk_vectors, k=4):
    """
    Returns [(chunk_index, score), ...] for the k most similar chunks.
    All vectors have length 1, so a dot product equals cosine similarity.
    """
    scores = chunk_vectors @ query_vector
    best = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in best]


def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)
