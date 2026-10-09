import numpy as np

from utils.retriever import top_k


def test_top_k_returns_most_similar_first():
    chunks = np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]], dtype=np.float32)
    hits = top_k(np.array([1, 0, 0], dtype=np.float32), chunks, k=2)
    assert [i for i, _ in hits] == [0, 2]
