"""BM25 formula pieces shared by the scorer (bm25.py) and the segment writer (segment.py).

Keeping them in one place means the scores at query time and the score bounds stored at
write time (UB, block_max) can never disagree.
"""
import math


def idf(N: int, df: int) -> float:
    """Lucene BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5)). Never negative; finite for df = 0."""
    return math.log(1 + (N - df + 0.5) / (df + 0.5))
