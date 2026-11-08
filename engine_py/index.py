"""In-memory inverted index.

After build(docs):
    doc_ids    list[str]                         position -> doc id
    docs_len   list[int]                         position -> number of tokens in that doc
    postings   dict[str, list[tuple[int, int]]]  term -> [(doc position, term frequency), ...]
    N          int                               number of docs
    avgdl      float                             average doc length (in tokens)

Postings store integer positions, not string ids (smaller, faster, and what Phase 2 compresses).
Each doc contributes at most one posting per term, so df(term) is the postings list length.
"""
from collections import Counter

from engine_py.analyzer import Analyzer


class InvertedIndex:
    def __init__(self, analyzer: Analyzer) -> None:
        self.analyzer: Analyzer = analyzer
        self.doc_ids: list[str] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.avgdl: float = 0
        self.docs_len: list[int] = []
        self.N: int = 0

    def reset_data(self) -> None:
        """Clear everything from a previous build so positions can't point at old docs."""
        self.postings.clear()
        self.avgdl = 0
        self.docs_len = []

    def build(self, docs: dict[str, str]) -> None:
        """Analyze every doc and record its length and one (position, tf) posting per distinct term."""
        self.reset_data()
        self.doc_ids = list(docs.keys())
        self.N = len(docs)
        for i, text in enumerate(docs.values()):
            tokens = self.analyzer.analyze(text)
            self.avgdl += len(tokens)
            self.docs_len.append(len(tokens))
            counts = Counter(tokens)
            for t, tf in counts.items():
                self.postings.setdefault(t, [])
                self.postings[t].append((i, tf))
        self.avgdl /= self.N

    def df(self, term: str) -> int:
        """Number of docs containing term (0 if unseen)."""
        return len(self.postings.get(term, []))
    
    
