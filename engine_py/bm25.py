"""BM25 scoring over the inverted index.

Uses InvertedIndex fields: doc_ids, docs_len, postings, N, avgdl, df(term).

Matches bm25s's defaults: k1=1.5, b=0.75, Lucene IDF:

    IDF(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    L        = 1 - b + b * dl / avgdl
    TF-part  = tf * (k1 + 1) / (tf + k1 * L)
    score    = sum over query terms of IDF(t) * TF-part

Search is term-at-a-time: walk each query term's postings, add into a {doc id: score}
accumulator, take the top k with heapq.nlargest. A query term repeated n times counts n times.

bm25s's "lucene" method drops the constant (k1 + 1), so these scores are exactly 2.5x its scores;
the ranking is identical.
"""
from engine_py.analyzer import Analyzer
from engine_py.index import InvertedIndex
import heapq
from pathlib import Path
from engine_py.segment import SegmentReader, write_segment
from engine_py import scoring


DEFAULT_INDEX_DIR=Path(__file__).resolve().parent.parent / "indexes" / "default"

class MyBM25Retriever:
    name: str = "my-bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75,indexes_dir:Path=DEFAULT_INDEX_DIR) -> None:
        self.k1 = k1
        self.b = b
        self.analyzer = Analyzer()
        self.path=indexes_dir / "segment_0"
        self.segment_reader:SegmentReader | None = None

    def index(self, docs: dict[str, str]) -> None:
        if self.segment_reader is not None:
            self.segment_reader.close()
            
        inverted_index=InvertedIndex(self.analyzer)
        inverted_index.build(docs=docs)
        write_segment(inverted_index,self.path)

        self.segment_reader=SegmentReader(self.path)

    def idf(self, term: str) -> float:
        return scoring.idf(N=self.segment_reader.N,df=self.segment_reader.df(term))

    def L(self,doc_len):
        return (1-self.b+(self.b*doc_len/self.segment_reader.avgdl))

    def tf(self,doc_pos:int,tf:int) -> float:
        doc_length=self.segment_reader.docs_len[doc_pos]
        L=self.L(doc_length)
        return ((self.k1+1)*tf/(tf+self.k1*L))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens=self.analyzer.analyze(query)
        results={}
        for token in tokens:
            idf=self.idf(token)
            for doc_pos,tf in self.segment_reader.postings(token):
                tf_part=self.tf(doc_pos=doc_pos,tf=tf)
                score=idf*tf_part
                results.setdefault(self.segment_reader.doc_ids[doc_pos],0)
                results[self.segment_reader.doc_ids[doc_pos]]+=score
        return heapq.nlargest(k,results.items(),key=lambda item:item[1])

