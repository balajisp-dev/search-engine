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
from collections import Counter
from engine_py.analyzer import Analyzer
from engine_py.index import InvertedIndex
import heapq
from pathlib import Path
from engine_py.segment import EXHAUSTED, SegmentReader, write_segment
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
        min_heap=[]
        freq=Counter(tokens)
        if not len(tokens):
            return []
        cursors={}
        idf_dict={}
        for term in freq.keys():
            cursors[term]=self.segment_reader.cursor(term)
            idf_dict[term]=self.idf(term)
        sorted_dict={}
        for cursor in cursors.values():
            doc_pos=cursor.doc()
            sorted_dict.setdefault(doc_pos,[])
            sorted_dict[doc_pos].append(cursor)
        while True:

            if len(min_heap)==k:
                ub_sum=0
                theta=min_heap[0][0]
                found_pivot=False
                for doc_pos,doc_cursors in sorted(sorted_dict.items()):
                    for cursor in doc_cursors:
                        ub_sum+=cursor.ub*freq[cursor.get_term()]
                        if ub_sum>theta:
                            for cursor in cursors.values():
                                if cursor.doc()<doc_pos:
                                    cursor.advance(doc_pos)
                            found_pivot=True
                            break
                    if found_pivot:break
                if not found_pivot:break
            sorted_dict={}
            for cursor in cursors.values():
                doc_pos=cursor.doc()
                sorted_dict.setdefault(doc_pos,[])
                sorted_dict[doc_pos].append(cursor)

            min_doc=min(sorted_dict)
            if min_doc==EXHAUSTED:
                break
            doc_score=0
            for cursor in sorted_dict.get(min_doc,[]):
                doc_pos=cursor.doc()
                tf=cursor.tf()
                idf=idf_dict[cursor.get_term()]
                tf_part=self.tf(doc_pos=doc_pos,tf=tf)
                doc_score+=(idf*tf_part)*freq[cursor.get_term()]
                cursor.next()
            entry=(doc_score,-1*min_doc)
            if len(min_heap)<k:
                heapq.heappush(min_heap,entry)
            elif doc_score>min_heap[0][0]:
                heapq.heappushpop(min_heap,entry)
        final_top_k=sorted(min_heap, key=lambda e: (-e[0], -e[1]))
        return [(self.segment_reader.doc_ids[-1*doc_pos],score) for score,doc_pos in final_top_k]

