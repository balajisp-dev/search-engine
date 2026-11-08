"""BM25 tests: index stats, formula pieces checked by hand, search behaviour, and agreement with bm25s.

Toy corpus (after analysis):  a = [cat, eat, fish]   b = [dog, chase, cat]   c = [fish, swim]
    N = 3, avgdl = 8/3, df(cat) = df(fish) = 2, df(swim) = 1
"""
from pathlib import Path

import pytest

from engine_py.analyzer import Analyzer
from engine_py.bm25 import MyBM25Retriever
from engine_py.index import InvertedIndex

SCIFACT = Path(__file__).resolve().parent.parent / "data" / "scifact"
needs_scifact = pytest.mark.skipif(not SCIFACT.exists(), reason="run scripts/download_beir.py scifact")

TOY_DOCS={"a":"cats eat fish","b":"dogs chase cats","c":"fish swim"}


def toy_retriever():
    retriever=MyBM25Retriever()
    retriever.index(TOY_DOCS)
    return retriever

def ids(results):
    return [doc_id for doc_id,_ in results]


# ---- index ----

def test_index_stats():
    index=toy_retriever().inverted_index
    assert index.N == 3
    assert index.docs_len == [3,3,2]
    assert index.avgdl == pytest.approx(8/3,abs=1e-4)
    assert index.doc_ids == ["a","b","c"]

def test_index_postings():
    index=toy_retriever().inverted_index
    assert index.postings["cat"] == [(0,1),(1,1)]
    assert index.postings["swim"] == [(2,1)]

def test_index_df():
    index=toy_retriever().inverted_index
    assert index.df("cat") == 2
    assert index.df("swim") == 1
    assert index.df("zebra") == 0

def test_index_one_posting_per_doc():
    # a repeated word must give ONE posting with tf=2, not two postings
    index=InvertedIndex(Analyzer())
    index.build({"d1":"dog is dog"})
    assert index.postings["dog"] == [(0,2)]
    assert index.df("dog") == 1

def test_index_rebuild_resets():
    index=InvertedIndex(Analyzer())
    index.build(TOY_DOCS)
    index.build({"x":"zebra zebra"})
    assert index.doc_ids == ["x"]
    assert index.docs_len == [2]
    assert index.avgdl == pytest.approx(2.0,abs=1e-4)
    assert "cat" not in index.postings


# ---- formula pieces ----

def test_idf_common_term():
    # ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    EXPECTED=0.4700
    assert toy_retriever().idf("cat") == pytest.approx(EXPECTED,abs=1e-4)

def test_idf_rare_term():
    # ln(1 + (3 - 1 + 0.5) / (1 + 0.5))
    EXPECTED=0.9808
    assert toy_retriever().idf("swim") == pytest.approx(EXPECTED,abs=1e-4)

def test_idf_unseen_term():
    # df = 0 must not divide by zero: ln(1 + 3.5 / 0.5) = ln(8)
    EXPECTED=2.0794
    assert toy_retriever().idf("zebra") == pytest.approx(EXPECTED,abs=1e-4)

def test_tf_longer_than_avg():
    # doc a: dl=3, L = 0.25 + 0.75 * 3/(8/3) = 1.09375 -> 2.5 / (1 + 1.5 * 1.09375)
    EXPECTED=0.9467
    assert toy_retriever().tf(doc_pos=0,tf=1) == pytest.approx(EXPECTED,abs=1e-4)

def test_tf_shorter_than_avg():
    # doc c: dl=2, L = 0.8125 -> 2.5 / (1 + 1.5 * 0.8125); shorter doc, bigger TF part
    EXPECTED=1.1268
    assert toy_retriever().tf(doc_pos=2,tf=1) == pytest.approx(EXPECTED,abs=1e-4)


# ---- search ----

def test_search_ranking():
    results=toy_retriever().search("cat fish",3)
    assert ids(results) == ["a","c","b"]
    # a matches both terms: 2 * idf(cat) * tf-part(a) = 2 * 0.4700 * 0.9467
    assert results[0][1] == pytest.approx(0.8899,abs=1e-4)

def test_search_repeated_query_term():
    # "cat" counts twice, so b (which only has cat) overtakes c (which only has fish)
    results=toy_retriever().search("cat cat fish",3)
    assert ids(results) == ["a","b","c"]

def test_search_unknown_term():
    assert toy_retriever().search("zebra",3) == []

def test_search_empty_and_stopword_query():
    retriever=toy_retriever()
    assert retriever.search("",3) == []
    assert retriever.search("the of",3) == []

def test_search_k_larger_than_corpus():
    assert len(toy_retriever().search("cat fish",100)) == 3

def test_search_returns_float_scores():
    results=toy_retriever().search("cat fish",2)
    scores=[score for _,score in results]
    assert all(type(score) is float for score in scores)
    assert scores == sorted(scores,reverse=True)


# ---- against bm25s ----

def test_scores_are_bm25s_times_k1_plus_1():
    # bm25s's "lucene" method drops the constant (k1 + 1) = 2.5; ranking is unaffected
    import bm25s
    import Stemmer
    stemmer=Stemmer.Stemmer("english")
    model=bm25s.BM25()
    model.index(bm25s.tokenize(list(TOY_DOCS.values()),stopwords="en",stemmer=stemmer,show_progress=False),show_progress=False)
    positions,scores=model.retrieve(bm25s.tokenize(["cat fish"],stopwords="en",stemmer=stemmer,show_progress=False),k=3,show_progress=False)
    doc_ids=list(TOY_DOCS)
    bm25s_scores={doc_ids[position]:float(score) for position,score in zip(positions[0],scores[0])}
    for doc_id,score in toy_retriever().search("cat fish",3):
        assert score/2.5 == pytest.approx(bm25s_scores[doc_id],abs=1e-4), doc_id

@needs_scifact
def test_ranking_matches_bm25s_on_scifact():
    from harness.baselines import BM25sRetriever
    from harness.datasets import load_dataset
    dataset=load_dataset("scifact")
    mine=MyBM25Retriever()
    reference=BM25sRetriever()
    mine.index(dataset.corpus)
    reference.index(dataset.corpus)
    for q_id,query in dataset.queries.items():
        assert ids(mine.search(query,10)) == ids(reference.search(query,10)), q_id
