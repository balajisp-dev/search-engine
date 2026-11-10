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


@pytest.fixture
def toy(tmp_path):
    """The toy corpus indexed into a throwaway folder, so tests never touch indexes/."""
    retriever=MyBM25Retriever(indexes_dir=tmp_path)
    retriever.index(TOY_DOCS)
    return retriever

def toy_index():
    index=InvertedIndex(Analyzer())
    index.build(TOY_DOCS)
    return index

def ids(results):
    return [doc_id for doc_id,_ in results]


# ---- index ----

def test_index_stats():
    index=toy_index()
    assert index.N == 3
    assert index.docs_len == [3,3,2]
    assert index.avgdl == pytest.approx(8/3,abs=1e-4)
    assert index.doc_ids == ["a","b","c"]

def test_index_postings():
    index=toy_index()
    assert index.postings["cat"] == [(0,1),(1,1)]
    assert index.postings["swim"] == [(2,1)]

def test_index_df():
    index=toy_index()
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

def test_idf_common_term(toy):
    # ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    EXPECTED=0.4700
    assert toy.idf("cat") == pytest.approx(EXPECTED,abs=1e-4)

def test_idf_rare_term(toy):
    # ln(1 + (3 - 1 + 0.5) / (1 + 0.5))
    EXPECTED=0.9808
    assert toy.idf("swim") == pytest.approx(EXPECTED,abs=1e-4)

def test_idf_unseen_term(toy):
    # df = 0 must not divide by zero: ln(1 + 3.5 / 0.5) = ln(8)
    EXPECTED=2.0794
    assert toy.idf("zebra") == pytest.approx(EXPECTED,abs=1e-4)

def test_tf_longer_than_avg(toy):
    # doc a: dl=3, L = 0.25 + 0.75 * 3/(8/3) = 1.09375 -> 2.5 / (1 + 1.5 * 1.09375)
    EXPECTED=0.9467
    assert toy.tf(doc_pos=0,tf=1) == pytest.approx(EXPECTED,abs=1e-4)

def test_tf_shorter_than_avg(toy):
    # doc c: dl=2, L = 0.8125 -> 2.5 / (1 + 1.5 * 0.8125); shorter doc, bigger TF part
    EXPECTED=1.1268
    assert toy.tf(doc_pos=2,tf=1) == pytest.approx(EXPECTED,abs=1e-4)


# ---- search ----

def test_search_ranking(toy):
    results=toy.search("cat fish",3)
    assert ids(results) == ["a","c","b"]
    # a matches both terms: 2 * idf(cat) * tf-part(a) = 2 * 0.4700 * 0.9467
    assert results[0][1] == pytest.approx(0.8899,abs=1e-4)

def test_search_repeated_query_term(toy):
    # "cat" counts twice, so b (which only has cat) overtakes c (which only has fish)
    results=toy.search("cat cat fish",3)
    assert ids(results) == ["a","b","c"]

def test_search_unknown_term(toy):
    assert toy.search("zebra",3) == []

def test_search_empty_and_stopword_query(toy):
    retriever=toy
    assert retriever.search("",3) == []
    assert retriever.search("the of",3) == []

def test_search_k_larger_than_corpus(toy):
    assert len(toy.search("cat fish",100)) == 3

def test_search_returns_float_scores(toy):
    results=toy.search("cat fish",2)
    scores=[score for _,score in results]
    assert all(type(score) is float for score in scores)
    assert scores == sorted(scores,reverse=True)


# ---- against bm25s ----

def test_scores_are_bm25s_times_k1_plus_1(toy):
    # bm25s's "lucene" method drops the constant (k1 + 1) = 2.5; ranking is unaffected
    import bm25s
    import Stemmer
    stemmer=Stemmer.Stemmer("english")
    model=bm25s.BM25()
    model.index(bm25s.tokenize(list(TOY_DOCS.values()),stopwords="en",stemmer=stemmer,show_progress=False),show_progress=False)
    positions,scores=model.retrieve(bm25s.tokenize(["cat fish"],stopwords="en",stemmer=stemmer,show_progress=False),k=3,show_progress=False)
    doc_ids=list(TOY_DOCS)
    bm25s_scores={doc_ids[position]:float(score) for position,score in zip(positions[0],scores[0])}
    for doc_id,score in toy.search("cat fish",3):
        assert score/2.5 == pytest.approx(bm25s_scores[doc_id],abs=1e-4), doc_id

@needs_scifact
def test_ranking_matches_bm25s_on_scifact(tmp_path):
    from harness.baselines import BM25sRetriever
    from harness.datasets import load_dataset
    dataset=load_dataset("scifact")
    mine=MyBM25Retriever(indexes_dir=tmp_path)
    reference=BM25sRetriever()
    mine.index(dataset.corpus)
    reference.index(dataset.corpus)
    for q_id,query in dataset.queries.items():
        assert ids(mine.search(query,10)) == ids(reference.search(query,10)), q_id


# ---- search modes: taat, daat, wand and bmw must return exactly the same top k ----

TOY_QUERIES=["cats","fish","cats fish","dogs swim fish","fish fish cats","zebra"]

@pytest.mark.parametrize("mode",["daat","wand","bmw"])
def test_modes_match_taat_on_toy(tmp_path,mode):
    taat=MyBM25Retriever(indexes_dir=tmp_path/"taat",mode="taat")
    other=MyBM25Retriever(indexes_dir=tmp_path/mode,mode=mode)
    taat.index(TOY_DOCS)
    other.index(TOY_DOCS)
    for query in TOY_QUERIES:
        for k in (1,2,3,10):
            assert ids(other.search(query,k)) == ids(taat.search(query,k)), (query,k)

def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        MyBM25Retriever(mode="fast")

def test_name_includes_mode():
    assert MyBM25Retriever(mode="wand").name == "my-bm25-wand"


@pytest.fixture(scope="module")
def scifact_modes(tmp_path_factory):
    """One SciFact index, opened by a retriever per mode (indexing once keeps this test fast)."""
    from harness.datasets import load_dataset
    dataset=load_dataset("scifact")
    path=tmp_path_factory.mktemp("scifact")
    retrievers={}
    for mode in MyBM25Retriever.MODES:
        retriever=MyBM25Retriever(indexes_dir=path/mode,mode=mode)
        retriever.index(dataset.corpus)
        retrievers[mode]=retriever
    return dataset,retrievers

@needs_scifact
@pytest.mark.parametrize("k",[10,101])
def test_modes_match_taat_on_scifact(scifact_modes,k):
    dataset,retrievers=scifact_modes
    for q_id,query in dataset.queries.items():
        expected=ids(retrievers["taat"].search(query,k))
        for mode in ("daat","wand","bmw"):
            assert ids(retrievers[mode].search(query,k)) == expected, (mode,q_id,k)

@needs_scifact
def test_skipping_scores_fewer_postings_on_scifact(scifact_modes):
    dataset,retrievers=scifact_modes
    scored={}
    for mode in ("daat","wand","bmw"):
        retriever=retrievers[mode]
        retriever.stats={"postings_scored":0,"blocks_decoded":0}
        for query in dataset.queries.values():
            retriever.search(query,10)
        scored[mode]=retriever.stats["postings_scored"]
    assert scored["bmw"] < scored["wand"] < scored["daat"]
