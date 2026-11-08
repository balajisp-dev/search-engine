"""Analyzer tests: hand cases for each pipeline step, plus an exact match with bm25s on SciFact."""
from pathlib import Path

import pytest

from engine_py.analyzer import Analyzer

SCIFACT = Path(__file__).resolve().parent.parent / "data" / "scifact"
needs_scifact = pytest.mark.skipif(not SCIFACT.exists(), reason="run scripts/download_beir.py scifact")


def test_analyzer_full_pipeline():
    text="The cats are RUNNING to a big park!"
    EXPECTED=["cat","run","big","park"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_hyphen_splits():
    text="COVID-19 vaccine"
    EXPECTED=["covid","19","vaccin"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_drops_single_chars():
    text="a b c x2"
    EXPECTED=["x2"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_drops_stopwords():
    text="this is the end"
    EXPECTED=["end"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_stopwords_before_stemming():
    # "ins" isn't a stopword but stems to "in", which is. Removing stopwords first (like bm25s)
    # keeps it; stemming first would wrongly drop it.
    text="the ins and outs"
    EXPECTED=["in","out"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_keeps_duplicates():
    text="dog is dog"
    EXPECTED=["dog","dog"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_stems_variants_together():
    text="Studies STUDY studying"
    EXPECTED=["studi","studi","studi"]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_empty_text():
    text=""
    EXPECTED=[]
    assert Analyzer().analyze(text) == EXPECTED

def test_analyzer_only_stopwords():
    text="the and of"
    EXPECTED=[]
    assert Analyzer().analyze(text) == EXPECTED

@needs_scifact
def test_analyzer_matches_bm25s_on_scifact():
    import bm25s
    import Stemmer
    from harness.datasets import load_dataset
    dataset=load_dataset("scifact")
    analyzer=Analyzer()
    for texts in (list(dataset.corpus.values()),list(dataset.queries.values())):
        expected=bm25s.tokenize(texts,stopwords="en",stemmer=Stemmer.Stemmer("english"),return_ids=False,show_progress=False)
        for text,tokens in zip(texts,expected):
            assert analyzer.analyze(text) == tokens, text[:80]
