"""Segment tests: postings encoding, writing a segment, and reading it back exactly.

Every test writes into pytest's tmp_path, so nothing touches indexes/.
"""
import json
from pathlib import Path

import pytest

from engine_py.analyzer import Analyzer
from engine_py.index import InvertedIndex
from engine_py.segment import SegmentReader, decode_postings, encode_postings, write_segment

SCIFACT = Path(__file__).resolve().parent.parent / "data" / "scifact"
needs_scifact = pytest.mark.skipif(not SCIFACT.exists(), reason="run scripts/download_beir.py scifact")

TOY_DOCS={"a":"cats eat fish","b":"dogs chase cats","c":"fish swim"}


def build(docs):
    index=InvertedIndex(Analyzer())
    index.build(docs)
    return index

def write_and_open(docs,tmp_path):
    index=build(docs)
    write_segment(index,tmp_path/"segment_0")
    return index,SegmentReader(tmp_path/"segment_0")


# ---- postings encoding ----

def test_encode_postings_stores_gaps():
    # positions 1, 5, 37 -> gaps 1, 4, 32, each followed by tf 1
    postings=[(1,1),(5,1),(37,1)]
    EXPECTED=b"\x01\x01\x04\x01\x20\x01"
    assert encode_postings(postings) == EXPECTED

def test_encode_postings_position_zero_tf_and_big_gap():
    # gap 0, tf 3, then gap 200 -> 2-byte varint [1|1001000][0|0000001], tf 1
    postings=[(0,3),(200,1)]
    EXPECTED=b"\x00\x03\xc8\x01\x01"
    assert encode_postings(postings) == EXPECTED

def test_encode_postings_empty():
    assert encode_postings([]) == b""

@pytest.mark.parametrize("postings", [
    [(1,1),(5,1),(37,1)],
    [(0,3),(200,1)],
    [(5182,7)],
    [(i,i%5+1) for i in range(0,5000,3)],    # many small gaps
    [(0,1),(100000,2),(100001,1)],           # a 3-byte gap next to a 1-byte gap
    [],
])
def test_postings_round_trip(postings):
    assert decode_postings(encode_postings(postings),len(postings)) == postings


# ---- writing ----

def test_write_segment_creates_files(tmp_path):
    write_segment(build(TOY_DOCS),tmp_path/"segment_0")
    EXPECTED={"meta.json","docs.json","terms.json","postings.bin"}
    assert {p.name for p in (tmp_path/"segment_0").iterdir()} == EXPECTED

def test_write_segment_terms_are_sorted_and_contiguous(tmp_path):
    # terms in sorted order, each term's bytes start where the previous one ended, and the
    # last one ends exactly at the end of postings.bin (no gaps, nothing missing)
    write_segment(build(TOY_DOCS),tmp_path/"segment_0")
    terms=json.loads((tmp_path/"segment_0"/"terms.json").read_text())
    assert list(terms) == sorted(terms)
    expected_offset=0
    for df,offset,length in terms.values():
        assert offset == expected_offset
        expected_offset=offset+length
    assert expected_offset == (tmp_path/"segment_0"/"postings.bin").stat().st_size


# ---- reading ----

def test_reader_matches_index_stats(tmp_path):
    index,reader=write_and_open(TOY_DOCS,tmp_path)
    assert reader.N == index.N == 3
    assert reader.avgdl == pytest.approx(index.avgdl,abs=1e-9)
    assert reader.doc_ids == index.doc_ids
    assert reader.docs_len == index.docs_len
    assert reader.num_terms == len(index.postings)

def test_reader_matches_every_term(tmp_path):
    index,reader=write_and_open(TOY_DOCS,tmp_path)
    for term,postings in index.postings.items():
        assert reader.postings(term) == postings, term
        assert reader.df(term) == index.df(term), term

def test_reader_repeated_word_tf(tmp_path):
    _,reader=write_and_open({"d1":"dog is dog","d2":"cats"},tmp_path)
    assert reader.postings("dog") == [(0,2)]

def test_reader_unseen_term(tmp_path):
    _,reader=write_and_open(TOY_DOCS,tmp_path)
    assert reader.df("zebra") == 0
    assert reader.postings("zebra") == []

def test_reader_rewrite_same_folder(tmp_path):
    # writing a new segment over an old one must not leave old terms behind
    write_and_open(TOY_DOCS,tmp_path)
    _,reader=write_and_open({"x":"zebra zebra"},tmp_path)
    assert reader.N == 1
    assert reader.postings("zebra") == [(0,2)]
    assert reader.postings("cat") == []


# ---- SciFact ----

@needs_scifact
def test_scifact_segment_round_trip(tmp_path):
    from harness.datasets import load_dataset
    index,reader=write_and_open(load_dataset("scifact").corpus,tmp_path)
    for term,postings in index.postings.items():
        assert reader.postings(term) == postings, term
    # delta + varint: ~2.17 bytes per posting (vs ~112 bytes as Python objects)
    size=(tmp_path/"segment_0"/"postings.bin").stat().st_size
    assert 1.0e6 < size < 1.2e6
