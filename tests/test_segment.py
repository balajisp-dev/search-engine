"""Segment tests: postings encoding, the block layout, writing a segment, and reading it back.

Every test writes into pytest's tmp_path, so nothing touches indexes/.
"""
import json
import math
from pathlib import Path

import pytest

from engine_py.analyzer import Analyzer
from engine_py.index import InvertedIndex
from engine_py.scoring import idf
from engine_py.segment import BLOCK_SIZE, SegmentReader, decode_postings, encode_postings, write_segment

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

def read_json(tmp_path,name):
    return json.loads((tmp_path/"segment_0"/name).read_text())

# 300 docs that all contain "common", so its postings span 3 blocks (128 + 128 + 44).
# Only docs in the first block repeat it (tf=2), so block 0's block_max is higher than
# blocks 1 and 2: a block_max wrongly computed over the whole list would show up.
BIG_DOCS={f"d{i}":"common "*(2 if i<20 else 1)+f"word{i}" for i in range(300)}

def piece(index,term,pos,tf,k1=1.5,b=0.75):
    """One BM25 piece computed from scratch, to check the stored bounds against."""
    L=1-b+b*index.docs_len[pos]/index.avgdl
    return idf(index.N,len(index.postings[term]))*tf*(k1+1)/(tf+k1*L)


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
    assert decode_postings(encode_postings(postings),len(postings),start=0) == postings

def test_postings_round_trip_with_start():
    # a later block: gaps are measured from the previous block's last doc (here 300)
    postings=[(305,1),(310,2)]
    data=encode_postings(postings,prev_pos=300)
    assert data == b"\x05\x01\x05\x02"       # first gap 305-300=5, not 305
    assert decode_postings(data,2,start=300) == postings


# ---- writing ----

def test_write_segment_creates_files(tmp_path):
    write_segment(build(TOY_DOCS),tmp_path/"segment_0")
    EXPECTED={"meta.json","docs.json","terms.json","blocks.json","postings.bin"}
    assert {p.name for p in (tmp_path/"segment_0").iterdir()} == EXPECTED

def test_write_segment_blocks_are_contiguous(tmp_path):
    # terms in sorted order; each term's headers follow the previous term's in blocks.json;
    # each block's bytes start where the previous block ended, and the last block ends
    # exactly at the end of postings.bin (no gaps, nothing missing)
    write_segment(build(TOY_DOCS),tmp_path/"segment_0")
    terms=read_json(tmp_path,"terms.json")
    blocks=read_json(tmp_path,"blocks.json")
    assert list(terms) == sorted(terms)
    expected_header=0
    for df,ub,first_header,num_headers in terms.values():
        assert first_header == expected_header
        expected_header=first_header+num_headers
    assert expected_header == len(blocks)
    expected_offset=0
    for last_doc,offset,length,block_max in blocks:
        assert offset == expected_offset
        expected_offset=offset+length
    assert expected_offset == (tmp_path/"segment_0"/"postings.bin").stat().st_size


# ---- blocks ----

def test_long_list_splits_into_blocks(tmp_path):
    index,_=write_and_open(BIG_DOCS,tmp_path)
    df,ub,first,count=read_json(tmp_path,"terms.json")["common"]
    assert df == 300
    assert count == math.ceil(300/BLOCK_SIZE) == 3

def test_block_last_doc_and_independent_decoding(tmp_path):
    # each header's last_doc is its block's last position, and each block decodes ON ITS OWN
    # from the previous block's last_doc (0 for the first) without decoding earlier blocks
    index,_=write_and_open(BIG_DOCS,tmp_path)
    df,ub,first,count=read_json(tmp_path,"terms.json")["common"]
    headers=read_json(tmp_path,"blocks.json")[first:first+count]
    data=(tmp_path/"segment_0"/"postings.bin").read_bytes()
    postings=index.postings["common"]
    for n,(last_doc,offset,length,_) in enumerate(headers):
        expected=postings[n*BLOCK_SIZE:(n+1)*BLOCK_SIZE]
        start=headers[n-1][0] if n>0 else 0
        assert last_doc == expected[-1][0]
        assert decode_postings(data[offset:offset+length],len(expected),start=start) == expected

def test_block_max_and_ub_match_brute_force(tmp_path):
    index,_=write_and_open(BIG_DOCS,tmp_path)
    terms=read_json(tmp_path,"terms.json")
    blocks=read_json(tmp_path,"blocks.json")
    for term,(df,ub,first,count) in terms.items():
        postings=index.postings[term]
        block_maxes=[h[3] for h in blocks[first:first+count]]
        for n,block_max in enumerate(block_maxes):
            block=postings[n*BLOCK_SIZE:(n+1)*BLOCK_SIZE]
            assert block_max == pytest.approx(max(piece(index,term,pos,tf) for pos,tf in block),abs=1e-9), term
        assert ub == pytest.approx(max(piece(index,term,pos,tf) for pos,tf in postings),abs=1e-9), term
        assert ub == pytest.approx(max(block_maxes),abs=1e-9), term
        assert ub > 0, term

def test_block_max_differs_between_blocks(tmp_path):
    write_and_open(BIG_DOCS,tmp_path)
    df,ub,first,count=read_json(tmp_path,"terms.json")["common"]
    block_maxes=[h[3] for h in read_json(tmp_path,"blocks.json")[first:first+count]]
    assert block_maxes[0] == pytest.approx(ub,abs=1e-9)    # the tf=2 docs are all in block 0
    assert block_maxes[1] < ub and block_maxes[2] < ub

def test_reader_long_list_across_blocks(tmp_path):
    index,reader=write_and_open(BIG_DOCS,tmp_path)
    assert reader.postings("common") == index.postings["common"]


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


# ---- format ----

def test_meta_records_format_and_bm25_params(tmp_path):
    _,reader=write_and_open(TOY_DOCS,tmp_path)
    meta=read_json(tmp_path,"meta.json")
    assert meta["format"] == 2 and meta["block_size"] == BLOCK_SIZE
    assert (reader.k1,reader.b) == (1.5,0.75)

def test_reader_rejects_other_format(tmp_path):
    # an old segment must fail with a clear message, not a confusing missing-file error
    write_and_open(TOY_DOCS,tmp_path)
    meta_path=tmp_path/"segment_0"/"meta.json"
    meta=json.loads(meta_path.read_text())
    del meta["format"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="format 1"):
        SegmentReader(tmp_path/"segment_0")
