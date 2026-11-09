"""PostingsCursor tests: walking with next(), jumping with advance(), and proof that blocks are skipped.

Every test writes into pytest's tmp_path, so nothing touches indexes/.
"""
import bisect
import random
from pathlib import Path

import pytest

from engine_py.analyzer import Analyzer
from engine_py.index import InvertedIndex
from engine_py.segment import BLOCK_SIZE, EXHAUSTED, SegmentReader, write_segment

SCIFACT = Path(__file__).resolve().parent.parent / "data" / "scifact"
needs_scifact = pytest.mark.skipif(not SCIFACT.exists(), reason="run scripts/download_beir.py scifact")

# 300 docs: "common" is in every doc, so its positions are 0..299 in 3 blocks with last_docs
# 127, 255, 299. "rare" is in every 3rd doc (100 postings, 1 block). Docs 0..19 repeat "common"
# so some tfs are 2.
DOCS={f"d{i}":"common "*(2 if i<20 else 1)+f"word{i}"+(" rare" if i%3==0 else "") for i in range(300)}


@pytest.fixture
def seg(tmp_path):
    index=InvertedIndex(Analyzer())
    index.build(DOCS)
    write_segment(index,tmp_path/"segment_0")
    return index,SegmentReader(tmp_path/"segment_0")

def walk(cursor):
    """Everything a cursor yields with next() until it is exhausted."""
    out=[]
    while cursor.doc()!=EXHAUSTED:
        out.append((cursor.doc(),cursor.tf()))
        cursor.next()
    return out

def first_at_or_after(postings,target):
    """What advance(target) should land on: the first doc >= target, or EXHAUSTED."""
    docs=[doc for doc,_ in postings]
    k=bisect.bisect_left(docs,target)
    return docs[k] if k<len(docs) else EXHAUSTED


# ---- starting and walking ----

def test_cursor_starts_on_first_posting(seg):
    index,reader=seg
    cursor=reader.cursor("common")
    assert (cursor.doc(),cursor.tf()) == index.postings["common"][0] == (0,2)
    assert cursor.blocks_decoded == 1

def test_cursor_next_walks_every_posting_across_blocks(seg):
    index,reader=seg
    assert walk(reader.cursor("common")) == index.postings["common"]   # 3 blocks
    assert walk(reader.cursor("rare")) == index.postings["rare"]       # 1 block

def test_cursor_stays_exhausted(seg):
    _,reader=seg
    cursor=reader.cursor("rare")
    walk(cursor)
    cursor.next()
    cursor.advance(5)
    assert cursor.doc() == EXHAUSTED

def test_cursor_unseen_term(seg):
    _,reader=seg
    cursor=reader.cursor("zebra")
    assert cursor.doc() == EXHAUSTED
    assert cursor.blocks_decoded == 0


# ---- advance ----

@pytest.mark.parametrize("target,EXPECTED", [
    (0,   0),           # already there: doesn't move
    (50,  50),          # inside block 0
    (127, 127),         # exactly block 0's last_doc
    (128, 128),         # first doc of block 1
    (255, 255),         # exactly block 1's last_doc
    (299, 299),         # the very last posting
    (300, EXHAUSTED),   # past the end
])
def test_cursor_advance_targets(seg,target,EXPECTED):
    _,reader=seg
    cursor=reader.cursor("common")
    cursor.advance(target)
    assert cursor.doc() == EXPECTED

def test_cursor_advance_between_postings(seg):
    # "rare" is at 0, 3, 6, ...: target 4 is not a posting, so land on the next one (6)
    _,reader=seg
    cursor=reader.cursor("rare")
    cursor.advance(4)
    assert cursor.doc() == 6

def test_cursor_advance_never_moves_backwards(seg):
    _,reader=seg
    cursor=reader.cursor("common")
    cursor.advance(200)
    cursor.advance(50)
    assert cursor.doc() == 200

def test_cursor_mixing_next_and_advance(seg):
    # after a jump, next() must continue from the new block, in order, to the end
    index,reader=seg
    cursor=reader.cursor("common")
    cursor.next()
    cursor.advance(200)
    cursor.next()
    assert cursor.doc() == 201
    assert walk(cursor) == [p for p in index.postings["common"] if p[0]>=201]

def test_cursor_random_advances_match_brute_force(seg):
    index,reader=seg
    rng=random.Random(7)
    for term in ("common","rare"):
        cursor=reader.cursor(term)
        for target in sorted(rng.sample(range(0,310),40)):
            cursor.advance(target)
            assert cursor.doc() == first_at_or_after(index.postings[term],target), (term,target)


# ---- skipping really happens ----

def test_cursor_advance_skips_blocks(seg):
    # jumping from block 0 straight into block 2 must not decode block 1
    _,reader=seg
    cursor=reader.cursor("common")
    cursor.advance(299)
    assert cursor.doc() == 299
    assert cursor.blocks_decoded == 2

def test_cursor_advance_past_end_decodes_nothing_new(seg):
    # the headers alone show 300 is past the last block
    _,reader=seg
    cursor=reader.cursor("common")
    cursor.advance(300)
    assert cursor.blocks_decoded == 1

def test_cursor_advance_inside_block_decodes_nothing_new(seg):
    _,reader=seg
    cursor=reader.cursor("common")
    cursor.advance(100)
    assert cursor.blocks_decoded == 1


# ---- SciFact ----

@needs_scifact
def test_cursor_scifact_long_list(tmp_path):
    from harness.datasets import load_dataset
    index=InvertedIndex(Analyzer())
    index.build(load_dataset("scifact").corpus)
    write_segment(index,tmp_path/"segment_0")
    reader=SegmentReader(tmp_path/"segment_0")
    postings=index.postings["we"]                       # 3,439 postings
    assert len(postings) > 20*BLOCK_SIZE
    assert walk(reader.cursor("we")) == postings
    cursor=reader.cursor("we")
    cursor.advance(postings[-1][0])
    assert cursor.doc() == postings[-1][0]
    assert cursor.blocks_decoded == 2                   # first and last of 27 blocks
