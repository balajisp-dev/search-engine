"""On-disk segment: an immutable, self-contained index written once.

Layout of a segment folder (format 2):
    meta.json      {"format": 2, "N", "avgdl", "num_terms", "block_size", "k1", "b"}
    docs.json      {"doc_ids": [...], "docs_len": [...]}                          (binary later)
    terms.json     {term: [df, ub, first_header, num_headers]}  in sorted order   (binary later)
    blocks.json    [[last_doc, offset, length, block_max], ...]  one header per block
    postings.bin   every term's postings, back to back, in sorted term order, cut into blocks

Each term's postings are split into blocks of BLOCK_SIZE (the last block holds the rest). A block
is interleaved varints:  gap, tf, gap, tf, ...  where each gap is measured from the previous
posting, and the first gap of a block from the previous block's last_doc (0 for a term's first
block). So any block decodes on its own, given the header before it.

A term's headers are the slice blocks[first_header : first_header + num_headers]. ub is the most
the term adds to any doc's BM25 score; block_max is the same, but only over that block. Both
depend on k1 and b, which is why meta.json records them.

SegmentReader mmaps postings.bin once, so reading a block is a slice of memory (the OS loads and
caches 4 KB pages on demand). PostingsCursor walks one term's postings, decoding a block only when
it lands in it; advance() skips whole blocks using last_doc.
"""
import json
import mmap
from pathlib import Path

from engine_py.index import InvertedIndex
from engine_py.scoring import idf
from engine_py.varint import decode_varint, encode_varint

FORMAT = 2
BLOCK_SIZE = 128
# BM25 parameters the stored bounds (ub, block_max) are computed with; must match the scorer's
K1 = 1.5
B = 0.75

# doc() of a cursor that has run past its last posting: bigger than any real doc position,
# so an exhausted cursor always sorts last (WAND relies on this in Phase 3)
EXHAUSTED = float("inf")


def encode_postings(postings: list[tuple[int, int]], prev_pos: int = 0) -> bytes:
    """[(pos, tf), ...] sorted by pos -> delta + varint bytes (gap, tf interleaved).

    prev_pos is the position the first gap is measured from (the previous block's last_doc).
    """
    buf: bytearray = bytearray()
    for doc_pos, tf in postings:
        encode_varint(doc_pos - prev_pos, buf)
        encode_varint(tf, buf)
        prev_pos = doc_pos
    return bytes(buf)


def decode_postings(data: bytes, df: int, start: int = 0) -> list[tuple[int, int]]:
    """Inverse of encode_postings: df (gap, tf) pairs -> [(pos, tf), ...], gaps counted from start."""
    i = 0
    res = []
    prev = start
    for _ in range(df):
        doc_pos, i = decode_varint(data, i)  # doc_pos is the gap from the previous position
        tf, i = decode_varint(data, i)
        res.append((prev + doc_pos, tf))
        prev += doc_pos
    return res


def block_score(block: list[tuple[int, int]], docs_len: list[int], avgdl: float) -> float:
    """Largest BM25 TF-part in the block (multiply by the term's IDF to get block_max)."""
    tf_part = -1
    for doc_pos, tf in block:
        l = L(docs_len[doc_pos], avgdl)
        tf_part = max(tf_part, (K1 + 1) * tf / (tf + K1 * l))
    return tf_part


def L(doc_len: int, avgdl: float) -> float:
    """BM25 length normalisation: 1 - b + b * dl / avgdl."""
    return 1 - B + (B * doc_len / avgdl)


def write_segment(index: InvertedIndex, path: Path) -> None:
    """Write a built in-memory index to the folder `path` (created if missing).

    For each term (in sorted order), its postings are cut into blocks; each block is encoded from
    the previous block's last_doc and written to postings.bin, and its header
    [last_doc, offset (from f.tell()), length, block_max] is appended to blocks.json. The term's
    entry records its df, ub (the max block_max) and where its headers start and how many there are.
    """
    path.mkdir(parents=True, exist_ok=True)
    terms_json = {}
    blocks_json = []
    with open(path / "postings.bin", mode="wb") as f:
        for term, postings in sorted(index.postings.items()):
            term_level_tf_part = -1
            df = len(postings)
            term_idf = idf(N=index.N, df=df)
            blocks = [postings[i:i + BLOCK_SIZE] for i in range(0, len(postings), BLOCK_SIZE)]
            start = 0
            for block in blocks:
                offset = f.tell()
                encoded_bytes = encode_postings(block, prev_pos=start)
                tf_part = block_score(block, index.docs_len, index.avgdl)
                term_level_tf_part = max(term_level_tf_part, tf_part)
                f.write(encoded_bytes)
                length = len(encoded_bytes)
                start = block[-1][0]
                blocks_json.append([start, offset, length, term_idf * tf_part])
            terms_json[term] = [df, term_idf * term_level_tf_part, len(blocks_json) - len(blocks), len(blocks)]

    with open(path / "terms.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(terms_json))

    with open(path / "blocks.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(blocks_json))

    with open(path / "meta.json", mode="w", encoding="utf-8") as f:
        meta_json = {
            "format": FORMAT,
            "N": index.N,
            "avgdl": index.avgdl,
            "num_terms": len(index.postings),
            "block_size": BLOCK_SIZE,
            "k1": K1,
            "b": B,
        }
        f.write(json.dumps(meta_json))

    with open(path / "docs.json", mode="w", encoding="utf-8") as f:
        docs_json = {
            "doc_ids": index.doc_ids,
            "docs_len": index.docs_len,
        }
        f.write(json.dumps(docs_json))


class SegmentReader:
    """Opens a segment written by write_segment.

    Opening loads meta.json, docs.json, terms.json and blocks.json into memory (no doc text is read
    or analyzed) and mmaps postings.bin. postings(term) decodes all of a term's blocks;
    cursor(term) decodes them lazily. Call close() before the segment's files are rewritten.
    """

    def __init__(self, path: Path) -> None:
        self.docs_json: dict[str, list] = {}
        self.terms_json: dict[str, list] = {}  # term -> [df, ub, first_header, num_headers]
        self.blocks_json: list[list] = []      # [last_doc, offset, length, block_max] per block
        self.path: Path = path
        with open(path / "meta.json", mode="r", encoding="utf-8") as f:
            meta_json = json.load(f)
            if meta_json.get("format") != FORMAT:
                raise ValueError(
                    f"{path} is segment format {meta_json.get('format', 1)}, this code reads format "
                    f"{FORMAT}; rebuild the index")
            self.N = meta_json["N"]
            self.avgdl = meta_json["avgdl"]
            self.num_terms = meta_json["num_terms"]
            self.k1 = meta_json["k1"]
            self.b = meta_json["b"]

        self.f = open(path / "postings.bin", mode="rb")
        self.mmap = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

        with open(path / "docs.json", mode="r", encoding="utf-8") as f:
            self.docs_json = json.load(f)
            # same names as InvertedIndex, so BM25 can use either
            self.doc_ids: list[str] = self.docs_json["doc_ids"]
            self.docs_len: list[int] = self.docs_json["docs_len"]

        with open(path / "terms.json", mode="r", encoding="utf-8") as f:
            self.terms_json = json.load(f)

        with open(path / "blocks.json", mode="r", encoding="utf-8") as f:
            self.blocks_json = json.load(f)

    def close(self) -> None:
        """Release the mmap and the file (the mmap first, since it was built on the file)."""
        self.mmap.close()
        self.f.close()

    def df(self, term: str) -> int:
        """Number of docs containing term (0 if unseen)."""
        return self.terms_json.get(term, [0])[0]

    def postings(self, term: str) -> list[tuple[int, int]]:
        """Decode all of one term's blocks into [(pos, tf), ...] ([] if unseen)."""
        term_data = self.terms_json.get(term)
        if not term_data:
            return []
        df, _, first_header, num_headers = term_data
        blocks = self.blocks_json[first_header:first_header + num_headers]
        decoded_posting = []
        last_block_doc = 0
        for block in blocks:
            last_doc, offset, length, _ = block
            bytes_data = self.mmap[offset:offset + length]
            decoded_block = decode_postings(bytes_data, min(BLOCK_SIZE, df - len(decoded_posting)), last_block_doc)
            decoded_posting.extend(decoded_block)
            last_block_doc = last_doc
        return decoded_posting

    def cursor(self, term: str) -> "PostingsCursor":
        """A cursor over one term's postings that decodes blocks lazily (exhausted at once if unseen)."""
        if not self.terms_json.get(term):
            return PostingsCursor(self, [], 0,term)
        df, _, first_header, num_headers = self.terms_json[term]
        sliced_headers = self.blocks_json[first_header:first_header + num_headers]
        return PostingsCursor(self, sliced_headers, df,term)


class PostingsCursor:
    """Walks one term's postings in doc order, decoding a block only when it lands in it.

    State:
        headers        this term's block headers: [last_doc, offset, length, block_max]
        block          index of the current block in headers
        postings       the current block's decoded [(pos, tf), ...]
        i              index of the current posting inside `postings`
        blocks_decoded how many blocks have been decoded so far (tests use it to prove skipping)

    Decoding block n needs: its bytes (offset, length from its header), its posting count
    (BLOCK_SIZE, or what's left for the last block) and its start value (headers[n - 1]'s last_doc,
    or 0 for n == 0).

    A new cursor is positioned on the term's first posting; past the last one, doc() is EXHAUSTED.
    """

    def __init__(self, reader: SegmentReader, headers: list[list], df: int,term:str | None) -> None:
        self.headers = headers
        self.reader = reader
        self.df = df
        self.block = 0
        self.postings = []
        self.i = 0
        self.blocks_decoded = 0
        self.load_block(0)
        self.term=term
        self.ub=0
        term_meta=self.reader.terms_json.get(term,[])
        if len(term_meta)>0:
            self.ub=term_meta[1]

    def get_term(self):
        return self.term

    def load_block(self, n: int) -> None:
        """Decode block n and point at its first posting; past the last block -> exhausted."""
        if n >= len(self.headers):
            self.postings = []
            self.i = 0
            return
        pre_block_last_doc = self.headers[n - 1][0] if n - 1 >= 0 else 0
        offset = self.headers[n][1]
        length = self.headers[n][2]
        data = self.reader.mmap[offset:offset + length]
        decoded_block = decode_postings(data=data, df=min(BLOCK_SIZE, self.df - BLOCK_SIZE * n), start=pre_block_last_doc)
        self.postings = decoded_block
        self.i = 0
        self.blocks_decoded += 1
        self.block = n

    def doc(self) -> float:
        """Doc position of the current posting, or EXHAUSTED past the end."""
        if self.i >= len(self.postings):
            return EXHAUSTED
        return self.postings[self.i][0]

    def tf(self) -> int:
        """tf of the current posting (-1 when exhausted)."""
        if self.i >= len(self.postings):
            return -1
        return self.postings[self.i][1]

    def next(self) -> None:
        """Move to the next posting; load the next block when this one runs out."""
        if self.i + 1 >= len(self.postings):
            self.block += 1
            self.load_block(self.block)
        else:
            self.i += 1

    def advance(self, target: int) -> None:
        """Move to the first posting with doc >= target. Never moves backwards.

        If target is inside the current block, step forward in memory. Otherwise use the headers'
        last_doc to skip whole blocks (without decoding them) and decode only the block that can
        contain target. Past the last block -> exhausted.
        """
        if self.block >= len(self.headers) or target < self.doc():
            return

        if target <= self.headers[self.block][0]:
            while self.doc() < target:
                self.i += 1
        else:
            next_block = self.block
            while next_block < len(self.headers) and self.headers[next_block][0] < target:
                next_block += 1
            self.load_block(next_block)
            while self.doc() < target:
                self.i += 1
