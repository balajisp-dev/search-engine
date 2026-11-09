"""On-disk segment: an immutable, self-contained index written once.

Layout of a segment folder:
    meta.json      {"N": ..., "avgdl": ..., "num_terms": ...}
    docs.json      {"doc_ids": [...], "docs_len": [...]}                (binary later)
    terms.json     {term: [df, offset, length], ...}  in sorted order   (binary later)
    postings.bin   every term's postings, back to back, in sorted term order

Postings for one term are interleaved varints:  gap, tf, gap, tf, ...
    gap = doc position - previous doc position (the first gap is the position itself)

Reading a term: look up (df, offset, length) in the dictionary, seek(offset), read(length),
then decode df (gap, tf) pairs, turning gaps back into positions with a running sum.
"""
import json
from pathlib import Path

from engine_py.index import InvertedIndex
from engine_py.varint import decode_varint, encode_varint


def encode_postings(postings: list[tuple[int, int]]) -> bytes:
    """[(pos, tf), ...] sorted by pos -> delta + varint bytes (gap, tf interleaved)."""
    prev_pos = 0
    buf: bytearray = bytearray()
    for doc_pos, tf in postings:
        encode_varint(doc_pos - prev_pos, buf)
        encode_varint(tf, buf)
        prev_pos = doc_pos
    return bytes(buf)


def decode_postings(data: bytes, df: int) -> list[tuple[int, int]]:
    """Inverse of encode_postings: df (gap, tf) pairs -> [(pos, tf), ...]."""
    i = 0
    res = []
    prev = 0
    for _ in range(df):
        doc_pos, i = decode_varint(data, i)  # doc_pos is the gap from the previous position
        tf, i = decode_varint(data, i)
        res.append((prev + doc_pos, tf))
        prev += doc_pos
    return res


def write_segment(index: InvertedIndex, path: Path) -> None:
    """Write a built in-memory index to the folder `path` (created if missing).

    postings.bin is written term by term in sorted order; each term's [df, offset, length]
    is collected while writing (offset from f.tell()) and saved to terms.json at the end.
    """
    path.mkdir(parents=True, exist_ok=True)
    terms_json = {}

    with open(path / "postings.bin", mode="wb") as f:
        for term, postings in sorted(index.postings.items()):
            offset = f.tell()
            encoded_bytes = encode_postings(postings)
            f.write(encoded_bytes)
            length = len(encoded_bytes)
            df = len(postings)
            terms_json[term] = [df, offset, length]

    with open(path / "terms.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(terms_json))

    with open(path / "meta.json", mode="w", encoding="utf-8") as f:
        meta_json = {
            "N": index.N,
            "avgdl": index.avgdl,
            "num_terms": len(index.postings),
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

    Opening loads meta.json, docs.json and the term dictionary (terms.json) into memory; no doc
    text is read or analyzed. postings.bin stays on disk: postings(term) reads only that term's
    byte range (seek + read) and decodes it.
    """

    def __init__(self, path: Path) -> None:
        self.docs_json: dict[str, list] = {}
        self.terms_json: dict[str, list[int]] = {}  # term -> [df, offset, length]
        self.path: Path = path
        with open(path / "meta.json", mode="r", encoding="utf-8") as f:
            meta_json = json.load(f)
            self.N = meta_json["N"]
            self.avgdl = meta_json["avgdl"]
            self.num_terms = meta_json["num_terms"]

        with open(path / "docs.json", mode="r", encoding="utf-8") as f:
            self.docs_json = json.load(f)
            # same names as InvertedIndex, so BM25 can use either
            self.doc_ids: list[str] = self.docs_json["doc_ids"]
            self.docs_len: list[int] = self.docs_json["docs_len"]

        with open(path / "terms.json", mode="r", encoding="utf-8") as f:
            self.terms_json = json.load(f)

    def df(self, term: str) -> int:
        """Number of docs containing term (0 if unseen)."""
        return self.terms_json.get(term, [0])[0]

    def postings(self, term: str) -> list[tuple[int, int]]:
        """Read and decode one term's postings from postings.bin ([] if unseen)."""
        term_data = self.terms_json.get(term)
        if not term_data:
            return []
        df, offset, length = term_data
        decoded_posting = []
        with open(self.path / "postings.bin", mode="rb") as f:
            f.seek(offset)
            bytes_data = f.read(length)
            decoded_posting = decode_postings(bytes_data, df)
        return decoded_posting


