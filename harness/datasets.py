"""Load BEIR-format datasets from data/<name>/.

Files:
    corpus.jsonl       one JSON object per line: {"_id", "title", "text", ...}
    queries.jsonl      one JSON object per line: {"_id", "text", ...}
    qrels/test.tsv     tab-separated with a header row: query-id  corpus-id  score

All ids are kept as strings: JSON can hand back numeric-looking ids while the
TSV always gives strings, and mixing the two makes qrels lookups silently miss.

    python -m harness.datasets scifact
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    """One BEIR dataset held in memory."""

    name: str
    corpus: dict[str, str]  # doc_id -> "title text"
    queries: dict[str, str]  # query_id -> query text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: grade}


def load_corpus(path: Path) -> dict[str, str]:
    """Read corpus.jsonl into {doc_id: "title text"}."""
    corpus_dict: dict[str, str] = {}
    with open(path, mode="r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            clean_line = line.strip()
            doc = json.loads(clean_line)
            corpus_dict[str(doc["_id"])] = ((doc.get("title") or "") + " " + (doc.get("text") or "")).strip()
    return corpus_dict


def load_queries(path: Path) -> dict[str, str]:
    """Read queries.jsonl into {query_id: text}."""
    queries_dict: dict[str, str] = {}
    with open(path, mode="r", encoding="utf-8") as lines:
        for line in lines:
            doc = json.loads(line)
            queries_dict[str(doc["_id"])] = doc.get("text", "")
    return queries_dict


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read a qrels TSV into {query_id: {doc_id: grade}}."""
    qrels_dict: dict[str, dict[str, int]] = {}
    with open(path, mode="r", encoding="utf-8") as qrels_file:
        lines = csv.DictReader(qrels_file, delimiter="\t")
        for line in lines:
            qrels_dict.setdefault(str(line["query-id"]), {})
            qrels_dict[str(line["query-id"])][str(line["corpus-id"])] = int(line["score"])
    return qrels_dict


def load_dataset(dataset_name: str) -> Dataset:
    """Load corpus, queries and test qrels for data/<dataset_name>/."""
    dataset_path = Path(__file__).resolve().parent.parent / "data" / dataset_name
    corpus_dict = load_corpus(dataset_path / "corpus.jsonl")
    if len(corpus_dict) > 0:
        print(f"corpus data loaded for {dataset_name}")
    queries_dict = load_queries(dataset_path / "queries.jsonl")
    if len(queries_dict) > 0:
        print(f"queries data loaded for {dataset_name}")
    qrels_dict = load_qrels(dataset_path / "qrels" / "test.tsv")
    if len(qrels_dict) > 1:
        print(f"qrels data loaded for {dataset_name}")
    filtered_queries_dict:dict[str,str]={}
    for key,value in queries_dict.items():
          if qrels_dict.get(key):
            filtered_queries_dict[key]=value
    return Dataset(name=dataset_name, corpus=corpus_dict, queries=filtered_queries_dict, qrels=qrels_dict)


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    load_dataset(dataset_name=dataset_name)
