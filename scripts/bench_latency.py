"""Latency benchmark: the four MyBM25Retriever search modes, side by side.

    python scripts/bench_latency.py --dataset scifact
    python scripts/bench_latency.py --dataset fiqa --ks 10 100

Indexes the corpus once, then opens that one segment with a retriever per mode (taat, daat, wand,
bmw). After one untimed warm-up pass, every query is run by every mode back to back, so machine
noise hits all modes alike. Each mode's top k is checked against taat (they must be identical).

Prints a table and saves results/latency-<dataset>-<timestamp>.json (with the git commit) and a
.png chart next to it: p50 latency, postings scored and blocks decoded per query, by mode and k.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_py.bm25 import MyBM25Retriever  # noqa: E402
from engine_py.segment import SegmentReader  # noqa: E402
from harness.datasets import load_dataset  # noqa: E402
from harness.run_eval import INDEXES_DIR, RESULTS_DIR, git_commit  # noqa: E402

MODES = MyBM25Retriever.MODES


def open_retrievers(dataset) -> tuple[dict[str, MyBM25Retriever], float]:
    """Index once (with the bmw retriever), then point the other modes at the same segment."""
    path = INDEXES_DIR / dataset.name
    builder = MyBM25Retriever(indexes_dir=path, mode="bmw")
    start = time.perf_counter()
    builder.index(dataset.corpus)
    index_s = time.perf_counter() - start
    retrievers = {"bmw": builder}
    for mode in MODES:
        if mode not in retrievers:
            retriever = MyBM25Retriever(indexes_dir=path, mode=mode)
            retriever.segment_reader = SegmentReader(retriever.path)
            retrievers[mode] = retriever
    return retrievers, index_s


def bench(dataset, retrievers: dict[str, MyBM25Retriever], k: int) -> dict[str, dict]:
    queries = list(dataset.queries.values())
    times = {mode: [] for mode in MODES}
    for retriever in retrievers.values():
        retriever.stats = {"postings_scored": 0, "blocks_decoded": 0}
    mismatches = {mode: 0 for mode in MODES}
    for query in queries:
        expected = None
        for mode in MODES:
            start = time.perf_counter()
            result = retrievers[mode].search(query, k)
            times[mode].append(time.perf_counter() - start)
            ids = [doc_id for doc_id, _ in result]
            if expected is None:
                expected = ids          # taat runs first: it is the reference
            elif ids != expected:
                mismatches[mode] += 1
    rows = {}
    n = len(queries)
    for mode in MODES:
        p50, p99 = numpy.percentile(times[mode], [50, 99])
        stats = retrievers[mode].stats
        rows[mode] = {
            "p50_ms": float(p50) * 1000,
            "p99_ms": float(p99) * 1000,
            "mean_ms": float(numpy.mean(times[mode])) * 1000,
            "postings_scored_per_query": stats["postings_scored"] / n,
            "blocks_decoded_per_query": stats["blocks_decoded"] / n,
            "mismatches_vs_taat": mismatches[mode],
        }
    return rows


def print_table(dataset_name: str, k: int, rows: dict[str, dict]) -> None:
    print(f"\n{dataset_name}  k={k}")
    print(f"{'mode':<6}{'p50 ms':>9}{'p99 ms':>9}{'postings/q':>12}{'blocks/q':>10}{'mismatch':>10}")
    for mode, r in rows.items():
        print(f"{mode:<6}{r['p50_ms']:>9.2f}{r['p99_ms']:>9.2f}{r['postings_scored_per_query']:>12.0f}"
              f"{r['blocks_decoded_per_query']:>10.0f}{r['mismatches_vs_taat']:>10}")


def plot(run: dict, path: Path) -> None:
    """Three small multiples (one measure each, never a dual axis): bars per mode, one series per k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series_colors = ["#2a78d6", "#eb6834"]   # categorical slots 1, 2 (validated: CVD dE 24.7)
    text, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    panels = [("p50_ms", "p50 latency (ms)"),
              ("postings_scored_per_query", "Postings scored per query"),
              ("blocks_decoded_per_query", "Blocks decoded per query")]
    ks = [entry["k"] for entry in run["by_k"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=surface)
    width = 0.8 / len(ks)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        for i, entry in enumerate(run["by_k"]):
            xs = [m + (i - (len(ks) - 1) / 2) * width for m in range(len(MODES))]
            values = [entry["modes"][mode][key] for mode in MODES]
            ax.bar(xs, values, width=width - 0.03, color=series_colors[i % 2], label=f"k={entry['k']}",
                   zorder=2)
        ax.set_title(title, color=text, fontsize=11, loc="left")
        ax.set_xticks(range(len(MODES)), MODES, color=text)
        ax.tick_params(axis="y", colors=muted, labelsize=9)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(muted)
    axes[0].legend(frameon=False, labelcolor=text, fontsize=9)
    fig.suptitle(f"{run['dataset']}: BM25 search modes ({run['num_queries']} queries, "
                 f"{run['num_docs']:,} docs, commit {run['git_commit']})", color=text, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 100])
    args = parser.parse_args()

    dataset = load_dataset(dataset_name=args.dataset)
    retrievers, index_s = open_retrievers(dataset)
    for query in dataset.queries.values():      # warm-up: page in the mmap, warm caches
        for retriever in retrievers.values():
            retriever.search(query, args.ks[0])

    now = datetime.now()
    run = {
        "dataset": dataset.name,
        "date": now.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "num_queries": len(dataset.queries),
        "num_docs": len(dataset.corpus),
        "index_s": index_s,
        "by_k": [],
    }
    for k in args.ks:
        rows = bench(dataset, retrievers, k)
        print_table(dataset.name, k, rows)
        run["by_k"].append({"k": k, "modes": rows})

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"latency-{dataset.name}-{now:%Y-%m-%dT%H-%M-%S}"
    with open(stem.with_suffix(".json"), mode="w", encoding="utf-8") as file:
        json.dump(run, file, indent=2)
    plot(run, stem.with_suffix(".png"))
    print(f"\nsaved {stem.with_suffix('.json').relative_to(RESULTS_DIR.parent)} and .png")
