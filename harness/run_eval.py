"""One command -> scorecard.

    python -m harness.run_eval --dataset scifact

Target output:
    Retriever  nDCG@10  Recall@100  MRR  p50 ms  p99 ms  index time

TODO:
    - load the dataset, then for each retriever: time index(), time each search()
    - exclude doc_id == query_id before scoring
    - average the per-query metrics from harness.metrics
    - print the scorecard
    - save a JSON run to results/ including the date and git commit
"""
