# Hybrid Search Engine — project context

## Goal
Portfolio + deep-learning project: a hybrid search engine built from scratch.
Keyword search (BM25 over my own inverted index) + vector search (my own HNSW),
fused into one ranked list. Judged by benchmarks against standard baselines.

## Custom features (what makes it mine)
- Glass-box "explain" view: per-term BM25 contributions, vector similarity, fusion math per result
- Tamil–English code-mixed search with transliteration ("vanakkam" matches "வணக்கம்")
- MCP server so AI agents can query it (search, explain, index_document)

## Stack
Core engine in Rust (from Phase 3.5), Python bindings via PyO3, FastAPI service on top.
Phases 0–3 are pure Python (venv on Python 3.12). ~8–10 hrs/week, ~14 weeks.

## How we work
Phase by phase. For each phase: LEARN the concepts first (explain simply,
give a quick self-check), THEN I implement. I write the core algorithms myself
(metrics, BM25, HNSW, etc.) — Claude scaffolds plumbing, reviews my code,
and explains. Don't write the learning parts for me unless I ask.

## Phases
0. Eval harness first (week 1) ✅
1. In-memory inverted index + BM25 (weeks 2–3): match baseline nDCG@10 within ~1% ✅
2. On-disk segments, postings compression (delta + varint, blocks), skip lists, mmap (weeks 4–5) ✅
3. WAND → Block-Max WAND top-k; latency chart (week 6) ✅
3.5. Rust core (weeks 7–8, hard 2-week time box) ← CURRENT
   Port varint, segment reader, cursor and daat/wand/bmw to Rust (PyO3 + maturin); Python keeps
   the writer, analyzer, taat (as the reference) and the harness. One Rust call per query.
   Done when: all-modes test passes against Rust, nDCG@10 = 0.6863, latency chart re-run.
4. HNSW from scratch, in Rust; recall-vs-latency vs FAISS (weeks 9–10)
5. Hybrid: RRF + weighted fusion (week 11)
6. Deletes (tombstones), updates, tiered merge policy (week 12)
7. Explain API + glass-box UI + Tamil-English analyzer (week 13)
8. FastAPI + MCP server, benchmark report, demo video, blog post (week 14)

Scope guardrails: single node only, no fancy query syntax. Never cut the eval harness or the write-up.

## Phase 0 conventions
- IDs are always strings (doc ids, query ids).
- Metrics: relevant = grade > 0; nDCG gain = grade, discount log2(rank+1);
  Recall denominator = number of relevant docs. Exclude doc_id == query_id (BEIR convention).
- Datasets: SciFact first, then NFCorpus, FiQA. Expected BM25 on SciFact ≈ 0.66–0.69 nDCG@10.
- Phase 0 done when: one-command scorecard works; my metrics match a library to 3 decimals;
  both baselines use the Retriever interface; results saved as JSON.

## Commands
- Setup: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[baselines,dev]"`
- Download: `python scripts/download_beir.py scifact`
- Eval: `python -m harness.run_eval --dataset scifact`
- Latency by search mode (taat/daat/wand/bmw) + chart: `python scripts/bench_latency.py --dataset scifact`
- Tests: `pytest`
