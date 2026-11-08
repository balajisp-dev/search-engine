"""One command -> scorecard.

    python -m harness.run_eval --dataset scifact

Indexes the corpus with each retriever, runs every test query once, scores it with
harness.metrics, prints a scorecard and saves the run to results/<dataset>-<timestamp>.json
(with the date and git commit, plus per-query scores for later comparisons).

    Retriever     nDCG@10  Recall@100  MRR@10  p50 ms  p99 ms  index s
"""
from engine_py.bm25 import MyBM25Retriever
from harness.baselines import DenseRetriever,BM25sRetriever
from harness.datasets import Dataset, load_dataset
from harness.retriever import Retriever
from harness.metrics import ndcg_at_k,recall_at_k,mrr_at_k
from datetime import datetime
from pathlib import Path
import argparse
import json
import subprocess
import time
import numpy

RESULTS_DIR=Path(__file__).resolve().parent.parent/"results"

def run_eval(dataset:Dataset,retriever:Retriever):
    response={}
    start=time.perf_counter()
    retriever.index(dataset.corpus)
    end=time.perf_counter()
    index_time=end-start
    fn_list=[("ndcg@10", ndcg_at_k, 10), ("recall@100", recall_at_k, 100), ("mrr@10", mrr_at_k, 10)]
    avg={"ndcg@10":0,"recall@100":0,"mrr@10":0}
    durations=[]
    for q_id,q in dataset.queries.items():
        response.setdefault(q_id,{})
        search_start=time.perf_counter()
        # 101 = 100 for Recall@100 + 1 spare, since the query itself may come back as a doc
        result = retriever.search(q,101)
        search_end=time.perf_counter()
        doc_ids=[doc_id for doc_id,_ in result if doc_id!=q_id]
        search_duration=search_end-search_start
        response[q_id]["search_ms"]=search_duration*1000
        durations.append(search_duration)
        qrels=dataset.qrels[q_id]
        for metric,fn,k in fn_list:
            response[q_id][metric]=fn(ranked=doc_ids,qrels=qrels,k=k)
            avg[metric]+=response[q_id][metric]
    p50,p99=numpy.percentile(durations,[50,99])
    n=len(dataset.queries)
    return {
        "retriever":retriever.name,
        "ndcg@10":avg["ndcg@10"]/n,
        "recall@100":avg["recall@100"]/n,
        "mrr@10":avg["mrr@10"]/n,
        # numpy floats aren't JSON-serializable, so convert
        "p50_ms":float(p50)*1000,
        "p99_ms":float(p99)*1000,
        "index_s":index_time,
        "per_query":response,
    }

def print_scorecard(dataset_name:str,results:list[dict]):
    print(f"\n{dataset_name}")
    print(f"{'Retriever':<14}{'nDCG@10':>9}{'Recall@100':>12}{'MRR@10':>8}{'p50 ms':>9}{'p99 ms':>9}{'index s':>9}")
    for r in results:
        print(f"{r['retriever']:<14}{r['ndcg@10']:>9.4f}{r['recall@100']:>12.4f}{r['mrr@10']:>8.4f}"
              f"{r['p50_ms']:>9.2f}{r['p99_ms']:>9.2f}{r['index_s']:>9.2f}")

def git_commit()->str:
    repo=RESULTS_DIR.parent
    commit=subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,cwd=repo).stdout.strip() or "unknown"
    # mark runs made with uncommitted changes, so a result can't be mistaken for that commit's code
    dirty=subprocess.run(["git","status","--porcelain"],capture_output=True,text=True,cwd=repo).stdout.strip()
    return f"{commit}-dirty" if dirty else commit

def save_results(dataset:Dataset,results:list[dict])->Path:
    now=datetime.now()
    run={
        "dataset":dataset.name,
        "date":now.isoformat(timespec="seconds"),
        "git_commit":git_commit(),
        "num_queries":len(dataset.queries),
        "num_docs":len(dataset.corpus),
        "results":results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path=RESULTS_DIR/f"{dataset.name}-{now:%Y-%m-%dT%H-%M-%S}.json"
    with open(path,mode="w",encoding="utf-8") as file:
        json.dump(run,file,indent=2)
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    arg=parser.parse_args()
    dataset=load_dataset(dataset_name=arg.dataset)
    results=[run_eval(dataset,BM25sRetriever()),run_eval(dataset,MyBM25Retriever()),run_eval(dataset,DenseRetriever())]
    print_scorecard(dataset.name,results)
    print(f"\nsaved {save_results(dataset,results).relative_to(RESULTS_DIR.parent)}")
