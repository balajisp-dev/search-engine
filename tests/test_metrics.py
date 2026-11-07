import enum
from numpy import full
import pytest
from harness.metrics import mrr_at_k, ndcg_at_k, recall_at_k
import random
from ranx import Qrels,Run,evaluate
def test_recall_hand_example():
    ranked=["d3","d1","d7","d2","d9"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=2
    EXPECTED=1/3
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_recall_hand_example_deeper():
    ranked=["d3","d1","d7","d2","d9"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=5
    EXPECTED=2/3
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_recall_only_relevant_docs():
    ranked=["d1","d5","d2"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=3
    EXPECTED=1.0
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_recall_only_irrelevant_docs():
    ranked=["d9","d11","d14"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=3
    EXPECTED=0.0
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_recall_contain_doc_with_grade_zero():
    ranked=["d9","d1"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=2
    EXPECTED=1/3
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_recall_k_greater_than_n():
    ranked=["d3","d1","d7","d2","d9"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=200
    EXPECTED=2/3
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_recall_no_relevant_docs_in_qrels():
    ranked=["d3","d1","d7","d2","d9"]
    qrels={"d1":0}
    k=200
    EXPECTED=0.0
    assert recall_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr():
    ranked=["d3","d1","d7","d2","d9"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}	
    k=10
    EXPECTED=0.5
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_first_is_relevant():
    ranked=["d1","x","y"]
    qrels={"d1":1}
    k=10
    EXPECTED=1.0
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_only_first_relevant_doc():
    ranked=["x","d1","d2"]
    qrels={"d1":1,"d2":1}
    k=10
    EXPECTED=0.5
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_only_relevant_at_k():
    ranked=["x","y","d1"]
    qrels={"d1":1}
    k=3
    EXPECTED=1/3
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_only_relevant_just_past_k():
    ranked=["x","y","d1"]
    qrels={"d1":1}
    k=2
    EXPECTED=0.0
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_grade_zero_skipped():
    ranked=["d9","d1"]
    qrels={"d9":0,"d1":1}
    k=3
    EXPECTED=0.5
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_empty_ranked_list():
    ranked=[]
    qrels={"d1":0}
    k=3
    EXPECTED=0.0
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_mrr_no_relevant_doc_in_qrels():
    ranked=["d3","d1","d7","d2","d9"]
    qrels={"d11":2,"d21":1}	
    k=10
    EXPECTED=0.0
    assert mrr_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg():
    ranked=["d3","d1","d7","d2","d9"]	
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=5
    EXPECTED=0.5406
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_perfect_ranking():
    ranked=["d1","d2","d5"]
    qrels={"d1":2,"d2":1,"d5":1,"d9":0}
    k=10
    EXPECTED=1.0
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_grade_order():
    ranked=["d2","d1"]
    qrels={"d1":2,"d2":1}
    k=10
    EXPECTED=0.8597
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_one_relevant_at_rank_two():
    ranked=["x","d1"]
    qrels={"d1":1}
    k=10
    EXPECTED=0.6309
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_missed_relevant_doc():
    ranked=["d1"]
    qrels={"d1":1,"d2":1}
    k=10
    EXPECTED=0.6131
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_idcg_cut_at_k():
    ranked=["d1","x"]
    qrels={"d1":1,"d2":1,"d3":1}
    k=1
    EXPECTED=1.0
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_relevant_just_past_k():
    ranked=["x","d1"]
    qrels={"d1":1}
    k=1
    EXPECTED=0.0
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_grade_zero_adds_nothing():
    ranked=["d9","d1"]
    qrels={"d9":0,"d1":1}
    k=10
    EXPECTED=0.6309
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_linear_gain():
    ranked=["d2","d1"]
    qrels={"d1":3,"d2":1}
    k=10
    EXPECTED=0.7967
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_empty_ranked_list():
    ranked=[]
    qrels={"d1":1}
    k=10
    EXPECTED=0.0
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def test_ndcg_no_relevant_doc_in_qrels():
    ranked=["d1"]
    qrels={"d1":0}
    k=10
    EXPECTED=0.0
    assert ndcg_at_k(ranked=ranked,qrels=qrels,k=k) == pytest.approx(EXPECTED,abs=1e-4)

def generate_docs(rng:random.Random):
    length=rng.randint(5,15)
    docs=[f"{i}" for i in range(31)]
    docs=rng.sample(docs,k=length)
    return docs

def generate_qrels(rng:random.Random):
    full_qrels={}
    for i in range(200):
        docs=generate_docs(rng)
        doc_scores={}
        for doc_id in docs:
            doc_scores[doc_id]=rng.randint(0,3)
        full_qrels[f"q{i}"]=doc_scores
    return full_qrels

def generate_ranked_list(rng:random.Random):
    full_ranked_list={}
    for query in range(200):
        docs=generate_docs(rng)
        ranked_list={}
        for i,doc_id in enumerate(docs):
            ranked_list[doc_id]=len(docs)-i
        full_ranked_list[f"q{query}"]=ranked_list
    return full_ranked_list
    
@pytest.mark.parametrize("k", [1, 5, 10, 100])
@pytest.mark.parametrize("fn,name", [(recall_at_k,"recall"),(ndcg_at_k,"ndcg"),(mrr_at_k,"mrr")])
def test_with_ranx(fn,name,k):
    rng=random.Random(42)
    qrels=generate_qrels(rng)
    ranked_list=generate_ranked_list(rng)
    run=Run(run=ranked_list)
    evaluate(qrels=Qrels(qrels=qrels),run=run,metrics=f"{name}@{k}")
    # ranx sorts query ids ("q1","q10","q2"...), so look its scores up by id, not by position
    ranx_response=run.scores[f"{name}@{k}"]
    our_response={q:fn(ranked=list(ranked_list[q]),qrels=qrels[q],k=k) for q in qrels}
    for q in qrels:
        assert our_response[q] == pytest.approx(ranx_response[q],abs=1e-3), q
        






