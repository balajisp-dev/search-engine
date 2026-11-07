"""Per-query ranking metrics. YOU implement these.

Shared inputs:
    ranked: doc ids, best first (already cut/filtered by the caller; may be longer than k)
    qrels:  {doc_id: grade} for ONE query; may include grade-0 entries

Conventions (see CLAUDE.md):
    - relevant means grade > 0
    - nDCG gain = grade, discount = log2(rank + 1) with rank starting at 1
    - Recall denominator = number of relevant docs in qrels (not k)
    - A query with no relevant docs: decide what you return and test it
"""


import math


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    rel_docs=0
    total_rel_docs=0
    for doc_id in ranked[:k]:
        if qrels.get(doc_id) and qrels.get(doc_id)>0 :
            rel_docs+=1
    for doc_id,grade in qrels.items():
        if grade > 0 : total_rel_docs+=1
    if total_rel_docs==0:
        return 0.0
    return rel_docs/total_rel_docs



def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    dcg=0
    for (i,doc_id) in enumerate(ranked[:k]):
        if qrels.get(doc_id) and qrels.get(doc_id)>0 :
            dcg+=(qrels.get(doc_id)/math.log2(1+(i+1)))
    idcg=0
    sorted_qrels=dict(sorted(qrels.items(),key= lambda item:item[1],reverse=True))
    for (i,(doc_id,score)) in enumerate(sorted_qrels.items()):
        if int(score)>0 and i<k:
            idcg+=(int(score)/math.log2(1+(i+1)))
    if idcg==0 :return 0.0
    return dcg/idcg


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    for (i,doc_id) in enumerate(ranked[:k]):
        if qrels.get(doc_id) and qrels.get(doc_id)>0:
            return 1/(i+1)
    return 0.0