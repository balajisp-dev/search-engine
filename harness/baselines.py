"""Reference retrievers to compare against. Both satisfy the Retriever Protocol.

    BM25sRetriever   bm25s with English stopwords + Snowball stemming (k1=1.5, b=0.75)
    DenseRetriever   all-MiniLM-L6-v2 embeddings, normalized, exact cosine search via FAISS IndexFlatIP

Both libraries address docs by position, so each retriever keeps doc_ids to map positions back to ids.
"""
import bm25s,Stemmer
from sentence_transformers import SentenceTransformer
import faiss

class BM25sRetriever:
    name:str="bm25s"
    def __init__(self,language="english") -> None:
        self.stemmer=Stemmer.Stemmer(language)

       
    def index(self,docs:dict[str,str])->None:
        self.doc_ids=list(docs)
        texts=list(docs.values())
        tokens=bm25s.tokenize(texts=texts,stopwords="en",stemmer=self.stemmer,show_progress=False)
        self.model=bm25s.BM25()
        self.model.index(corpus=tokens,show_progress=False)
    
    def search(self,query: str,k: int)->list[tuple[str,float]]:
        tokens=bm25s.tokenize(texts=[query],stopwords="en",stemmer=self.stemmer,show_progress=False)
        positions,scores=self.model.retrieve(query_tokens=tokens,k=min(k,len(self.doc_ids)),show_progress=False)
        positions,scores=positions[0],scores[0]
        zipped=zip(positions,scores)
        return [(self.doc_ids[position],float(score)) for (position,score) in zipped]

class DenseRetriever:
    name:str="minilm-faiss"
    def __init__(self):
        self.model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def index(self,docs:dict[str,str])->None:
        self.doc_ids=list(docs)
        texts=list(docs.values())
        emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        self._index=faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
    
    def search(self,query: str,k: int)->list[tuple[str,float]]:
        query_emb=self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores,positions=self._index.search(query_emb,k)
        positions,scores=positions[0],scores[0]
        zipped=zip(positions,scores)
        # FAISS pads with -1 when k > number of docs; doc_ids[-1] would silently return the last doc
        return [(self.doc_ids[position],float(score)) for (position,score) in zipped if position!=-1]


       
        

     