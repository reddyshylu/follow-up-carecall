"""Step 3: find the chunks that answer a question, for ONE patient only.

Hybrid search, as taught in the course:
  - meaning search: Pinecone, filtered by metadata to the selected patient
  - keyword search: BM25 over the same patient's chunks
  - fusion: weighted Reciprocal Rank Fusion (the same method as LangChain's EnsembleRetriever)
  - optional reranker: a small cross-encoder re-scores the fused list
The patient's summary is always included first, because most follow-up call
questions (allergies, active medicines, last visit) are answered there.
"""
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from common import CHUNKS, MANIFEST, NAMESPACE, QUERY_PREFIX, AppError, embedder, read_jsonl, settings

TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenize(text):
    return TOKEN.findall(text.lower())


class Retriever:
    def __init__(self):
        if not CHUNKS.exists() or not MANIFEST.exists():
            raise AppError("Run build_documents.py and ingest.py --upload first.")
        import json
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.chunks = {c["chunk_id"]: c for c in read_jsonl(CHUNKS)}
        self.by_patient = defaultdict(list)
        for c in self.chunks.values():
            self.by_patient[c["metadata"]["patient_id"]].append(c)
        self._bm25 = {}
        self._index = None
        self._reranker = None

    # ---------- the two searches
    def index(self):
        if self._index is None:
            config = settings()
            if not config["PINECONE_API_KEY"] or not config["PINECONE_INDEX"]:
                raise AppError("Add PINECONE_API_KEY and PINECONE_INDEX to .env first.")
            if not self.manifest.get("uploaded"):
                raise AppError("Chunks are not uploaded yet. Run: python ingest.py --upload")
            from pinecone import Pinecone
            self._index = Pinecone(api_key=config["PINECONE_API_KEY"]).Index(config["PINECONE_INDEX"])
        return self._index

    def semantic(self, question, patient_id, k=10):
        vector = embedder().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
        result = self.index().query(vector=vector, top_k=k, namespace=NAMESPACE, include_metadata=True,
                                    filter={"patient_id": {"$eq": patient_id}})
        hits = [m.id for m in result.matches if m.id in self.chunks]
        # Safety check: never accept a chunk that belongs to another patient.
        return [h for h in hits if self.chunks[h]["metadata"]["patient_id"] == patient_id]

    def keyword(self, question, patient_id, k=10):
        pool = self.by_patient.get(patient_id, [])
        if not pool:
            return []
        if patient_id not in self._bm25:
            self._bm25[patient_id] = BM25Okapi([tokenize(c["text"]) for c in pool])
        scores = self._bm25[patient_id].get_scores(tokenize(question))
        ranked = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        return [pool[i]["chunk_id"] for i in ranked[:k] if scores[i] > 0]

    # ---------- fusion and reranking
    @staticmethod
    def fuse(lists, weights):
        score = defaultdict(float)
        for hits, w in zip(lists, weights):
            for rank, cid in enumerate(hits):
                score[cid] += w / (RRF_K + rank + 1)
        return sorted(score, key=score.get, reverse=True)

    def rerank(self, question, ids):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = self._reranker.predict([(question, self.chunks[i]["text"]) for i in ids])
        return [i for _, i in sorted(zip(scores, ids), key=lambda x: x[0], reverse=True)]

    def search(self, question, patient_id, strategy="Hybrid", k=6, rerank=False):
        if patient_id not in self.by_patient:
            raise AppError("This patient is not in the 200-patient dataset.")
        trace = {}
        if strategy in ("Hybrid", "Semantic"):
            trace["semantic"] = self.semantic(question, patient_id)
        if strategy in ("Hybrid", "Keyword"):
            trace["keyword"] = self.keyword(question, patient_id)
        lists = [trace[s] for s in ("semantic", "keyword") if s in trace]
        fused = self.fuse(lists, [0.5] * len(lists))
        if rerank and fused:
            fused = self.rerank(question, fused[:20])
        summary = [c["chunk_id"] for c in self.by_patient[patient_id] if c["metadata"]["doc_type"] == "patient_summary"]
        picked = summary + [cid for cid in fused if cid not in summary][:k]
        trace["final"] = picked
        evidence = []
        for n, cid in enumerate(picked, 1):
            c = self.chunks[cid]
            evidence.append({"label": f"S{n}", "chunk_id": cid, "text": c["text"], **c["metadata"],
                             "sources": c["sources"]})
        return evidence, trace
