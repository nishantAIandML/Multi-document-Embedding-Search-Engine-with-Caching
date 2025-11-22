import os
import hashlib
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from typing import List
from .cache_manager import CacheManager
from .embedder import Embedder
from .utils import clean_text, tokenize


class SearchEngine:
    def __init__(self, docs_folder: str = "data/docs", use_faiss: bool = True):
        self.docs_folder = docs_folder
        self.cache = CacheManager()
        self.embedder = Embedder()
        self.doc_texts = {}  # doc_id -> cleaned text
        self.doc_meta = {}
        self.embeddings = None
        self.doc_ids = []
        self.index = None
        self.use_faiss = use_faiss and FAISS_AVAILABLE

    def load_docs(self):
        # load all .txt files
        files = [f for f in os.listdir(self.docs_folder) if f.endswith('.txt')]
        files.sort()
        for fn in files:
            path = os.path.join(self.docs_folder, fn)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
            cleaned = clean_text(raw)
            doc_id = os.path.splitext(fn)[0]
            self.doc_texts[doc_id] = cleaned
            self.doc_meta[doc_id] = {"filename": fn, "length": len(cleaned.split())}
        return list(self.doc_texts.keys())

    def build_index(self, force_recompute=False):
        # iterate docs, get embeddings (cache-aware)
        texts = []
        doc_ids = []
        for doc_id, text in self.doc_texts.items():
            doc_ids.append(doc_id)
            texts.append(text)

        # check cache per doc
        emb_list = []
        to_compute = []
        to_compute_idx = []
        for i, doc_id in enumerate(doc_ids):
            cache_row = self.cache.get(doc_id)
            current_hash = hashlib.sha256(self.doc_texts[doc_id].encode('utf-8')).hexdigest()
            if cache_row and cache_row['hash'] == current_hash and not force_recompute:
                emb_list.append(cache_row['embedding'])
            else:
                # placeholder, will compute later
                emb_list.append(None)
                to_compute.append(self.doc_texts[doc_id])
                to_compute_idx.append(i)

        if to_compute:
            computed = self.embedder.embed(to_compute)
            for j, idx in enumerate(to_compute_idx):
                emb_list[idx] = computed[j]
                doc_id = doc_ids[idx]
                self.cache.upsert(doc_id, self.doc_meta[doc_id]['filename'], hashlib.sha256(self.doc_texts[doc_id].encode('utf-8')).hexdigest(), computed[j])

        embs = np.vstack([np.array(e, dtype=np.float32) for e in emb_list])
        self.embeddings = embs
        self.doc_ids = doc_ids

        # build FAISS index or fallback
        if self.use_faiss:
            # normalize for cosine similarity (inner product after L2 normalize)
            faiss.normalize_L2(self.embeddings)
            d = self.embeddings.shape[1]
            idx = faiss.IndexFlatIP(d)
            idx.add(self.embeddings)
            self.index = idx
        else:
            # store normalized embeddings for cosine via numpy
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            self.embeddings = self.embeddings / norms
            self.index = None

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        q_emb = self.embedder.embed_one(query).astype('float32')
        # normalize
        if self.use_faiss:
            import faiss
            q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            D, I = self.index.search(np.expand_dims(q, axis=0), top_k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
        else:
            q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            sims = np.dot(self.embeddings, q)
            idxs = list(np.argsort(-sims)[:top_k])
            scores = [float(sims[i]) for i in idxs]

        results = []
        q_tokens = set(tokenize(query))
        for i, s in zip(idxs, scores):
            doc_id = self.doc_ids[i]
            text = self.doc_texts[doc_id]
            
            # Compute token overlap for match info
            doc_tokens = set(tokenize(text))
            overlap = q_tokens & doc_tokens
            overlap_ratio = len(overlap) / len(q_tokens) if q_tokens else 0
            
            # Create preview (first 200 chars)
            preview = text[:200] + "..." if len(text) > 200 else text
            
            results.append({
                "doc_id": doc_id,
                "score": float(s),
                "preview": preview,
                "match_info": {
                    "token_overlap": len(overlap),
                    "overlap_ratio": round(overlap_ratio, 3),
                    "matched_tokens": list(overlap)[:10]  # top 10 tokens
                }
            })
        
        return results