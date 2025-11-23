import torch
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load embedding model (GPU if available)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {device}...")
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def compute_hash(text: str) -> str:
        """
        Generate a SHA256 hash for caching document states.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_embedding(self, text: str):
        """
        Compute embedding vector for a single text string.
        """
        if not text.strip():
            return None
        return self.model.encode(text, show_progress_bar=False).tolist()

    def get_embeddings_batch(self, texts: list):
        """
        Compute embeddings for batch inputs.
        """
        if not texts:
            return []
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_one(self, text: str):
        """
        Alias for get_embedding - compute embedding for a single text.
        Returns numpy array.
        """
        if not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, show_progress_bar=False)

    def embed(self, texts: list):
        """
        Alias for get_embeddings_batch - compute embeddings for batch.
        Returns list of numpy arrays.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return [emb for emb in embeddings]
