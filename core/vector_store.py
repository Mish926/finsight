"""
Vector Store - improved TF-IDF with better financial keyword matching.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.document_processor import Chunk


class VectorStore:
    def __init__(self, index_dir: str = "data/index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.vectors = None
        self.chunks: List[Chunk] = []
        self.fitted = False
        print("Vector store initialized (TF-IDF mode)")

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v / norms

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        texts = [c.text for c in self.chunks]
        print(f"Embedding {len(texts)} chunks...")
        self.vectors = self._normalize(
            self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        )
        self.fitted = True
        print(f"Index now contains {len(self.chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.fitted or self.vectors is None:
            raise RuntimeError("Vector store is empty.")
        q_vec = self._normalize(
            self.vectorizer.transform([query]).toarray().astype(np.float32)
        )
        scores = cosine_similarity(q_vec, self.vectors)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, name: str = "finsight") -> None:
        with open(self.index_dir / f"{name}.pkl", "wb") as f:
            pickle.dump({
                "vectors": self.vectors,
                "chunks": self.chunks,
                "vectorizer": self.vectorizer,
                "fitted": self.fitted
            }, f)
        print(f"Saved index: {name} ({len(self.chunks)} chunks)")

    def load(self, name: str = "finsight") -> bool:
        path = self.index_dir / f"{name}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectors = data["vectors"]
        self.chunks = data["chunks"]
        self.vectorizer = data["vectorizer"]
        self.fitted = data["fitted"]
        print(f"Loaded index: {name} ({len(self.chunks)} chunks)")
        return True

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.chunks) == 0

    def stats(self) -> dict:
        sources = list({c.source for c in self.chunks})
        return {
            "total_chunks": len(self.chunks),
            "documents": sources,
            "num_documents": len(sources)
        }
