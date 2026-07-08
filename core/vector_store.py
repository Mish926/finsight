"""
Vector Store - Hybrid retrieval: TF-IDF (lexical) + LSA (semantic).

TF-IDF alone is pure keyword overlap. If Apple's 10-K says "net sales" and
Amazon's says "total revenue" for the same concept, TF-IDF sees zero overlap
and can miss the relevant chunk entirely -- especially across two different
companies' filings that don't share vocabulary.

The semantic layer here is Latent Semantic Analysis (LSA): TruncatedSVD
applied to the TF-IDF matrix, projecting every chunk into a lower-dimensional
"topic" space where terms that tend to co-occur across the corpus end up
close together -- so "net sales" and "total revenue" can end up near each
other if the surrounding context is similar, even with zero literal word
overlap. This is the same idea as sentence embeddings, but built entirely
from scikit-learn (already a dependency) with no torch/transformers, no
model download, and no platform-specific wheel issues (e.g. PyTorch dropped
Intel macOS support after 2.2.2 -- LSA has none of that baggage).

Each chunk gets two scores per query:
  - Lexical score  (TF-IDF cosine similarity)  -- exact terms, tickers, $ figures
  - Semantic score (LSA cosine similarity)      -- co-occurrence-based concept match
They're combined into one hybrid ranking score via a weighted sum.

MEMORY: the TF-IDF matrix is kept SPARSE throughout (never converted to a
dense numpy array). A TF-IDF matrix for text like financial filings is
typically 95%+ zeros (each chunk only uses a tiny fraction of an 8000-word
vocabulary), so densifying it means storing a huge number of zeros
explicitly. For a large multi-document corpus (e.g. a 400+ page 10-K
producing 4000+ chunks), a dense float32 matrix at max_features=8000 can
run into hundreds of MB -- more than the entire memory budget on a
resource-constrained host (this was found and fixed after a real
out-of-memory crash on a 512MB deployment). scikit-learn's
cosine_similarity() and TruncatedSVD both operate directly and efficiently
on sparse matrices, so there's no need to ever densify the TF-IDF side at
all -- only the (much smaller, fixed-width) LSA output vectors are dense.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize

from core.document_processor import Chunk

# Cap on LSA components. Actual component count used is also bounded by
# (n_chunks - 1) and (n_features - 1), since TruncatedSVD requires
# n_components < min(n_samples, n_features).
MAX_SVD_COMPONENTS = 100


class VectorStore:
    def __init__(self, index_dir: str = "data/index", semantic_weight: float = 0.4):
        """
        semantic_weight: 0.0 = pure TF-IDF, 1.0 = pure LSA, 0.4 = default hybrid
        (leans lexical, since exact dollar figures/tickers matter a lot for
        financial documents -- pure semantic similarity can blur precise
        numbers together). Tune based on observed query results.
        """
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
        self.vectors = None  # scipy sparse matrix, NOT a dense numpy array
        self.chunks: List[Chunk] = []
        self.fitted = False

        self.semantic_weight = semantic_weight
        self.svd = None
        self.semantic_vectors = None  # dense (LSA output is low-dimensional, fine)
        print("Vector store initialized (hybrid TF-IDF + LSA semantic mode)")

    def _normalize_dense(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v / norms

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)

        # chunk_id is assigned locally (starting at 0) by DocumentProcessor
        # for EACH document processed, so uploading a second document
        # produces chunk_ids that collide with the first (both start at 0).
        # Reassigning globally-unique ids here, across the whole store, is
        # what makes cross-document deduplication in the retriever safe --
        # without this, two different documents' chunks with the same
        # number could get silently treated as duplicates of each other.
        for i, chunk in enumerate(self.chunks):
            chunk.chunk_id = i

        texts = [c.text for c in self.chunks]

        print(f"Embedding {len(texts)} chunks (TF-IDF)...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)  # sparse, stays sparse
        self.vectors = sk_normalize(tfidf_matrix, norm='l2', axis=1).astype(np.float32)

        n_components = min(MAX_SVD_COMPONENTS, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if n_components >= 2:
            print(f"Fitting LSA (semantic) with {n_components} components...")
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            semantic_raw = self.svd.fit_transform(tfidf_matrix).astype(np.float32)
            self.semantic_vectors = self._normalize_dense(semantic_raw)
        else:
            # Too few chunks/terms for a meaningful LSA space yet -- falls
            # back to lexical-only until enough content is indexed.
            self.svd = None
            self.semantic_vectors = None

        self.fitted = True
        print(f"Index now contains {len(self.chunks)} chunks.")

    def remove_document(self, source: str) -> int:
        """Remove all chunks belonging to one document, then rebuild the
        TF-IDF/LSA index from whatever remains -- other documents in the
        index are untouched. Returns the number of chunks removed.

        This is what lets a non-technical user clean up the shared index
        through the UI (delete one bad upload, or just tidy up) without
        needing to `rm` files and restart the server themselves."""
        before = len(self.chunks)
        remaining = [c for c in self.chunks if c.source != source]
        removed_count = before - len(remaining)
        if removed_count == 0:
            return 0

        self.chunks = []
        self.vectors = None
        self.svd = None
        self.semantic_vectors = None
        self.fitted = False

        if remaining:
            self.add_chunks(remaining)
        else:
            print("Index now contains 0 chunks.")

        return removed_count

    def search(self, query: str, top_k: int = 5, source: str = None) -> List[Tuple[Chunk, float]]:
        if not self.fitted or self.vectors is None:
            raise RuntimeError("Vector store is empty.")

        q_tfidf_sparse = self.vectorizer.transform([query])  # sparse
        q_tfidf = sk_normalize(q_tfidf_sparse, norm='l2', axis=1)
        # cosine_similarity handles sparse x sparse natively -- no densifying needed
        lexical_scores = cosine_similarity(q_tfidf, self.vectors)[0]

        if self.svd is not None and self.semantic_vectors is not None:
            q_semantic = self._normalize_dense(
                self.svd.transform(q_tfidf_sparse).astype(np.float32)
            )
            semantic_scores = cosine_similarity(q_semantic, self.semantic_vectors)[0]
            scores = (1 - self.semantic_weight) * lexical_scores + self.semantic_weight * semantic_scores
        else:
            scores = lexical_scores

        if source is not None:
            mask = np.array([c.source == source for c in self.chunks])
            scores = np.where(mask, scores, -1.0)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, name: str = "finsight") -> None:
        with open(self.index_dir / f"{name}.pkl", "wb") as f:
            pickle.dump({
                "vectors": self.vectors,
                "semantic_vectors": self.semantic_vectors,
                "svd": self.svd,
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
        if isinstance(self.vectors, np.ndarray):
            # Backward compatible with indexes saved before the sparse-matrix
            # fix -- convert an old dense index to sparse on load instead of
            # requiring everyone to re-upload.
            self.vectors = sparse.csr_matrix(self.vectors)
        # Backward compatible with indexes saved before hybrid search existed.
        self.semantic_vectors = data.get("semantic_vectors")
        self.svd = data.get("svd")
        self.chunks = data["chunks"]
        self.vectorizer = data["vectorizer"]
        self.fitted = data["fitted"]
        print(f"Loaded index: {name} ({len(self.chunks)} chunks)")
        if self.svd is None:
            print("  Note: this index predates LSA semantic search. Re-upload "
                  "a document (or clear the index) to enable hybrid retrieval.")
        return True

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.chunks) == 0

    def stats(self) -> dict:
        sources = list({c.source for c in self.chunks})
        return {
            "total_chunks": len(self.chunks),
            "documents": sources,
            "num_documents": len(sources),
            "retrieval_mode": "hybrid (TF-IDF + LSA semantic)" if self.svd is not None else "TF-IDF only"
        }
