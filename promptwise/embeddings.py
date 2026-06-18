"""Module for computing user query embeddings.

Wraps ``sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")``
behind a :class:`QueryEmbedder` class that returns L2-normalised 384-dim
vectors.  The underlying model is loaded once and cached as a singleton.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Singleton model cache
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (lazy-loaded)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# QueryEmbedder
# ---------------------------------------------------------------------------


class QueryEmbedder:
    """Embed user queries into dense vectors for contextual-bandit features.

    The heavy ``SentenceTransformer`` model is loaded only once regardless
    of how many ``QueryEmbedder`` instances are created.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``"all-MiniLM-L6-v2"``.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        # Allow overriding the model name, but default to the singleton.
        if model_name == _MODEL_NAME:
            self._model = _get_model()
        else:
            self._model = SentenceTransformer(model_name)

    # -- public API ---------------------------------------------------------

    def embed(self, query: str) -> np.ndarray:
        """Return a 384-dim L2-normalised embedding for *query*.

        Returns
        -------
        np.ndarray
            Shape ``(384,)``, dtype ``float32``, unit L2 norm.
        """
        vec = self._model.encode(query, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, queries: list[str]) -> np.ndarray:
        """Embed a list of queries in a single forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(len(queries), 384)``, dtype ``float32``,
            each row L2-normalised.
        """
        vecs = self._model.encode(queries, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    # -- convenience --------------------------------------------------------

    @staticmethod
    def dim() -> int:
        """Return the embedding dimensionality (384)."""
        return _EMBEDDING_DIM


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embedder = QueryEmbedder()

    sample_queries = [
        "I want to book an appointment with a cardiologist",
        "What are your clinic hours",
        "Can I cancel my appointment for tomorrow",
        "Do you accept Star Health insurance",
        "This is an emergency, I need help now",
    ]

    print("Embedding sample queries …")
    matrix = embedder.embed_batch(sample_queries)
    print(f"  Shape: {matrix.shape}  dtype: {matrix.dtype}\n")

    # Cosine similarity — since vectors are L2-normalised, cos_sim = dot product.
    cos_sim = matrix @ matrix.T

    # Pretty-print pairwise similarities.
    header = "  " + "".join(f"  Q{i}" for i in range(len(sample_queries)))
    print("Pairwise cosine similarities:")
    print(header)
    for i, q in enumerate(sample_queries):
        row = "  ".join(f"{cos_sim[i, j]:.2f}" for j in range(len(sample_queries)))
        label = f"Q{i}"
        print(f"{label}  {row}   ← {q[:50]}")
