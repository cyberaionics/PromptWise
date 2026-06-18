"""Tests for promptwise.embeddings module."""

import numpy as np
import pytest

from promptwise.embeddings import QueryEmbedder, _get_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> QueryEmbedder:
    """Module-scoped embedder so the model is loaded only once per run."""
    return QueryEmbedder()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_embeddings_import():
    """Original smoke test: module is importable."""
    from promptwise import embeddings
    assert embeddings is not None


# ---------------------------------------------------------------------------
# Singleton caching
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_model_is_singleton(self):
        """Two QueryEmbedder instances share the same underlying model."""
        a = QueryEmbedder()
        b = QueryEmbedder()
        assert a._model is b._model

    def test_get_model_returns_same_object(self):
        m1 = _get_model()
        m2 = _get_model()
        assert m1 is m2


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_returns_ndarray(self, embedder: QueryEmbedder):
        vec = embedder.embed("book an appointment")
        assert isinstance(vec, np.ndarray)

    def test_shape_384(self, embedder: QueryEmbedder):
        vec = embedder.embed("book an appointment")
        assert vec.shape == (384,)

    def test_dtype_float32(self, embedder: QueryEmbedder):
        vec = embedder.embed("book an appointment")
        assert vec.dtype == np.float32

    def test_l2_normalised(self, embedder: QueryEmbedder):
        vec = embedder.embed("check doctor availability")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"L2 norm is {norm}, expected ~1.0"

    def test_deterministic(self, embedder: QueryEmbedder):
        q = "I need a dermatology appointment"
        v1 = embedder.embed(q)
        v2 = embedder.embed(q)
        np.testing.assert_array_almost_equal(v1, v2)


# ---------------------------------------------------------------------------
# embed_batch()
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    QUERIES = [
        "book an appointment",
        "cancel my visit",
        "what insurance do you accept",
    ]

    def test_batch_shape(self, embedder: QueryEmbedder):
        mat = embedder.embed_batch(self.QUERIES)
        assert mat.shape == (3, 384)

    def test_batch_dtype(self, embedder: QueryEmbedder):
        mat = embedder.embed_batch(self.QUERIES)
        assert mat.dtype == np.float32

    def test_batch_l2_normalised(self, embedder: QueryEmbedder):
        mat = embedder.embed_batch(self.QUERIES)
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_batch_matches_single(self, embedder: QueryEmbedder):
        """embed_batch results should match individual embed calls."""
        mat = embedder.embed_batch(self.QUERIES)
        for i, q in enumerate(self.QUERIES):
            single = embedder.embed(q)
            np.testing.assert_array_almost_equal(mat[i], single, decimal=5)


# ---------------------------------------------------------------------------
# dim()
# ---------------------------------------------------------------------------

class TestDim:
    def test_dim_value(self):
        assert QueryEmbedder.dim() == 384


# ---------------------------------------------------------------------------
# Semantic sanity
# ---------------------------------------------------------------------------

class TestSemanticSanity:
    """Verify that cosine similarity reflects semantic relatedness."""

    def test_similar_queries_cluster(self, embedder: QueryEmbedder):
        booking = embedder.embed("I want to book an appointment with a cardiologist")
        scheduling = embedder.embed("Schedule a visit with the heart doctor")
        insurance = embedder.embed("Do you accept Star Health insurance")

        # booking & scheduling should be more similar to each other
        # than either is to the insurance query.
        sim_book_sched = float(np.dot(booking, scheduling))
        sim_book_insur = float(np.dot(booking, insurance))
        assert sim_book_sched > sim_book_insur, (
            f"Expected booking↔scheduling ({sim_book_sched:.3f}) > "
            f"booking↔insurance ({sim_book_insur:.3f})"
        )
