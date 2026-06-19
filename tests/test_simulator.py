"""Tests for promptwise.simulator module."""

import pytest
import numpy as np

from promptwise.simulator import (
    QueryDistribution,
    SimulatedEnvironment,
    SimulationStep,
    simulated_quality,
    _SAMPLE_QUERIES,
)
from promptwise.prompts import Prompt, load_prompt_pool


# ---------------------------------------------------------------------------
# QueryDistribution
# ---------------------------------------------------------------------------


class TestQueryDistribution:
    def test_sample_returns_tuple(self):
        dist = QueryDistribution(seed=42)
        query, intent = dist.sample()
        assert isinstance(query, str)
        assert isinstance(intent, str)

    def test_sample_intent_is_valid(self):
        dist = QueryDistribution(seed=42)
        for _ in range(50):
            _, intent = dist.sample()
            assert intent in _SAMPLE_QUERIES

    def test_sample_query_in_pool(self):
        dist = QueryDistribution(seed=42)
        all_queries = {q for qs in _SAMPLE_QUERIES.values() for q in qs}
        for _ in range(50):
            query, _ = dist.sample()
            assert query in all_queries

    def test_batch(self):
        dist = QueryDistribution(seed=42)
        batch = dist.sample_batch(10)
        assert len(batch) == 10

    def test_seed_reproducibility(self):
        d1 = QueryDistribution(seed=123)
        d2 = QueryDistribution(seed=123)
        assert d1.sample_batch(5) == d2.sample_batch(5)

    def test_custom_weights(self):
        # Heavily bias towards emergency.
        weights = {k: 0.01 for k in _SAMPLE_QUERIES}
        weights["emergency"] = 100.0
        dist = QueryDistribution(weights=weights, seed=42)
        intents = [dist.sample()[1] for _ in range(50)]
        assert intents.count("emergency") > 30  # should dominate


# ---------------------------------------------------------------------------
# simulated_quality
# ---------------------------------------------------------------------------


class TestSimulatedQuality:
    def test_no_prompts(self):
        score = simulated_quality([], "booking", rng=np.random.default_rng(42))
        assert 0.0 <= score <= 1.0

    def test_relevant_prompts_score_higher(self):
        pool = load_prompt_pool()
        relevant = [p for p in pool if p.id == "example_booking_dialogue"]
        irrelevant = [p for p in pool if p.id == "ctx_insurance_accepted"]

        rng = np.random.default_rng(42)
        score_rel = simulated_quality(relevant, "booking", noise_std=0.0, rng=rng)
        rng = np.random.default_rng(42)
        score_irr = simulated_quality(irrelevant, "booking", noise_std=0.0, rng=rng)
        assert score_rel > score_irr

    def test_output_range(self):
        pool = load_prompt_pool()
        rng = np.random.default_rng(42)
        for _ in range(20):
            subset = list(np.random.default_rng().choice(pool, size=3, replace=False))
            s = simulated_quality(subset, "booking", rng=rng)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# SimulatedEnvironment (lightweight — uses embedder)
# ---------------------------------------------------------------------------


class TestSimulatedEnvironment:
    @pytest.fixture(scope="class")
    @classmethod
    def env(cls):
        return SimulatedEnvironment(budget=300, alpha=1.0, seed=42)

    def test_step_returns_result(self, env: SimulatedEnvironment):
        result = env.step()
        assert isinstance(result, SimulationStep)

    def test_step_fields(self, env: SimulatedEnvironment):
        result = env.step()
        assert isinstance(result.query, str)
        assert isinstance(result.intent, str)
        assert result.selection is not None
        assert result.reward is not None

    def test_budget_respected(self, env: SimulatedEnvironment):
        result = env.step()
        assert result.selection.total_token_cost <= result.selection.budget

    def test_run_multiple(self, env: SimulatedEnvironment):
        results = env.run(5)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, SimulationStep)

    def test_reset(self, env: SimulatedEnvironment):
        env.reset()
        assert env._step_count == 0
