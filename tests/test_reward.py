"""Tests for promptwise.reward module."""

import pytest

from promptwise.reward import (
    RewardBreakdown,
    quality_score,
    token_efficiency_score,
    latency_score,
    compute_simulated_reward,
)


# ---------------------------------------------------------------------------
# quality_score
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_empty_response(self):
        assert quality_score("", "book an appointment") == 0.0

    def test_relevant_response(self):
        response = "Your appointment is booked for Tuesday at 10 AM with Dr. Sharma."
        score = quality_score(response, "I want to book an appointment")
        assert 0.0 < score <= 1.0

    def test_irrelevant_response(self):
        response = "The weather is sunny today."
        score_relevant = quality_score(
            "Your appointment is confirmed for Monday.",
            "book an appointment",
        )
        score_irrelevant = quality_score(response, "book an appointment")
        assert score_relevant > score_irrelevant

    def test_output_range(self):
        for _ in range(10):
            s = quality_score("Some response text with appointment details.", "test query")
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# token_efficiency_score
# ---------------------------------------------------------------------------


class TestTokenEfficiency:
    def test_zero_tokens(self):
        score = token_efficiency_score(0, 0, budget=100)
        assert score == pytest.approx(1.0)

    def test_at_budget(self):
        # ratio = 1.0 → score = 0.5
        score = token_efficiency_score(100, 0, budget=100)
        assert score == pytest.approx(0.5)

    def test_over_budget(self):
        score = token_efficiency_score(200, 0, budget=100)
        assert score == 0.0

    def test_output_range(self):
        for total in range(0, 300, 20):
            s = token_efficiency_score(total, 0, budget=100)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# latency_score
# ---------------------------------------------------------------------------


class TestLatencyScore:
    def test_instant(self):
        assert latency_score(0.0) == 1.0

    def test_at_target(self):
        # latency = target → 1 - (target / 2*target) = 0.5
        assert latency_score(3.0, target_s=3.0) == pytest.approx(0.5)

    def test_very_slow(self):
        score = latency_score(10.0, target_s=3.0)
        assert score == 0.0

    def test_output_range(self):
        for t in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            s = latency_score(t)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# compute_simulated_reward
# ---------------------------------------------------------------------------


class TestComputeSimulatedReward:
    def test_returns_breakdown(self):
        result = compute_simulated_reward(
            response_text="Appointment confirmed with Dr. Sharma.",
            query="Book an appointment",
            total_token_cost=50,
            budget=200,
        )
        assert isinstance(result, RewardBreakdown)

    def test_total_in_range(self):
        result = compute_simulated_reward(
            response_text="Your appointment is cancelled.",
            query="Cancel my appointment",
            total_token_cost=30,
            budget=200,
        )
        assert 0.0 <= result.total <= 1.0

    def test_weights_sum_to_one(self):
        """Default weights should produce a sensible total."""
        result = compute_simulated_reward(
            response_text="Available slots: Monday 9 AM, Tuesday 2 PM.",
            query="When is the doctor available?",
            total_token_cost=80,
            budget=300,
        )
        # total should be weighted avg of components, all in [0,1].
        assert result.total <= max(result.quality, result.token_efficiency, result.latency_score)
