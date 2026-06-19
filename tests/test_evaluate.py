"""Tests for promptwise.evaluate module."""

import numpy as np
import pandas as pd
import pytest

from promptwise.evaluate import (
    EvalSummary,
    compare_runs,
    cumulative_reward,
    extract_metrics,
    rolling_mean_reward,
    summarise,
)
from promptwise.knapsack import KnapsackResult
from promptwise.prompts import Prompt
from promptwise.reward import RewardBreakdown
from promptwise.simulator import SimulationStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_steps(n: int = 10) -> list[SimulationStep]:
    """Build fake SimulationStep objects for testing."""
    steps = []
    for i in range(n):
        p = Prompt(id=f"p{i}", category="instruction", text="test", tags=["test"])
        sel = KnapsackResult(
            selected=[p],
            total_token_cost=20,
            total_reward=0.5 + 0.01 * i,
            budget=100,
        )
        rew = RewardBreakdown(
            quality=0.6 + 0.01 * i,
            token_efficiency=0.8,
            latency_score=0.9,
            total=0.5 + 0.02 * i,
        )
        steps.append(
            SimulationStep(
                step=i,
                query=f"query {i}",
                intent="booking" if i % 2 == 0 else "cancellation",
                selection=sel,
                reward=rew,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_returns_dataframe(self):
        df = extract_metrics(_make_steps(5))
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = extract_metrics(_make_steps(5))
        expected = {
            "step", "query", "intent", "n_prompts",
            "total_token_cost", "budget", "reward_quality",
            "reward_token_eff", "reward_latency", "reward_total",
            "selected_ids",
        }
        assert expected.issubset(set(df.columns))

    def test_row_count(self):
        df = extract_metrics(_make_steps(7))
        assert len(df) == 7


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_returns_summary(self):
        df = extract_metrics(_make_steps(10))
        s = summarise(df)
        assert isinstance(s, EvalSummary)

    def test_n_steps(self):
        df = extract_metrics(_make_steps(10))
        s = summarise(df)
        assert s.n_steps == 10

    def test_mean_reward_in_range(self):
        df = extract_metrics(_make_steps(10))
        s = summarise(df)
        assert 0.0 <= s.mean_reward <= 1.0

    def test_reward_per_intent(self):
        df = extract_metrics(_make_steps(10))
        s = summarise(df)
        assert "booking" in s.reward_per_intent
        assert "cancellation" in s.reward_per_intent


# ---------------------------------------------------------------------------
# cumulative / rolling
# ---------------------------------------------------------------------------


class TestCumulativeRolling:
    def test_cumulative_shape(self):
        df = extract_metrics(_make_steps(10))
        cum = cumulative_reward(df)
        assert len(cum) == 10

    def test_cumulative_monotonic(self):
        df = extract_metrics(_make_steps(10))
        cum = cumulative_reward(df)
        assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))

    def test_rolling_shape(self):
        df = extract_metrics(_make_steps(20))
        roll = rolling_mean_reward(df, window=5)
        assert len(roll) == 20

    def test_rolling_values_in_range(self):
        df = extract_metrics(_make_steps(20))
        roll = rolling_mean_reward(df, window=5)
        assert all(0.0 <= v <= 1.0 for v in roll)


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_returns_dataframe(self):
        df1 = extract_metrics(_make_steps(10))
        df2 = extract_metrics(_make_steps(10))
        result = compare_runs({"run_a": df1, "run_b": df2})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_index_is_run_name(self):
        df1 = extract_metrics(_make_steps(5))
        result = compare_runs({"alpha=0.5": df1})
        assert "alpha=0.5" in result.index
