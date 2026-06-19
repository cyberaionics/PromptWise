"""Tests for promptwise.knapsack module."""

import pytest

from promptwise.knapsack import (
    KnapsackResult,
    knapsack_dp,
    knapsack_greedy,
    select_prompts,
)
from promptwise.prompts import Prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt(pid: str, cost: int) -> Prompt:
    """Create a minimal Prompt with a known token cost.

    We override token_cost after construction to control it exactly.
    """
    p = Prompt(id=pid, category="instruction", text="x" * cost, tags=["test"])
    # Override the tiktoken-computed cost with an exact value for testing.
    object.__setattr__(p, "token_cost", cost)
    return p


# ---------------------------------------------------------------------------
# knapsack_dp
# ---------------------------------------------------------------------------


class TestKnapsackDP:
    def test_empty_pool(self):
        result = knapsack_dp([], {}, budget=100)
        assert result.selected == []
        assert result.total_token_cost == 0

    def test_single_item_fits(self):
        p = _make_prompt("a", cost=10)
        result = knapsack_dp([p], {"a": 5.0}, budget=20)
        assert len(result.selected) == 1
        assert result.selected[0].id == "a"

    def test_single_item_too_expensive(self):
        p = _make_prompt("a", cost=30)
        result = knapsack_dp([p], {"a": 5.0}, budget=20)
        assert result.selected == []

    def test_selects_optimal_subset(self):
        # Item a: cost=5, reward=4  → density=0.8
        # Item b: cost=4, reward=5  → density=1.25 ← best
        # Item c: cost=3, reward=3  → density=1.0
        # Budget=7: optimal is b+c (reward=8, cost=7).
        a = _make_prompt("a", 5)
        b = _make_prompt("b", 4)
        c = _make_prompt("c", 3)
        scores = {"a": 4.0, "b": 5.0, "c": 3.0}
        result = knapsack_dp([a, b, c], scores, budget=7)
        selected_ids = {p.id for p in result.selected}
        assert selected_ids == {"b", "c"}
        assert result.total_reward == pytest.approx(8.0)
        assert result.total_token_cost == 7

    def test_budget_respected(self):
        items = [_make_prompt(f"p{i}", cost=10) for i in range(5)]
        scores = {f"p{i}": 1.0 for i in range(5)}
        result = knapsack_dp(items, scores, budget=25)
        assert result.total_token_cost <= 25

    def test_result_type(self):
        result = knapsack_dp([], {}, budget=10)
        assert isinstance(result, KnapsackResult)


# ---------------------------------------------------------------------------
# knapsack_greedy
# ---------------------------------------------------------------------------


class TestKnapsackGreedy:
    def test_empty_pool(self):
        result = knapsack_greedy([], {}, budget=100)
        assert result.selected == []

    def test_greedy_selects_high_efficiency_first(self):
        # a: cost=10, reward=1 → eff=0.1
        # b: cost=5,  reward=3 → eff=0.6  ← should be picked first
        a = _make_prompt("a", 10)
        b = _make_prompt("b", 5)
        scores = {"a": 1.0, "b": 3.0}
        result = knapsack_greedy([a, b], scores, budget=15)
        # Greedy picks b first (eff=0.6), then a fits in remaining budget.
        assert {p.id for p in result.selected} == {"a", "b"}

    def test_budget_respected(self):
        items = [_make_prompt(f"p{i}", cost=10) for i in range(5)]
        scores = {f"p{i}": 1.0 for i in range(5)}
        result = knapsack_greedy(items, scores, budget=25)
        assert result.total_token_cost <= 25


# ---------------------------------------------------------------------------
# select_prompts (unified API)
# ---------------------------------------------------------------------------


class TestSelectPrompts:
    def test_dp_method(self):
        p = _make_prompt("a", 10)
        result = select_prompts([p], {"a": 5.0}, budget=20, method="dp")
        assert len(result.selected) == 1

    def test_greedy_method(self):
        p = _make_prompt("a", 10)
        result = select_prompts([p], {"a": 5.0}, budget=20, method="greedy")
        assert len(result.selected) == 1

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            select_prompts([], {}, budget=10, method="unknown")

    def test_budget_field_set(self):
        result = select_prompts([], {}, budget=42, method="dp")
        assert result.budget == 42
