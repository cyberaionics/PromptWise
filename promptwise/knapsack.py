"""Budget-constrained prompt selection via the 0-1 knapsack problem.

Given a set of candidate prompts, each with a predicted reward (UCB score)
and a token cost, select the subset that **maximises total reward** while
keeping the **total token cost ≤ budget**.

Two solvers are provided:

* :func:`knapsack_dp` — exact dynamic-programming solution (pseudo-
  polynomial in the budget).
* :func:`knapsack_greedy` — fast reward/cost ratio greedy heuristic with
  a ½-approximation guarantee.

A convenience wrapper :func:`select_prompts` ties the bandit scores to
the knapsack solver for the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from promptwise.prompts import Prompt


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class KnapsackResult:
    """Result of a budget-constrained prompt selection.

    Attributes
    ----------
    selected : list[Prompt]
        The chosen prompts (order preserved from input).
    total_token_cost : int
        Sum of ``token_cost`` for the selected prompts.
    total_reward : float
        Sum of predicted rewards for the selected prompts.
    budget : int
        The token budget that was enforced.
    """

    selected: list[Prompt]
    total_token_cost: int
    total_reward: float
    budget: int


# ---------------------------------------------------------------------------
# Exact DP solver
# ---------------------------------------------------------------------------


def knapsack_dp(
    prompts: Sequence[Prompt],
    scores: dict[str, float],
    budget: int,
) -> KnapsackResult:
    """Solve the 0-1 knapsack exactly with dynamic programming.

    Parameters
    ----------
    prompts : Sequence[Prompt]
        Candidate prompts (each has ``.id`` and ``.token_cost``).
    scores : dict[str, float]
        Mapping of ``prompt.id → predicted reward`` (e.g. UCB scores).
    budget : int
        Maximum total token cost allowed.

    Returns
    -------
    KnapsackResult

    Notes
    -----
    Time complexity: O(n × budget).  For very large budgets consider
    :func:`knapsack_greedy`.
    """
    n = len(prompts)
    # dp[i][w] = best reward using first i items with capacity w
    dp = np.zeros((n + 1, budget + 1), dtype=np.float64)

    for i in range(1, n + 1):
        p = prompts[i - 1]
        w = p.token_cost
        v = scores.get(p.id, 0.0)
        for cap in range(budget + 1):
            if w <= cap:
                dp[i, cap] = max(dp[i - 1, cap], dp[i - 1, cap - w] + v)
            else:
                dp[i, cap] = dp[i - 1, cap]

    # Backtrack to find selected items.
    selected: list[Prompt] = []
    cap = budget
    for i in range(n, 0, -1):
        if dp[i, cap] != dp[i - 1, cap]:
            selected.append(prompts[i - 1])
            cap -= prompts[i - 1].token_cost

    selected.reverse()  # preserve original order
    total_cost = sum(p.token_cost for p in selected)
    total_reward = sum(scores.get(p.id, 0.0) for p in selected)
    return KnapsackResult(
        selected=selected,
        total_token_cost=total_cost,
        total_reward=total_reward,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Greedy solver (½-approximation)
# ---------------------------------------------------------------------------


def knapsack_greedy(
    prompts: Sequence[Prompt],
    scores: dict[str, float],
    budget: int,
) -> KnapsackResult:
    """Greedy knapsack by reward-to-cost ratio (½-approximation).

    Faster than DP for large budgets.  Sorts prompts by
    ``score / token_cost`` descending and greedily adds items that fit.

    Parameters
    ----------
    prompts : Sequence[Prompt]
        Candidate prompts.
    scores : dict[str, float]
        ``prompt.id → predicted reward``.
    budget : int
        Token budget.

    Returns
    -------
    KnapsackResult
    """
    # Compute efficiency = score / cost for each prompt.
    items = []
    for p in prompts:
        s = scores.get(p.id, 0.0)
        efficiency = s / p.token_cost if p.token_cost > 0 else 0.0
        items.append((efficiency, p, s))

    # Sort by efficiency descending, break ties by higher absolute score.
    items.sort(key=lambda x: (x[0], x[2]), reverse=True)

    selected: list[Prompt] = []
    remaining_budget = budget
    total_reward = 0.0

    for _, p, s in items:
        if p.token_cost <= remaining_budget:
            selected.append(p)
            remaining_budget -= p.token_cost
            total_reward += s

    total_cost = budget - remaining_budget
    return KnapsackResult(
        selected=selected,
        total_token_cost=total_cost,
        total_reward=total_reward,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Pipeline convenience
# ---------------------------------------------------------------------------


def select_prompts(
    prompts: Sequence[Prompt],
    scores: dict[str, float],
    budget: int,
    *,
    method: str = "dp",
) -> KnapsackResult:
    """Select prompts under a token budget using the specified solver.

    Parameters
    ----------
    prompts : Sequence[Prompt]
        Full prompt pool.
    scores : dict[str, float]
        ``prompt.id → predicted reward`` (typically UCB scores from
        :class:`~promptwise.bandit.LinUCBAgent`).
    budget : int
        Maximum total token cost.
    method : str
        ``"dp"`` for exact DP or ``"greedy"`` for the greedy heuristic.

    Returns
    -------
    KnapsackResult
    """
    if method == "dp":
        return knapsack_dp(prompts, scores, budget)
    elif method == "greedy":
        return knapsack_greedy(prompts, scores, budget)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'dp' or 'greedy'.")
