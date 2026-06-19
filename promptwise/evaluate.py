"""Evaluation utilities for the PromptWise prompt-selection system.

Provides functions to:

* Compute cumulative and rolling metrics from simulation results.
* Compare different bandit configurations or baselines.
* Generate summary tables and (optionally) matplotlib plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from promptwise.simulator import SimulationStep


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_metrics(steps: Sequence[SimulationStep]) -> pd.DataFrame:
    """Convert simulation steps into a tidy DataFrame.

    Columns
    -------
    step, query, intent, n_prompts, total_token_cost, budget,
    reward_quality, reward_token_eff, reward_latency, reward_total
    """
    rows = []
    for s in steps:
        rows.append(
            {
                "step": s.step,
                "query": s.query,
                "intent": s.intent,
                "n_prompts": len(s.selection.selected),
                "total_token_cost": s.selection.total_token_cost,
                "budget": s.selection.budget,
                "reward_quality": s.reward.quality,
                "reward_token_eff": s.reward.token_efficiency,
                "reward_latency": s.reward.latency_score,
                "reward_total": s.reward.total,
                "selected_ids": [p.id for p in s.selection.selected],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@dataclass
class EvalSummary:
    """Aggregate evaluation statistics.

    Attributes
    ----------
    n_steps : int
    mean_reward : float
    std_reward : float
    mean_token_cost : float
    mean_n_prompts : float
    cumulative_reward : float
    reward_per_intent : dict[str, float]
    """

    n_steps: int
    mean_reward: float
    std_reward: float
    mean_token_cost: float
    mean_n_prompts: float
    cumulative_reward: float
    reward_per_intent: dict[str, float] = field(default_factory=dict)


def summarise(df: pd.DataFrame) -> EvalSummary:
    """Compute aggregate statistics from an evaluation DataFrame."""
    per_intent = (
        df.groupby("intent")["reward_total"]
        .mean()
        .to_dict()
    )
    return EvalSummary(
        n_steps=len(df),
        mean_reward=float(df["reward_total"].mean()),
        std_reward=float(df["reward_total"].std()),
        mean_token_cost=float(df["total_token_cost"].mean()),
        mean_n_prompts=float(df["n_prompts"].mean()),
        cumulative_reward=float(df["reward_total"].sum()),
        reward_per_intent=per_intent,
    )


# ---------------------------------------------------------------------------
# Rolling / cumulative helpers
# ---------------------------------------------------------------------------


def cumulative_reward(df: pd.DataFrame) -> np.ndarray:
    """Return array of cumulative reward at each step."""
    return np.cumsum(df["reward_total"].values)


def rolling_mean_reward(df: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Return rolling mean of ``reward_total`` with given *window*."""
    return (
        df["reward_total"]
        .rolling(window=window, min_periods=1)
        .mean()
        .values
    )


# ---------------------------------------------------------------------------
# Comparison across runs
# ---------------------------------------------------------------------------


def compare_runs(
    runs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare aggregate metrics across multiple named runs.

    Parameters
    ----------
    runs : dict[str, pd.DataFrame]
        Mapping of ``run_name → evaluation DataFrame``.

    Returns
    -------
    pd.DataFrame
        One row per run with summary statistics.
    """
    records = []
    for name, df in runs.items():
        s = summarise(df)
        records.append(
            {
                "run": name,
                "n_steps": s.n_steps,
                "mean_reward": s.mean_reward,
                "std_reward": s.std_reward,
                "mean_token_cost": s.mean_token_cost,
                "mean_n_prompts": s.mean_n_prompts,
                "cumulative_reward": s.cumulative_reward,
            }
        )
    return pd.DataFrame(records).set_index("run")


# ---------------------------------------------------------------------------
# Plotting (optional — guarded behind matplotlib import)
# ---------------------------------------------------------------------------


def plot_learning_curve(
    df: pd.DataFrame,
    window: int = 20,
    title: str = "PromptWise Learning Curve",
    save_path: str | None = None,
) -> None:
    """Plot cumulative and rolling-mean reward curves.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`extract_metrics`.
    window : int
        Rolling window size.
    title : str
        Plot title.
    save_path : str | None
        If given, save the figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative reward.
    cum = cumulative_reward(df)
    axes[0].plot(cum, linewidth=1.5)
    axes[0].set_title("Cumulative Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].grid(alpha=0.3)

    # Rolling mean reward.
    roll = rolling_mean_reward(df, window)
    axes[1].plot(roll, linewidth=1.5, color="orange")
    axes[1].set_title(f"Rolling Mean Reward (window={window})")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean Reward")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_comparison(
    runs: dict[str, pd.DataFrame],
    window: int = 20,
    title: str = "Strategy Comparison",
    save_path: str | None = None,
) -> None:
    """Overlay rolling-mean reward curves for multiple runs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in runs.items():
        roll = rolling_mean_reward(df, window)
        ax.plot(roll, label=name, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Rolling Mean Reward (window={window})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
