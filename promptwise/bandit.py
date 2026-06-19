"""Contextual bandit algorithms for adaptive prompt selection.

Implements **LinUCB** (Li et al., 2010) — a linear upper-confidence-bound
algorithm that models expected reward as a linear function of the context
(query embedding) for each arm (candidate prompt).  An exploration bonus
encourages trying under-explored arms.

Key classes
-----------
LinUCBArm
    Per-arm ridge regression model with UCB exploration bonus.
LinUCBAgent
    Manages a collection of arms and implements the ``select`` / ``update``
    loop used by the prompt-selection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# LinUCB Arm
# ---------------------------------------------------------------------------


class LinUCBArm:
    """A single arm in the LinUCB bandit, maintaining its own ridge model.

    Parameters
    ----------
    arm_id : str
        Unique identifier (matches ``Prompt.id``).
    d : int
        Dimensionality of the context vector (e.g. 384 for MiniLM).
    """

    def __init__(self, arm_id: str, d: int) -> None:
        self.arm_id = arm_id
        self.d = d

        # A = d×d identity matrix  (acts as regularised design matrix)
        self.A: np.ndarray = np.eye(d, dtype=np.float64)
        # b = d-dim zero vector (reward-weighted context accumulator)
        self.b: np.ndarray = np.zeros(d, dtype=np.float64)

    # -- prediction ---------------------------------------------------------

    def ucb(self, context: np.ndarray, alpha: float) -> float:
        """Return the upper confidence bound for *context*.

        Parameters
        ----------
        context : np.ndarray
            Shape ``(d,)``.
        alpha : float
            Exploration parameter — higher means more exploration.

        Returns
        -------
        float
            ``θᵀx + α √(xᵀ A⁻¹ x)``
        """
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b  # ridge regression coefficients
        x = context.astype(np.float64)
        exploitation = float(theta @ x)
        exploration = alpha * float(np.sqrt(x @ A_inv @ x))
        return exploitation + exploration

    def predict(self, context: np.ndarray) -> float:
        """Return the predicted reward (exploitation term only)."""
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        return float(theta @ context.astype(np.float64))

    # -- learning -----------------------------------------------------------

    def update(self, context: np.ndarray, reward: float) -> None:
        """Incorporate a new ``(context, reward)`` observation.

        Parameters
        ----------
        context : np.ndarray
            Shape ``(d,)``.
        reward : float
            Observed reward for this arm given *context*.
        """
        x = context.astype(np.float64)
        self.A += np.outer(x, x)
        self.b += reward * x

    # -- introspection ------------------------------------------------------

    @property
    def theta(self) -> np.ndarray:
        """Current ridge regression weight vector θ = A⁻¹b."""
        return np.linalg.inv(self.A) @ self.b

    @property
    def n_updates(self) -> int:
        """Number of times this arm has been updated.

        Since ``A`` starts as ``I``, the number of rank-1 updates is
        ``trace(A) - d`` (each outer-product adds 1 to the trace for
        unit-norm contexts), but we track it approximately via the
        trace.  For an exact count we fall back to the determinant
        growth heuristic.  A simpler proxy: ``trace(A) - d``.
        """
        return max(0, int(round(np.trace(self.A) - self.d)))

    def reset(self) -> None:
        """Reset the arm to its initial state."""
        self.A = np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)


# ---------------------------------------------------------------------------
# LinUCB Agent
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """Result of a single ``LinUCBAgent.select()`` call.

    Attributes
    ----------
    arm_ids : list[str]
        Ordered list of selected arm identifiers.
    ucb_scores : dict[str, float]
        UCB score for every arm at selection time.
    """

    arm_ids: list[str]
    ucb_scores: dict[str, float]


class LinUCBAgent:
    """LinUCB agent managing multiple arms (one per candidate prompt).

    Parameters
    ----------
    arm_ids : list[str]
        Identifiers for each arm (one per prompt in the pool).
    d : int
        Context dimensionality (default 384 for MiniLM embeddings).
    alpha : float
        Exploration parameter (default 1.0).  Higher → more exploration.
    """

    def __init__(
        self,
        arm_ids: list[str],
        d: int = 384,
        alpha: float = 1.0,
    ) -> None:
        self.d = d
        self.alpha = alpha
        self.arms: dict[str, LinUCBArm] = {
            aid: LinUCBArm(aid, d) for aid in arm_ids
        }

    # -- selection ----------------------------------------------------------

    def score_all(self, context: np.ndarray) -> dict[str, float]:
        """Return UCB scores for all arms given *context*."""
        return {
            aid: arm.ucb(context, self.alpha)
            for aid, arm in self.arms.items()
        }

    def rank(self, context: np.ndarray) -> list[tuple[str, float]]:
        """Return all arms ranked by descending UCB score.

        Returns
        -------
        list[tuple[str, float]]
            ``[(arm_id, ucb_score), ...]`` sorted highest-first.
        """
        scores = self.score_all(context)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def select_top_k(self, context: np.ndarray, k: int) -> SelectionResult:
        """Select the *k* arms with the highest UCB scores.

        This is a simple top-k selection **without** budget constraints.
        For budget-aware selection, pipe the UCB scores into the
        :mod:`promptwise.knapsack` module.

        Parameters
        ----------
        context : np.ndarray
            Shape ``(d,)``.
        k : int
            Number of arms to select.

        Returns
        -------
        SelectionResult
        """
        ranked = self.rank(context)
        selected = [aid for aid, _ in ranked[:k]]
        scores = {aid: score for aid, score in ranked}
        return SelectionResult(arm_ids=selected, ucb_scores=scores)

    # -- learning -----------------------------------------------------------

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """Update a single arm with an observed reward.

        Parameters
        ----------
        arm_id : str
            The arm that was played.
        context : np.ndarray
            The context that was presented.
        reward : float
            The observed reward.

        Raises
        ------
        KeyError
            If *arm_id* is not known.
        """
        self.arms[arm_id].update(context, reward)

    def batch_update(
        self,
        arm_ids: list[str],
        context: np.ndarray,
        rewards: list[float],
    ) -> None:
        """Update multiple arms for the same context.

        Useful when a subset of prompts was selected and each received
        the same (or decomposed) reward signal.
        """
        for aid, r in zip(arm_ids, rewards):
            self.update(aid, context, r)

    # -- introspection ------------------------------------------------------

    @property
    def arm_ids(self) -> list[str]:
        return list(self.arms.keys())

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def get_arm(self, arm_id: str) -> LinUCBArm:
        return self.arms[arm_id]

    def reset(self) -> None:
        """Reset all arms to their initial state."""
        for arm in self.arms.values():
            arm.reset()
