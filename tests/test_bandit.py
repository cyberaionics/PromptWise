"""Tests for promptwise.bandit module."""

import numpy as np
import pytest

from promptwise.bandit import LinUCBArm, LinUCBAgent, SelectionResult


# ---------------------------------------------------------------------------
# LinUCBArm
# ---------------------------------------------------------------------------


class TestLinUCBArm:
    D = 8  # small dimensionality for fast tests

    def test_init_shapes(self):
        arm = LinUCBArm("test", self.D)
        assert arm.A.shape == (self.D, self.D)
        assert arm.b.shape == (self.D,)

    def test_initial_A_is_identity(self):
        arm = LinUCBArm("test", self.D)
        np.testing.assert_array_equal(arm.A, np.eye(self.D))

    def test_initial_b_is_zero(self):
        arm = LinUCBArm("test", self.D)
        np.testing.assert_array_equal(arm.b, np.zeros(self.D))

    def test_ucb_positive_alpha(self):
        arm = LinUCBArm("test", self.D)
        ctx = np.random.randn(self.D)
        # With identity A and zero b, exploitation=0, exploration > 0.
        score = arm.ucb(ctx, alpha=1.0)
        assert score > 0

    def test_ucb_zero_alpha_equals_predict(self):
        arm = LinUCBArm("test", self.D)
        ctx = np.random.randn(self.D)
        arm.update(ctx, 1.0)
        assert abs(arm.ucb(ctx, alpha=0.0) - arm.predict(ctx)) < 1e-10

    def test_update_changes_A_and_b(self):
        arm = LinUCBArm("test", self.D)
        A_before = arm.A.copy()
        b_before = arm.b.copy()
        ctx = np.ones(self.D)
        arm.update(ctx, 1.0)
        assert not np.allclose(arm.A, A_before)
        assert not np.allclose(arm.b, b_before)

    def test_predict_after_update(self):
        arm = LinUCBArm("test", self.D)
        ctx = np.ones(self.D) / np.sqrt(self.D)  # unit norm
        arm.update(ctx, 1.0)
        # After one positive update, prediction on same ctx should be > 0.
        assert arm.predict(ctx) > 0

    def test_theta_shape(self):
        arm = LinUCBArm("test", self.D)
        assert arm.theta.shape == (self.D,)

    def test_reset(self):
        arm = LinUCBArm("test", self.D)
        arm.update(np.ones(self.D), 1.0)
        arm.reset()
        np.testing.assert_array_equal(arm.A, np.eye(self.D))
        np.testing.assert_array_equal(arm.b, np.zeros(self.D))


# ---------------------------------------------------------------------------
# LinUCBAgent
# ---------------------------------------------------------------------------


class TestLinUCBAgent:
    ARM_IDS = ["a", "b", "c", "d"]
    D = 8

    def _make_agent(self, alpha: float = 1.0) -> LinUCBAgent:
        return LinUCBAgent(self.ARM_IDS, d=self.D, alpha=alpha)

    def test_n_arms(self):
        agent = self._make_agent()
        assert agent.n_arms == len(self.ARM_IDS)

    def test_arm_ids(self):
        agent = self._make_agent()
        assert set(agent.arm_ids) == set(self.ARM_IDS)

    def test_score_all_returns_all_arms(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        scores = agent.score_all(ctx)
        assert set(scores.keys()) == set(self.ARM_IDS)

    def test_rank_returns_sorted(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        ranked = agent.rank(ctx)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_select_top_k_count(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        result = agent.select_top_k(ctx, k=2)
        assert isinstance(result, SelectionResult)
        assert len(result.arm_ids) == 2

    def test_update_known_arm(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        agent.update("a", ctx, 1.0)  # should not raise

    def test_update_unknown_arm_raises(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        with pytest.raises(KeyError):
            agent.update("nonexistent", ctx, 1.0)

    def test_batch_update(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        agent.batch_update(["a", "b"], ctx, [1.0, 0.5])
        # After updates, arm "a" and "b" should have different predictions.
        pred_a = agent.get_arm("a").predict(ctx)
        pred_b = agent.get_arm("b").predict(ctx)
        assert pred_a > pred_b

    def test_learning_shifts_preferences(self):
        """After repeated positive feedback, the rewarded arm should rank higher."""
        agent = self._make_agent(alpha=0.1)
        ctx = np.random.randn(self.D)
        # Repeatedly reward arm "c".
        for _ in range(20):
            agent.update("c", ctx, 1.0)
        ranked = agent.rank(ctx)
        top_arm = ranked[0][0]
        assert top_arm == "c", f"Expected 'c' on top, got {top_arm}"

    def test_reset(self):
        agent = self._make_agent()
        ctx = np.random.randn(self.D)
        agent.update("a", ctx, 1.0)
        agent.reset()
        # After reset, all predictions should be zero.
        for aid in self.ARM_IDS:
            assert agent.get_arm(aid).predict(ctx) == pytest.approx(0.0, abs=1e-10)
