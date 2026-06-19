"""Simulation environment for offline bandit evaluation.

Generates synthetic user queries drawn from a distribution of intents
(booking, cancellation, availability, …) and provides a simulated reward
function so the bandit can be trained and evaluated **without** real API
calls.

Key classes
-----------
QueryDistribution
    Samples synthetic queries from weighted intent categories.
SimulatedEnvironment
    Full loop: sample query → embed → bandit select → knapsack →
    simulated reward → bandit update.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from promptwise.bandit import LinUCBAgent, SelectionResult
from promptwise.embeddings import QueryEmbedder
from promptwise.knapsack import KnapsackResult, select_prompts
from promptwise.prompts import Prompt, load_prompt_pool
from promptwise.reward import RewardBreakdown, compute_simulated_reward


# ---------------------------------------------------------------------------
# Sample queries by intent
# ---------------------------------------------------------------------------

_SAMPLE_QUERIES: dict[str, list[str]] = {
    "booking": [
        "I want to book an appointment with a cardiologist next week.",
        "Can I schedule a visit with Dr. Sharma on Tuesday?",
        "I need to see an orthopedic doctor as soon as possible.",
        "Book me a follow-up appointment with Dr. Chen.",
        "I'd like a new patient appointment for a wellness check.",
    ],
    "cancellation": [
        "I need to cancel my appointment tomorrow.",
        "Please cancel my visit with Dr. Okafor on Friday.",
        "Can I cancel and rebook for next month?",
        "I won't be able to make my 3 PM appointment today.",
    ],
    "availability": [
        "What times are available for Dr. Kim this week?",
        "Is there a morning slot open with any dermatologist?",
        "When is Dr. Fernandez next available?",
        "Do you have any openings for a telehealth visit?",
    ],
    "insurance": [
        "Do you accept Star Health insurance?",
        "Is BlueCross BlueShield covered at your clinic?",
        "What insurance plans do you take?",
        "I have Medicare — can I book an appointment?",
    ],
    "emergency": [
        "I'm having severe chest pain, what should I do?",
        "My child has a very high fever and is not responding.",
        "This is an emergency, I need help now!",
    ],
    "general_info": [
        "What are your clinic hours?",
        "Where is the clinic located?",
        "What is your cancellation policy?",
        "Do I need a referral to see a specialist?",
        "How do I prepare for my first visit?",
    ],
}


@dataclass
class QueryDistribution:
    """Weighted sampler over intent categories.

    Parameters
    ----------
    weights : dict[str, float] | None
        Mapping of ``intent → relative weight``.  If ``None``, uniform
        weights are used.
    seed : int | None
        Random seed for reproducibility.
    """

    weights: dict[str, float] | None = None
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.weights is None:
            self.weights = {k: 1.0 for k in _SAMPLE_QUERIES}

    def sample(self) -> tuple[str, str]:
        """Sample a ``(query_text, intent)`` pair."""
        intents = list(self.weights.keys())
        wts = [self.weights[i] for i in intents]
        intent = self._rng.choices(intents, weights=wts, k=1)[0]
        query = self._rng.choice(_SAMPLE_QUERIES[intent])
        return query, intent

    def sample_batch(self, n: int) -> list[tuple[str, str]]:
        """Sample *n* ``(query_text, intent)`` pairs."""
        return [self.sample() for _ in range(n)]


# ---------------------------------------------------------------------------
# Simulated reward
# ---------------------------------------------------------------------------

# Ground-truth prompt relevance: which prompts are most useful per intent.
# Higher weight → this prompt genuinely helps the response for this intent.
_PROMPT_INTENT_AFFINITY: dict[str, dict[str, float]] = {
    "booking": {
        "instr_confirm_booking": 0.9,
        "instr_tone": 0.5,
        "instr_verify_identity": 0.6,
        "example_booking_dialogue": 0.95,
        "ctx_doctor_specialties": 0.8,
        "ctx_doctor_schedules": 0.85,
        "ctx_appointment_types": 0.7,
        "ctx_referral_requirements": 0.5,
    },
    "cancellation": {
        "instr_tone": 0.4,
        "instr_verify_identity": 0.5,
        "example_cancellation_dialogue": 0.9,
        "ctx_cancellation_policy": 0.95,
    },
    "availability": {
        "instr_clarify_ambiguity": 0.6,
        "ctx_doctor_specialties": 0.85,
        "ctx_doctor_schedules": 0.9,
        "ctx_appointment_types": 0.5,
    },
    "insurance": {
        "instr_privacy": 0.5,
        "ctx_insurance_accepted": 0.95,
        "ctx_new_vs_returning": 0.4,
    },
    "emergency": {
        "instr_escalate_emergency": 0.99,
        "instr_no_medical_advice": 0.7,
        "ctx_urgent_care_guidelines": 0.6,
    },
    "general_info": {
        "instr_tone": 0.5,
        "ctx_clinic_hours": 0.7,
        "ctx_appointment_types": 0.5,
        "ctx_new_vs_returning": 0.4,
        "ctx_followup_rules": 0.3,
    },
}


def simulated_quality(
    selected_prompts: Sequence[Prompt],
    intent: str,
    *,
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute a simulated quality score in [0, 1].

    The score is based on how well the *selected_prompts* match the
    ground-truth affinity for *intent*, plus Gaussian noise.
    """
    rng = rng or np.random.default_rng()
    affinities = _PROMPT_INTENT_AFFINITY.get(intent, {})

    if not selected_prompts:
        return max(0.0, 0.1 + rng.normal(0, noise_std))

    # Average affinity of selected prompts for this intent.
    total_affinity = sum(
        affinities.get(p.id, 0.05) for p in selected_prompts
    )
    avg_affinity = total_affinity / len(selected_prompts)

    # Bonus for including at least one highly relevant prompt.
    max_affinity = max(
        (affinities.get(p.id, 0.05) for p in selected_prompts), default=0.05
    )
    bonus = 0.15 if max_affinity > 0.8 else 0.0

    score = avg_affinity + bonus + rng.normal(0, noise_std)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Simulation step result
# ---------------------------------------------------------------------------


@dataclass
class SimulationStep:
    """Record of a single simulation step.

    Attributes
    ----------
    step : int
        Step index.
    query : str
        The sampled user query.
    intent : str
        Ground-truth intent category.
    selection : KnapsackResult
        Prompts selected and budget info.
    reward : RewardBreakdown
        Decomposed reward signal.
    """

    step: int
    query: str
    intent: str
    selection: KnapsackResult
    reward: RewardBreakdown


# ---------------------------------------------------------------------------
# Simulated Environment
# ---------------------------------------------------------------------------


class SimulatedEnvironment:
    """Full simulation loop for offline bandit evaluation.

    Parameters
    ----------
    prompts : list[Prompt] | None
        Prompt pool (defaults to :func:`load_prompt_pool`).
    budget : int
        Token budget per query (default 300).
    alpha : float
        LinUCB exploration parameter (default 1.0).
    seed : int | None
        Random seed.
    query_weights : dict[str, float] | None
        Intent distribution weights.
    knapsack_method : str
        ``"dp"`` or ``"greedy"``.
    """

    def __init__(
        self,
        prompts: list[Prompt] | None = None,
        budget: int = 300,
        alpha: float = 1.0,
        seed: int | None = 42,
        query_weights: dict[str, float] | None = None,
        knapsack_method: str = "dp",
    ) -> None:
        self.prompts = prompts or load_prompt_pool()
        self.budget = budget
        self.knapsack_method = knapsack_method

        self.embedder = QueryEmbedder()
        self.agent = LinUCBAgent(
            arm_ids=[p.id for p in self.prompts],
            d=self.embedder.dim(),
            alpha=alpha,
        )
        self.query_dist = QueryDistribution(weights=query_weights, seed=seed)
        self._np_rng = np.random.default_rng(seed)
        self._step_count = 0

    def step(self) -> SimulationStep:
        """Run one step: sample query → select → reward → update."""
        query, intent = self.query_dist.sample()
        context = self.embedder.embed(query)

        # Score all arms and select under budget.
        ucb_scores = self.agent.score_all(context)
        knapsack_result = select_prompts(
            self.prompts,
            ucb_scores,
            self.budget,
            method=self.knapsack_method,
        )

        # Simulated reward.
        sim_quality = simulated_quality(
            knapsack_result.selected,
            intent,
            rng=self._np_rng,
        )
        # Build a fake response text based on quality to feed reward module.
        fake_response = f"Simulated response for intent={intent} quality={sim_quality:.2f}"
        reward_breakdown = compute_simulated_reward(
            response_text=fake_response,
            query=query,
            total_token_cost=knapsack_result.total_token_cost,
            budget=self.budget,
            latency_s=0.5,
        )

        # Use simulated quality as the primary reward signal for bandit update.
        reward_value = sim_quality

        # Update bandit for each selected arm.
        for p in knapsack_result.selected:
            self.agent.update(p.id, context, reward_value)

        result = SimulationStep(
            step=self._step_count,
            query=query,
            intent=intent,
            selection=knapsack_result,
            reward=reward_breakdown,
        )
        self._step_count += 1
        return result

    def run(self, n_steps: int) -> list[SimulationStep]:
        """Run *n_steps* simulation steps and return all results."""
        return [self.step() for _ in range(n_steps)]

    def reset(self) -> None:
        """Reset the agent (but keep the environment config)."""
        self.agent.reset()
        self._step_count = 0
