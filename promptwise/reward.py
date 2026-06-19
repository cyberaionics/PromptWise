"""Reward computation for the contextual-bandit feedback loop.

Defines reward signals that combine **response quality**, **token
efficiency**, and **latency** into a single scalar reward in [0, 1].

Components
----------
* **Quality score**: keyword/heuristic-based check that the response
  addresses the query intent (booking confirmation, cancellation ack, etc.).
* **Token efficiency**: ratio of budget actually used — lower is better.
* **Latency penalty**: mild penalty for slow responses.

These are combined via a weighted sum to produce the final reward.
"""

from __future__ import annotations

from dataclasses import dataclass

from promptwise.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Detailed decomposition of a single reward signal.

    Attributes
    ----------
    quality : float
        Response quality score in [0, 1].
    token_efficiency : float
        Token efficiency score in [0, 1].  1 = used few tokens.
    latency_score : float
        Latency score in [0, 1].  1 = fast response.
    total : float
        Weighted combination in [0, 1].
    """

    quality: float
    token_efficiency: float
    latency_score: float
    total: float


# ---------------------------------------------------------------------------
# Quality heuristics
# ---------------------------------------------------------------------------

# Keywords that indicate the response is addressing common intents.
_QUALITY_KEYWORDS: dict[str, list[str]] = {
    "booking": [
        "appointment", "booked", "scheduled", "confirmed", "slot",
        "date", "time", "doctor",
    ],
    "cancellation": [
        "cancel", "cancelled", "cancellation", "removed",
    ],
    "availability": [
        "available", "availability", "open", "slot", "schedule",
    ],
    "insurance": [
        "insurance", "coverage", "plan", "accepted", "accept",
    ],
    "emergency": [
        "emergency", "911", "urgent", "immediately", "call",
    ],
    "general_info": [
        "hours", "location", "address", "phone", "clinic",
    ],
}


def quality_score(response_text: str, query: str) -> float:
    """Heuristic quality score for a response given the original query.

    Checks how many relevant keyword groups are hit.  Returns a float
    in [0, 1].

    Parameters
    ----------
    response_text : str
        The LLM's generated answer.
    query : str
        The original user query.
    """
    if not response_text.strip():
        return 0.0

    response_lower = response_text.lower()
    query_lower = query.lower()

    # Determine which intent groups are relevant to the query.
    relevant_groups: list[str] = []
    for group, keywords in _QUALITY_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            relevant_groups.append(group)

    # Fallback: if no group matched, use all groups.
    if not relevant_groups:
        relevant_groups = list(_QUALITY_KEYWORDS.keys())

    # Score: fraction of relevant groups whose keywords appear in response.
    hits = 0
    for group in relevant_groups:
        keywords = _QUALITY_KEYWORDS[group]
        if any(kw in response_lower for kw in keywords):
            hits += 1

    # Base quality from keyword coverage.
    keyword_score = hits / len(relevant_groups) if relevant_groups else 0.5

    # Bonus for non-trivially long responses (at least 20 chars).
    length_bonus = min(1.0, len(response_text.strip()) / 100.0) * 0.2

    return min(1.0, keyword_score * 0.8 + length_bonus)


# ---------------------------------------------------------------------------
# Token efficiency
# ---------------------------------------------------------------------------


def token_efficiency_score(
    input_tokens: int,
    output_tokens: int,
    budget: int,
) -> float:
    """Score token usage efficiency.

    A lower total token count relative to *budget* yields a higher score.

    Returns a float in [0, 1].
    """
    total = input_tokens + output_tokens
    if budget <= 0:
        return 0.5
    ratio = total / budget
    # Sigmoid-like mapping: ratio=1 → ~0.5, ratio<1 → higher, ratio>1 → lower.
    return max(0.0, min(1.0, 1.0 - 0.5 * ratio))


# ---------------------------------------------------------------------------
# Latency score
# ---------------------------------------------------------------------------


def latency_score(latency_s: float, target_s: float = 3.0) -> float:
    """Score response latency.

    Returns 1.0 for instant responses, decaying towards 0 as latency
    exceeds *target_s*.

    Parameters
    ----------
    latency_s : float
        Observed latency in seconds.
    target_s : float
        Acceptable latency threshold (default 3 s).
    """
    if latency_s <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (latency_s / (2.0 * target_s))))


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------


def compute_reward(
    response: LLMResponse,
    query: str,
    budget: int,
    *,
    w_quality: float = 0.6,
    w_token: float = 0.25,
    w_latency: float = 0.15,
) -> RewardBreakdown:
    """Compute the composite reward for a single interaction.

    Parameters
    ----------
    response : LLMResponse
        The LLM response (text + usage + latency).
    query : str
        The original user query.
    budget : int
        Token budget that was enforced.
    w_quality, w_token, w_latency : float
        Weights for the three components (must sum to 1).

    Returns
    -------
    RewardBreakdown
    """
    q = quality_score(response.text, query)
    t = token_efficiency_score(response.input_tokens, response.output_tokens, budget)
    l = latency_score(response.latency_s)

    total = w_quality * q + w_token * t + w_latency * l
    return RewardBreakdown(quality=q, token_efficiency=t, latency_score=l, total=total)


def compute_simulated_reward(
    response_text: str,
    query: str,
    total_token_cost: int,
    budget: int,
    latency_s: float = 0.5,
    *,
    w_quality: float = 0.6,
    w_token: float = 0.25,
    w_latency: float = 0.15,
) -> RewardBreakdown:
    """Compute reward without a real LLMResponse (for simulation)."""
    q = quality_score(response_text, query)
    t = token_efficiency_score(total_token_cost, 0, budget)
    l = latency_score(latency_s)

    total = w_quality * q + w_token * t + w_latency * l
    return RewardBreakdown(quality=q, token_efficiency=t, latency_score=l, total=total)
