"""Thin wrapper around the Anthropic Messages API.

Provides :class:`LLMClient` which assembles selected prompts + user query
into a single API call and returns the response along with usage metadata
(tokens consumed, latency).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

from dotenv import load_dotenv

from promptwise.prompts import Prompt

# Load .env so ANTHROPIC_API_KEY is available.
load_dotenv()


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Structured response from a single LLM call.

    Attributes
    ----------
    text : str
        The assistant's reply.
    input_tokens : int
        Tokens consumed by the input (system + user messages).
    output_tokens : int
        Tokens in the generated response.
    latency_s : float
        Wall-clock seconds for the API call.
    model : str
        Model identifier used.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Client for calling Anthropic's Claude API.

    Parameters
    ----------
    model : str
        Anthropic model name (default ``"claude-sonnet-4-20250514"``).
    max_tokens : int
        Maximum output tokens (default 512).
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in your "
                "environment or pass api_key= to LLMClient."
            )

        # Lazy import so tests that don't call the API don't need the SDK.
        import anthropic

        self._client = anthropic.Anthropic(api_key=resolved_key)

    # -- core API -----------------------------------------------------------

    def generate(
        self,
        user_query: str,
        selected_prompts: Sequence[Prompt] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a query (with optional prompt context) to Claude.

        Parameters
        ----------
        user_query : str
            The patient / end-user message.
        selected_prompts : Sequence[Prompt] | None
            Prompts selected by the bandit + knapsack pipeline.
            They are assembled into a ``system`` message.
        system : str | None
            Optional explicit system message.  If both *system* and
            *selected_prompts* are given they are concatenated.
        temperature : float
            Sampling temperature (default 0.3 for determinism).

        Returns
        -------
        LLMResponse
        """
        # Build system message from selected prompts.
        parts: list[str] = []
        if system:
            parts.append(system)
        if selected_prompts:
            parts.append(self._assemble_prompt_context(selected_prompts))
        system_text = "\n\n".join(parts) if parts else None

        # Call Anthropic API.
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user_query}],
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text

        response = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0

        text = response.content[0].text if response.content else ""
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            model=self.model,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _assemble_prompt_context(prompts: Sequence[Prompt]) -> str:
        """Join selected prompts into a single system context block."""
        sections: list[str] = []
        for p in prompts:
            header = f"[{p.category.upper()} — {p.id}]"
            sections.append(f"{header}\n{p.text}")
        return "\n\n".join(sections)
