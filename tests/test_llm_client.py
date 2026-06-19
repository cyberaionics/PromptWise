"""Tests for promptwise.llm_client module.

These tests verify the module structure and helper logic WITHOUT making
real API calls.  The Anthropic client is not instantiated unless an API
key is present.
"""

import pytest

from promptwise.prompts import Prompt


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_llm_client_import():
    """Module and key classes are importable."""
    from promptwise.llm_client import LLMClient, LLMResponse
    assert LLMClient is not None
    assert LLMResponse is not None


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_total_tokens(self):
        from promptwise.llm_client import LLMResponse

        resp = LLMResponse(
            text="hello",
            input_tokens=100,
            output_tokens=50,
            latency_s=1.0,
            model="test",
        )
        assert resp.total_tokens == 150

    def test_fields(self):
        from promptwise.llm_client import LLMResponse

        resp = LLMResponse(
            text="hi",
            input_tokens=10,
            output_tokens=5,
            latency_s=0.5,
            model="test-model",
        )
        assert resp.text == "hi"
        assert resp.model == "test-model"
        assert resp.latency_s == 0.5


# ---------------------------------------------------------------------------
# Prompt assembly helper
# ---------------------------------------------------------------------------


class TestAssemblePromptContext:
    def test_single_prompt(self):
        from promptwise.llm_client import LLMClient

        p = Prompt(id="instr_tone", category="instruction", text="Be polite.", tags=["general_info"])
        result = LLMClient._assemble_prompt_context([p])
        assert "INSTRUCTION" in result
        assert "Be polite." in result
        assert "instr_tone" in result

    def test_multiple_prompts(self):
        from promptwise.llm_client import LLMClient

        prompts = [
            Prompt(id="a", category="instruction", text="Text A", tags=["t"]),
            Prompt(id="b", category="example", text="Text B", tags=["t"]),
        ]
        result = LLMClient._assemble_prompt_context(prompts)
        assert "Text A" in result
        assert "Text B" in result
        assert "INSTRUCTION" in result
        assert "EXAMPLE" in result


# ---------------------------------------------------------------------------
# Client instantiation
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_no_key_raises(self, monkeypatch):
        """LLMClient should raise if no API key is available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No Anthropic API key"):
            from promptwise.llm_client import LLMClient
            LLMClient(api_key="")
