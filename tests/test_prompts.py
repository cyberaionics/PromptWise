"""Tests for promptwise.prompts module."""

from promptwise.prompts import Prompt, PromptCategory, load_prompt_pool, count_tokens


def test_prompts_import():
    """Smoke test: module is importable."""
    assert Prompt is not None
    assert load_prompt_pool is not None


class TestCountTokens:
    """Verify the tiktoken-based token counter."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_string(self):
        # A simple sentence should yield a positive token count.
        tokens = count_tokens("Hello, world!")
        assert tokens > 0

    def test_deterministic(self):
        text = "Book an appointment with Dr. Sharma on Tuesday."
        assert count_tokens(text) == count_tokens(text)


class TestPromptDataclass:
    """Verify the Prompt dataclass behaviour."""

    def test_token_cost_computed_automatically(self):
        p = Prompt(
            id="test_1",
            category="instruction",
            text="Short instruction.",
            tags=["general_info"],
        )
        assert p.token_cost == count_tokens("Short instruction.")
        assert p.token_cost > 0

    def test_category_literal(self):
        for cat in ("instruction", "example", "domain_context"):
            p = Prompt(id="t", category=cat, text="x", tags=[])
            assert p.category == cat


class TestLoadPromptPool:
    """Validate the full prompt pool returned by load_prompt_pool()."""

    def setup_method(self):
        self.pool = load_prompt_pool()

    def test_pool_size(self):
        """Pool should contain between 18 and 25 prompts."""
        assert 18 <= len(self.pool) <= 25

    def test_unique_ids(self):
        ids = [p.id for p in self.pool]
        assert len(ids) == len(set(ids)), f"Duplicate ids found: {ids}"

    def test_valid_categories(self):
        valid = {"instruction", "example", "domain_context"}
        for p in self.pool:
            assert p.category in valid, (
                f"Prompt '{p.id}' has invalid category '{p.category}'"
            )

    def test_all_categories_present(self):
        categories = {p.category for p in self.pool}
        assert categories == {"instruction", "example", "domain_context"}

    def test_token_cost_positive(self):
        for p in self.pool:
            assert p.token_cost > 0, (
                f"Prompt '{p.id}' has token_cost={p.token_cost}"
            )

    def test_token_cost_matches_text(self):
        """token_cost must equal the real tiktoken count of the text."""
        for p in self.pool:
            expected = count_tokens(p.text)
            assert p.token_cost == expected, (
                f"Prompt '{p.id}': token_cost={p.token_cost} != "
                f"count_tokens={expected}"
            )

    def test_token_cost_range(self):
        """Pool should have prompts spanning a wide token-cost range."""
        costs = [p.token_cost for p in self.pool]
        assert min(costs) < 30, f"Minimum cost {min(costs)} not low enough"
        assert max(costs) > 80, f"Maximum cost {max(costs)} not high enough"

    def test_tags_non_empty(self):
        for p in self.pool:
            assert len(p.tags) > 0, f"Prompt '{p.id}' has no tags"

    def test_text_non_empty(self):
        for p in self.pool:
            assert len(p.text.strip()) > 0, f"Prompt '{p.id}' has empty text"
