# tests/test_personalities.py
"""Tests for the built-in personalities."""

import pytest

from emotigrad.personalities import (
    academic,
    arrogant,
    chaotic,
    get_personality,
    hype,
    list_personalities,
    nervous,
    pirate,
    register_personality,
    sassy,
    tired,
    wholesome,
    zen,
)


class TestPersonalityFunctions:
    """Test individual personality functions."""

    @pytest.mark.parametrize(
        "personality_fn",
        [wholesome, sassy, nervous, arrogant, tired, hype, academic, pirate, zen],
    )
    def test_personality_returns_string_on_first_step(self, personality_fn):
        """All personalities should return a string message on the first step."""
        result = personality_fn(loss=1.0, prev_loss=None, step=1)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "personality_fn",
        [wholesome, sassy, nervous, arrogant, tired, hype, academic, pirate, zen],
    )
    def test_personality_returns_string_on_loss_decrease(self, personality_fn):
        """All personalities should return a string when loss decreases."""
        result = personality_fn(loss=0.5, prev_loss=1.0, step=2)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "personality_fn",
        [wholesome, sassy, nervous, arrogant, tired, hype, academic, pirate, zen],
    )
    def test_personality_returns_string_on_loss_increase(self, personality_fn):
        """All personalities should return a string when loss increases."""
        result = personality_fn(loss=1.5, prev_loss=1.0, step=2)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "personality_fn",
        [wholesome, sassy, nervous, arrogant, tired, hype, academic, pirate, zen],
    )
    def test_personality_handles_equal_loss(self, personality_fn):
        """All personalities should handle equal loss (may return None or string)."""
        result = personality_fn(loss=1.0, prev_loss=1.0, step=2)
        # Result can be None or a string
        assert result is None or isinstance(result, str)


class TestChaoticPersonality:
    """Special tests for chaotic personality due to its random nature."""

    def test_chaotic_returns_different_messages(self):
        """Chaotic personality should have variety in its messages."""
        # Run multiple times to increase chance of seeing variation
        messages = set()
        for _ in range(20):
            msg = chaotic(loss=0.5, prev_loss=1.0, step=2)
            messages.add(msg)

        # We expect at least 2 different messages out of 20 attempts
        # (there are 3 possible messages for loss decrease)
        assert len(messages) >= 2


class TestAcademicPersonality:
    """Special tests for academic personality to verify statistical output."""

    def test_academic_includes_delta(self):
        """Academic personality should include delta in loss change messages."""
        result = academic(loss=0.8, prev_loss=1.0, step=2)
        assert "Δ" in result or "delta" in result.lower() or "-0.2" in result

    def test_academic_includes_percentage(self):
        """Academic personality should include percentage change."""
        result = academic(loss=0.8, prev_loss=1.0, step=2)
        assert "%" in result


class TestRegistry:
    """Tests for the personality registry."""

    def test_list_personalities_includes_new_personalities(self):
        """The registry should include all new personalities."""
        available = list_personalities()
        expected = [
            "wholesome",
            "sassy",
            "quiet",
            "nervous",
            "chaotic",
            "arrogant",
            "tired",
            "hype",
            "academic",
            "pirate",
            "zen",
        ]
        for name in expected:
            assert name in available, f"'{name}' should be in the registry"

    def test_get_personality_returns_callable(self):
        """get_personality should return callable personalities."""
        for name in list_personalities():
            personality = get_personality(name)
            assert callable(personality)

    def test_get_personality_case_insensitive(self):
        """get_personality should be case-insensitive."""
        assert get_personality("NERVOUS") == get_personality("nervous")
        assert get_personality("Academic") == get_personality("academic")

    def test_get_personality_raises_on_unknown(self):
        """get_personality should raise KeyError for unknown personalities."""
        with pytest.raises(KeyError):
            get_personality("nonexistent_personality")

    def test_register_personality(self):
        """Test registering a custom personality."""

        def custom_personality(loss, prev_loss, step):
            return "Custom message"

        register_personality("custom_test", custom_personality)

        assert "custom_test" in list_personalities()
        assert get_personality("custom_test") == custom_personality

    def test_register_personality_prevents_overwrite_by_default(self):
        """Registering an existing name should raise ValueError."""
        with pytest.raises(ValueError):
            register_personality("wholesome", lambda _l, _p, _s: "test")

    def test_register_personality_allows_overwrite_when_specified(self):
        """Overwrite should work when explicitly allowed."""

        def new_wholesome(loss, prev_loss, step):
            return "New wholesome"

        register_personality("wholesome", new_wholesome, overwrite=True)
        assert get_personality("wholesome") == new_wholesome

        # Restore original
        from emotigrad.personalities import wholesome as original_wholesome

        register_personality("wholesome", original_wholesome, overwrite=True)
