# tests/test_registry_and_integration.py
"""
Comprehensive tests for Issue #13: Add more tests.

Tests cover:
- Registry functionality (personalities as objects and strings)
- Adding to the registry
- Specific personality behaviors
- message_every smoothing and averages
"""

import pytest
import torch

from emotigrad import EmotionalOptimizer
from emotigrad.personalities import (
    QuietPersonality,
    get_personality,
    list_personalities,
    register_personality,
    sassy,
    wholesome,
)


class TestRegistryWithStrings:
    """Test that personalities can be provided as registry strings."""

    def test_optimizer_accepts_string_personality(self):
        """EmotionalOptimizer should accept a string personality name."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        emo_opt = EmotionalOptimizer(optimizer, personality="wholesome")

        # Should resolve the string to the actual personality
        assert callable(emo_opt.personality)

    def test_optimizer_resolves_all_registered_personalities(self):
        """All registered personality strings should be resolvable."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for name in list_personalities():
            emo_opt = EmotionalOptimizer(optimizer, personality=name)
            assert callable(emo_opt.personality)

    def test_optimizer_raises_on_unknown_personality_string(self):
        """Unknown personality string should raise KeyError."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with pytest.raises(KeyError):
            EmotionalOptimizer(optimizer, personality="nonexistent_personality")


class TestRegistryWithObjects:
    """Test that personalities can be provided as callable objects."""

    def test_optimizer_accepts_function_personality(self):
        """EmotionalOptimizer should accept a function as personality."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def custom_personality(loss, prev_loss, step):
            return "Custom message"

        emo_opt = EmotionalOptimizer(optimizer, personality=custom_personality)

        assert emo_opt.personality == custom_personality

    def test_optimizer_accepts_class_instance_personality(self):
        """EmotionalOptimizer should accept a class instance as personality."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        quiet = QuietPersonality(every_n_steps=5)
        emo_opt = EmotionalOptimizer(optimizer, personality=quiet)

        assert emo_opt.personality == quiet

    def test_optimizer_accepts_lambda_personality(self):
        """EmotionalOptimizer should accept a lambda as personality."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        emo_opt = EmotionalOptimizer(
            optimizer, personality=lambda loss, _p, _s: f"Lambda: {loss}"
        )

        assert callable(emo_opt.personality)


class TestAddingToRegistry:
    """Test adding custom personalities to the registry."""

    def test_register_new_personality(self):
        """Should be able to register a new personality."""

        def my_custom_personality(loss, prev_loss, step):
            return "My custom message"

        # Use a unique name to avoid conflicts
        name = "test_custom_personality_unique"
        register_personality(name, my_custom_personality)

        assert name in list_personalities()
        assert get_personality(name) == my_custom_personality

    def test_register_class_personality(self):
        """Should be able to register a class-based personality."""

        class MyClassPersonality:
            def __call__(self, loss, prev_loss, step):
                return "Class message"

        name = "test_class_personality_unique"
        instance = MyClassPersonality()
        register_personality(name, instance)

        assert name in list_personalities()
        retrieved = get_personality(name)
        assert callable(retrieved)

    def test_cannot_overwrite_existing_without_flag(self):
        """Should raise ValueError when overwriting without flag."""

        def dummy(loss, prev_loss, step):
            return "Dummy"

        # wholesome is already registered
        with pytest.raises(ValueError):
            register_personality("wholesome", dummy, overwrite=False)

    def test_can_overwrite_with_flag(self):
        """Should allow overwriting with overwrite=True."""
        original = get_personality("sassy")

        def new_sassy(loss, prev_loss, step):
            return "New sassy"

        register_personality("sassy", new_sassy, overwrite=True)

        assert get_personality("sassy") == new_sassy

        # Restore original
        register_personality("sassy", original, overwrite=True)


class TestSpecificPersonalities:
    """Test specific personality behaviors."""

    def test_wholesome_first_step(self):
        """Wholesome should return encouraging message on first step."""
        result = wholesome(loss=1.0, prev_loss=None, step=1)
        assert "✨" in result or "Let's" in result.lower()
        assert "1.0" in result

    def test_wholesome_loss_decrease(self):
        """Wholesome should celebrate loss decrease."""
        result = wholesome(loss=0.5, prev_loss=1.0, step=2)
        assert "💖" in result or "Nice" in result

    def test_wholesome_loss_increase(self):
        """Wholesome should be encouraging on loss increase."""
        result = wholesome(loss=1.5, prev_loss=1.0, step=2)
        assert "🌱" in result or "okay" in result.lower()

    def test_sassy_first_step(self):
        """Sassy should be dismissive on first step."""
        result = sassy(loss=1.0, prev_loss=None, step=1)
        assert "😒" in result or "Fine" in result

    def test_sassy_loss_decrease(self):
        """Sassy should be grudgingly approving on loss decrease."""
        result = sassy(loss=0.5, prev_loss=1.0, step=2)
        assert "👏" in result or "About time" in result

    def test_sassy_loss_increase(self):
        """Sassy should be sarcastic on loss increase."""
        result = sassy(loss=1.5, prev_loss=1.0, step=2)
        assert "🙄" in result or "Bold" in result.lower()

    def test_quiet_personality_respects_interval(self):
        """QuietPersonality should only output at specified intervals."""
        quiet = QuietPersonality(every_n_steps=5)

        # Steps 1-4 should return None
        for step in range(1, 5):
            assert quiet(loss=1.0, prev_loss=None, step=step) is None

        # Step 5 should return a message
        result = quiet(loss=1.0, prev_loss=None, step=5)
        assert result is not None
        assert "Step 5" in result


class TestMessageEverySmoothing:
    """Test message_every parameter and loss averaging."""

    def test_message_every_counts_steps_correctly(self):
        """Should emit messages at correct intervals."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []

        def capture(msg):
            messages.append(msg)

        def echo(loss, prev, step):
            return f"step={step}"

        emo_opt = EmotionalOptimizer(
            optimizer,
            personality=echo,
            print_fn=capture,
            message_every=2,
        )

        # Run 6 steps
        for _ in range(6):
            emo_opt.step(loss=1.0)

        # Should have messages at steps 2, 4, 6
        assert len(messages) == 3

    def test_message_every_averages_losses(self):
        """Loss should be averaged over the message_every window."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        captured_losses = []

        def capture_loss(loss, prev, step):
            captured_losses.append(loss)
            return "msg"

        emo_opt = EmotionalOptimizer(
            optimizer,
            personality=capture_loss,
            print_fn=lambda x: None,
            message_every=3,
        )

        # Provide losses: 1, 2, 3 -> avg = 2
        for loss in [1.0, 2.0, 3.0]:
            emo_opt.step(loss=loss)

        assert len(captured_losses) == 1
        assert abs(captured_losses[0] - 2.0) < 0.01

    def test_message_every_tracks_previous_average(self):
        """Previous loss should be the previous window's average."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        prev_losses = []

        def capture_prev(loss, prev, step):
            prev_losses.append(prev)
            return "msg"

        emo_opt = EmotionalOptimizer(
            optimizer,
            personality=capture_prev,
            print_fn=lambda x: None,
            message_every=2,
        )

        # Window 1: [1, 3] avg=2
        # Window 2: [5, 7] avg=6
        for loss in [1.0, 3.0, 5.0, 7.0]:
            emo_opt.step(loss=loss)

        # First window: prev is None
        assert prev_losses[0] is None
        # Second window: prev is 2.0
        assert abs(prev_losses[1] - 2.0) < 0.01

    def test_message_every_zero_disables_messages(self):
        """message_every=0 should disable all messages."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []

        emo_opt = EmotionalOptimizer(
            optimizer,
            personality=lambda _l, _p, _s: "msg",
            print_fn=messages.append,
            message_every=0,
        )

        for _ in range(10):
            emo_opt.step(loss=1.0)

        assert len(messages) == 0

    def test_message_every_one_emits_every_step(self):
        """message_every=1 should emit a message every step."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []

        emo_opt = EmotionalOptimizer(
            optimizer,
            personality=lambda _l, _p, _s: "msg",
            print_fn=messages.append,
            message_every=1,
        )

        for _ in range(5):
            emo_opt.step(loss=1.0)

        assert len(messages) == 5


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_full_training_loop_with_string_personality(self):
        """Test a complete training loop with string personality."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []
        emo_opt = EmotionalOptimizer(
            optimizer,
            personality="wholesome",
            print_fn=messages.append,
            message_every=1,
        )

        x = torch.randn(4, 2)
        y = torch.randn(4, 1)

        for _ in range(3):
            emo_opt.zero_grad()
            pred = model(x)
            loss = (pred - y).pow(2).mean()
            loss.backward()
            emo_opt.step(loss=loss.item())

        assert len(messages) == 3
        assert emo_opt.step_count == 3

    def test_switching_personalities_mid_training(self):
        """Test changing personality during training."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []
        emo_opt = EmotionalOptimizer(
            optimizer,
            personality="wholesome",
            print_fn=messages.append,
        )

        # Train with wholesome
        emo_opt.step(loss=1.0)

        # Switch to sassy
        emo_opt.personality = get_personality("sassy")
        emo_opt.step(loss=0.5)

        # Should have different message styles
        assert len(messages) == 2

    def test_disabled_optimizer_produces_no_messages(self):
        """Disabled optimizer should not produce any messages."""
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []
        emo_opt = EmotionalOptimizer(
            optimizer,
            personality="wholesome",
            print_fn=messages.append,
            enabled=False,
        )

        for _ in range(10):
            emo_opt.step(loss=1.0)

        assert len(messages) == 0
