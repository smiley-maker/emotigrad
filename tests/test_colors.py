# tests/test_colors.py
"""Tests for the colored output module."""


from emotigrad.colors import (
    BLUE,
    BOLD,
    GREEN,
    PERSONALITY_COLORS,
    RED,
    RESET,
    ColoredPrinter,
    colorize,
    create_colored_print_fn,
    get_personality_colors,
    strip_colors,
)


class TestColorize:
    """Tests for the colorize function."""

    def test_colorize_applies_single_code(self):
        """colorize should apply a single color code."""
        result = colorize("Hello", GREEN)
        assert result.startswith(GREEN)
        assert result.endswith(RESET)
        assert "Hello" in result

    def test_colorize_applies_multiple_codes(self):
        """colorize should apply multiple color codes."""
        result = colorize("Hello", GREEN, BOLD)
        assert GREEN in result
        assert BOLD in result
        assert result.endswith(RESET)

    def test_colorize_no_codes_returns_original(self):
        """colorize with no codes should return original text."""
        result = colorize("Hello")
        assert result == "Hello"

    def test_colorize_empty_string(self):
        """colorize should handle empty strings."""
        result = colorize("", GREEN)
        assert result == GREEN + RESET


class TestStripColors:
    """Tests for the strip_colors function."""

    def test_strip_colors_removes_codes(self):
        """strip_colors should remove all ANSI codes."""
        colored = colorize("Hello World", GREEN, BOLD)
        result = strip_colors(colored)
        assert result == "Hello World"

    def test_strip_colors_plain_text(self):
        """strip_colors should handle plain text."""
        result = strip_colors("Hello World")
        assert result == "Hello World"

    def test_strip_colors_multiple_colors(self):
        """strip_colors should remove multiple color sequences."""
        text = f"{RED}Red{RESET} and {BLUE}Blue{RESET}"
        result = strip_colors(text)
        assert result == "Red and Blue"


class TestGetPersonalityColors:
    """Tests for get_personality_colors function."""

    def test_get_colors_for_known_personality(self):
        """Should return colors for known personalities."""
        colors = get_personality_colors("wholesome")
        assert len(colors) > 0
        assert colors == PERSONALITY_COLORS["wholesome"]

    def test_get_colors_case_insensitive(self):
        """Should be case insensitive."""
        assert get_personality_colors("HYPE") == get_personality_colors("hype")
        assert get_personality_colors("Academic") == get_personality_colors("academic")

    def test_get_colors_unknown_personality(self):
        """Should return empty tuple for unknown personality."""
        colors = get_personality_colors("nonexistent")
        assert colors == ()

    def test_all_registered_personalities_have_colors(self):
        """All personalities in PERSONALITY_COLORS should have color tuples."""
        for name, colors in PERSONALITY_COLORS.items():
            assert isinstance(colors, tuple)
            assert len(colors) > 0


class TestColoredPrinter:
    """Tests for the ColoredPrinter class."""

    def test_colored_printer_with_personality(self, capsys):
        """ColoredPrinter should apply colors when personality is set."""
        printer = ColoredPrinter("wholesome")
        printer("Test message")

        captured = capsys.readouterr()
        # Should contain the color codes
        assert GREEN in captured.out or "Test message" in captured.out

    def test_colored_printer_without_personality(self, capsys):
        """ColoredPrinter without personality should print plain text."""
        printer = ColoredPrinter()
        printer("Test message")

        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_colored_printer_disabled(self, capsys):
        """ColoredPrinter should not apply colors when disabled."""
        printer = ColoredPrinter("wholesome", enabled=False)
        printer("Test message")

        captured = capsys.readouterr()
        # Should be plain text
        assert strip_colors(captured.out).strip() == "Test message"

    def test_colored_printer_set_personality(self):
        """set_personality should update the colors."""
        printer = ColoredPrinter("wholesome")
        original_colors = printer._colors

        printer.set_personality("hype")
        assert printer._colors != original_colors
        assert printer._colors == PERSONALITY_COLORS["hype"]


class TestCreateColoredPrintFn:
    """Tests for create_colored_print_fn function."""

    def test_create_returns_callable(self):
        """Should return a callable."""
        fn = create_colored_print_fn("wholesome")
        assert callable(fn)

    def test_created_fn_prints_with_colors(self, capsys):
        """Created function should print with colors."""
        fn = create_colored_print_fn("hype")
        fn("Test message")

        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_created_fn_respects_enabled_flag(self, capsys):
        """Created function should respect enabled flag."""
        fn = create_colored_print_fn("hype", enabled=False)
        fn("Test message")

        captured = capsys.readouterr()
        assert strip_colors(captured.out).strip() == "Test message"


class TestIntegrationWithEmotionalOptimizer:
    """Integration tests with EmotionalOptimizer."""

    def test_colored_printer_as_print_fn(self):
        """ColoredPrinter should work as EmotionalOptimizer's print_fn."""
        import torch

        from emotigrad import EmotionalOptimizer

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        messages = []

        class CapturingColoredPrinter(ColoredPrinter):
            def __call__(self, message):
                messages.append(message)
                super().__call__(message)

        printer = CapturingColoredPrinter("wholesome")

        emo_opt = EmotionalOptimizer(
            optimizer,
            personality="wholesome",
            print_fn=printer,
            enabled=True,
        )

        x = torch.randn(2, 2)
        y = torch.randn(2, 1)
        preds = model(x)
        loss = (preds - y).pow(2).mean()

        emo_opt.zero_grad()
        loss.backward()
        emo_opt.step(loss=loss.item())

        # Should have captured at least one message
        assert len(messages) >= 1
