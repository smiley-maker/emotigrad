# src/emotigrad/colors.py
"""Colored console output support for emotigrad.

This module provides ANSI color codes and utility functions for
colorful console output. It includes personality-specific color schemes.
"""

from __future__ import annotations

from typing import Optional

# ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"

# Foreground colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright foreground colors
BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Background colors
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"


def colorize(text: str, *codes: str) -> str:
    """Apply ANSI color codes to text.

    Parameters
    ----------
    text : str
        The text to colorize.
    *codes : str
        ANSI color codes to apply.

    Returns
    -------
    str
        The colorized text with reset code at the end.

    Examples
    --------
    >>> colorize("Hello", GREEN, BOLD)
    '\\033[32m\\033[1mHello\\033[0m'
    """
    if not codes:
        return text
    return "".join(codes) + text + RESET


def strip_colors(text: str) -> str:
    """Remove ANSI color codes from text.

    Parameters
    ----------
    text : str
        Text potentially containing ANSI codes.

    Returns
    -------
    str
        Text with all ANSI codes removed.
    """
    import re

    ansi_escape = re.compile(r"\033\[[0-9;]*m")
    return ansi_escape.sub("", text)


# Personality-specific color schemes
PERSONALITY_COLORS = {
    "wholesome": (GREEN, BOLD),
    "sassy": (MAGENTA, BOLD),
    "quiet": (DIM,),
    "nervous": (YELLOW,),
    "chaotic": (BRIGHT_MAGENTA, BOLD),
    "arrogant": (CYAN, ITALIC),
    "tired": (DIM, ITALIC),
    "hype": (BRIGHT_YELLOW, BOLD),
    "academic": (BLUE,),
    "pirate": (BRIGHT_RED, BOLD),
    "zen": (BRIGHT_CYAN, DIM),
}


def get_personality_colors(personality_name: str) -> tuple:
    """Get the color codes for a specific personality.

    Parameters
    ----------
    personality_name : str
        Name of the personality.

    Returns
    -------
    tuple
        Tuple of ANSI color codes for the personality.
        Returns empty tuple if personality not found.
    """
    return PERSONALITY_COLORS.get(personality_name.lower(), ())


class ColoredPrinter:
    """A printer that applies personality-specific colors to output.

    This can be used as the `print_fn` parameter in EmotionalOptimizer
    to get colored output based on the personality being used.

    Parameters
    ----------
    personality_name : str, optional
        Name of the personality to use for coloring.
        If None, no colors will be applied.
    enabled : bool, optional
        Whether to enable colored output. Default is True.
        Set to False to disable colors (useful for non-TTY output).

    Examples
    --------
    >>> from emotigrad import EmotionalOptimizer
    >>> from emotigrad.colors import ColoredPrinter
    >>>
    >>> printer = ColoredPrinter("hype")
    >>> emo_opt = EmotionalOptimizer(optimizer, personality="hype", print_fn=printer)
    """

    def __init__(
        self,
        personality_name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.personality_name = personality_name
        self.enabled = enabled
        self._colors = (
            get_personality_colors(personality_name) if personality_name else ()
        )

    def __call__(self, message: str) -> None:
        """Print the message with personality-specific colors.

        Parameters
        ----------
        message : str
            The message to print.
        """
        if self.enabled and self._colors:
            print(colorize(message, *self._colors))
        else:
            print(message)

    def set_personality(self, personality_name: str) -> None:
        """Update the personality and associated colors.

        Parameters
        ----------
        personality_name : str
            Name of the new personality.
        """
        self.personality_name = personality_name
        self._colors = get_personality_colors(personality_name)


def create_colored_print_fn(
    personality_name: Optional[str] = None,
    enabled: bool = True,
) -> callable:
    """Create a colored print function for a specific personality.

    This is a convenience function that returns a callable
    suitable for use as EmotionalOptimizer's print_fn.

    Parameters
    ----------
    personality_name : str, optional
        Name of the personality for coloring.
    enabled : bool, optional
        Whether to enable colors.

    Returns
    -------
    callable
        A function that prints with colors.

    Examples
    --------
    >>> from emotigrad import EmotionalOptimizer
    >>> from emotigrad.colors import create_colored_print_fn
    >>>
    >>> emo_opt = EmotionalOptimizer(
    ...     optimizer,
    ...     personality="academic",
    ...     print_fn=create_colored_print_fn("academic"),
    ... )
    """
    return ColoredPrinter(personality_name, enabled)
