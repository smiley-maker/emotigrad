# src/emotigrad/personalities.py
from __future__ import annotations

from typing import Dict, List, Optional

from .types import Personality


# --- Built-in personalities -------------------------------------------------
def _default_personality(
    loss: float,
    prev_loss: Optional[float],
    step: int,
) -> Optional[str]:
    """Very minimal 'wholesome' personality for early versions.

    You can expand this or move it into a dedicated personalities module later.
    """
    if prev_loss is None:
        return f"✨ Starting our journey! Initial loss: {loss:.4f}"

    if loss < prev_loss:
        return f"💖 Nice! Loss improved from {prev_loss:.4f} to {loss:.4f}."

    if loss > prev_loss:
        return (
            f"🌱 It's okay! Loss went from {prev_loss:.4f} to {loss:.4f}. "
            "Learning isn't always linear."
        )

    return None


def wholesome(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    if prev_loss is None:
        return f"✨ Let's get started! Initial loss: {loss:.4f}"

    if loss < prev_loss:
        return f"💖 Nice! Loss improved from {prev_loss:.4f} to {loss:.4f}."

    if loss > prev_loss:
        return (
            f"🌱 It's okay! Loss went from {prev_loss:.4f} to {loss:.4f}. "
            "Learning isn't always monotonic."
        )

    return None  # no message if unchanged


def sassy(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    if prev_loss is None:
        return "😒 Fine, let's see what you've got."

    if loss > prev_loss:
        return f"🙄 Bold move: loss got worse ({prev_loss:.4f} → {loss:.4f})."

    if loss < prev_loss:
        return f"👏 About time: {prev_loss:.4f} → {loss:.4f}."

    return "🤨 Exactly the same? Interesting choice."


class QuietPersonality:
    """Example of a personality implemented as a class with state."""

    def __init__(self, every_n_steps: int = 10) -> None:
        self.every_n_steps = every_n_steps

    def __call__(self, loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
        if step % self.every_n_steps != 0:
            return None
        return f"🔎 Step {step}: current loss {loss:.4f}"


# --- Registry ---------------------------------------------------------------


_PERSONALITY_REGISTRY: Dict[str, Personality] = {
    "wholesome": wholesome,
    "sassy": sassy,
    "quiet": QuietPersonality(),  # instance is fine; it's still callable
}


def register_personality(
    name: str,
    personality: Personality,
    *,
    overwrite: bool = False,
) -> None:
    """Register a new personality under a string name.

    Users can call this to add their own personalities.
    """
    key = name.lower()
    if not overwrite and key in _PERSONALITY_REGISTRY:
        raise ValueError(f"Personality '{name}' is already registered.")
    _PERSONALITY_REGISTRY[key] = personality


def get_personality(name: str) -> Personality:
    """Look up a personality by name, raising KeyError if not found."""
    key = name.lower()
    try:
        return _PERSONALITY_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PERSONALITY_REGISTRY.keys()))
        raise KeyError(f"Unknown personality '{name}'. Available: {available}") from exc


def list_personalities() -> List[str]:
    """Return a sorted list of available personality names."""
    return sorted(_PERSONALITY_REGISTRY.keys())
