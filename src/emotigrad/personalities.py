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

    def __call__(
        self, loss: float, prev_loss: Optional[float], step: int
    ) -> Optional[str]:
        if step % self.every_n_steps != 0:
            return None
        return f"🔎 Step {step}: current loss {loss:.4f}"


def nervous(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """A nervous, anxious personality that worries about everything."""
    if prev_loss is None:
        return (
            f"😰 Oh no, here we go... Initial loss is {loss:.4f}. I hope this works..."
        )

    if loss < prev_loss:
        return (
            f"😅 Phew! Loss dropped from {prev_loss:.4f} to {loss:.4f}. "
            "But what if it goes back up?!"
        )

    if loss > prev_loss:
        return (
            f"😱 I KNEW IT! Loss went up from {prev_loss:.4f} to {loss:.4f}! "
            "Is everything okay?!"
        )

    return f"😬 Loss is exactly the same... {loss:.4f}. That's... concerning?"


def chaotic(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """A chaotic, unpredictable personality that says random things."""
    import random

    if prev_loss is None:
        chaos_starts = [
            f"🎲 CHAOS BEGINS! Loss: {loss:.4f}! LET'S GOOOOO!",
            f"🌪️ *appears from nowhere* Oh, we're training? Loss is {loss:.4f}!",
            f"🃏 Wild card activated! Starting loss: {loss:.4f}!",
        ]
        return random.choice(chaos_starts)

    if loss < prev_loss:
        good_chaos = [
            f"🎉 YEET! {prev_loss:.4f} → {loss:.4f}! *does a backflip*",
            f"🦄 Loss improved! {prev_loss:.4f} → {loss:.4f}! Is this magic?!",
            f"🚀 TO THE MOON! Well, to lower loss at least: {loss:.4f}!",
        ]
        return random.choice(good_chaos)

    if loss > prev_loss:
        bad_chaos = [
            f"💥 BOOM! Loss exploded: {prev_loss:.4f} → {loss:.4f}! EXCITING!",
            f"🎢 Wheeeee! Loss went UP to {loss:.4f}! What a ride!",
            f"🔥 This is fine. Loss: {loss:.4f}. Everything is fine. 🔥",
        ]
        return random.choice(bad_chaos)

    return f"🌀 Time is a flat circle. Loss: {loss:.4f}. Always has been."


def arrogant(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """An arrogant, condescending personality that thinks it knows better."""
    if prev_loss is None:
        return (
            f"🧐 *adjusts monocle* Initial loss of {loss:.4f}? "
            "I suppose that's... acceptable for a beginner."
        )

    if loss < prev_loss:
        return (
            f"😏 Obviously the loss improved ({prev_loss:.4f} → {loss:.4f}). "
            "You're welcome for my guidance."
        )

    if loss > prev_loss:
        return (
            f"🙄 Loss increased to {loss:.4f}? "
            "Perhaps you should have listened to my earlier suggestions."
        )

    return f"😤 No change at {loss:.4f}. Clearly, you need my expertise more than ever."


def tired(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """A tired, exhausted personality that just wants this to be over."""
    if prev_loss is None:
        return f"😴 *yawn* Oh, we're starting? Loss is {loss:.4f}... wake me when it's over."

    if loss < prev_loss:
        return (
            f"😪 Cool, loss went down... {prev_loss:.4f} → {loss:.4f}... "
            "can I go back to sleep now?"
        )

    if loss > prev_loss:
        return (
            f"😩 Ugh, loss went up to {loss:.4f}. Of course it did. "
            "I'm too tired for this."
        )

    return f"💤 Loss is still {loss:.4f}... zzzz..."


def hype(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """An extremely hyped, enthusiastic personality."""
    if prev_loss is None:
        return (
            f"🔥🔥🔥 LET'S GOOOOOO!!! Initial loss: {loss:.4f}! "
            "THIS IS GONNA BE AMAZING!!!"
        )

    if loss < prev_loss:
        return (
            f"🎊🎊🎊 YOOOOO!!! LOSS DROPPED FROM {prev_loss:.4f} TO {loss:.4f}!!! "
            "WE'RE LITERALLY UNSTOPPABLE!!! 💪💪💪"
        )

    if loss > prev_loss:
        return (
            f"😤😤😤 OKAY SO LOSS WENT UP TO {loss:.4f} BUT THAT'S JUST "
            "MAKING THE COMEBACK EVEN MORE EPIC!!! LET'S GO!!!"
        )

    return f"⚡⚡⚡ LOSS HOLDING STEADY AT {loss:.4f}!!! THE TENSION IS REAL!!!"


def academic(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """An academic, research-paper style personality."""
    if prev_loss is None:
        return (
            f"📊 Initial observation: loss function yields {loss:.4f}. "
            "Proceeding with gradient descent optimization."
        )

    delta = loss - prev_loss
    pct_change = (delta / prev_loss) * 100 if prev_loss != 0 else 0

    if loss < prev_loss:
        return (
            f"📈 Statistically significant improvement observed. "
            f"Loss decreased from {prev_loss:.4f} to {loss:.4f} "
            f"(Δ = {delta:.4f}, {pct_change:.2f}% reduction)."
        )

    if loss > prev_loss:
        return (
            f"📉 Note: Loss increased from {prev_loss:.4f} to {loss:.4f} "
            f"(Δ = {delta:.4f}, {abs(pct_change):.2f}% increase). "
            "Further investigation may be warranted."
        )

    return (
        f"📋 No statistically significant change detected. "
        f"Loss remains at {loss:.4f}. Null hypothesis cannot be rejected."
    )


def pirate(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """A pirate-themed personality. Arrr!"""
    if prev_loss is None:
        return f"🏴‍☠️ Ahoy! We be settin' sail! Initial loss be {loss:.4f}, matey!"

    if loss < prev_loss:
        return (
            f"⚓ Shiver me timbers! Loss dropped from {prev_loss:.4f} to {loss:.4f}! "
            "That be treasure, arr!"
        )

    if loss > prev_loss:
        return (
            f"☠️ Blimey! Loss went up to {loss:.4f}! "
            "We be sailin' into rough waters, ye scallywag!"
        )

    return f"🦜 The seas be calm, loss steady at {loss:.4f}. Onwards, me hearties!"


def zen(loss: float, prev_loss: Optional[float], step: int) -> Optional[str]:
    """A zen, peaceful personality focused on the journey."""
    if prev_loss is None:
        return (
            f"🧘 The journey of a thousand gradients begins with a single step. "
            f"Loss: {loss:.4f}."
        )

    if loss < prev_loss:
        return (
            f"☯️ Like water flowing downhill, the loss descends: "
            f"{prev_loss:.4f} → {loss:.4f}. Breathe."
        )

    if loss > prev_loss:
        return (
            f"🍃 The wind sometimes blows against us. Loss: {loss:.4f}. "
            "This too shall pass."
        )

    return f"🌸 Stillness. Loss remains at {loss:.4f}. Find peace in the plateau."


# --- Registry ---------------------------------------------------------------


_PERSONALITY_REGISTRY: Dict[str, Personality] = {
    "wholesome": wholesome,
    "sassy": sassy,
    "quiet": QuietPersonality(),  # instance is fine; it's still callable
    "nervous": nervous,
    "chaotic": chaotic,
    "arrogant": arrogant,
    "tired": tired,
    "hype": hype,
    "academic": academic,
    "pirate": pirate,
    "zen": zen,
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
