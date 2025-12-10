from .types import Personality
from typing import Optional

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