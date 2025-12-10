from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from torch.optim import Optimizer


@runtime_checkable
class Personality(Protocol):
    """Protocol for personality callables."""

    def __call__(
        self,
        loss: float,
        prev_loss: Optional[float],
        step: int,
    ) -> Optional[str]:
        """Generate an emotional message based on the optimization state.

        Parameters
        ----------
        loss:
            Current scalar loss value.
        prev_loss:
            Previous scalar loss value, or None if this is the first step.
        step:
            Current optimization step count (starting from 1).
        Returns:
            An optional string message to emit.
        """
        ...


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


@dataclass
class EmotionalOptimizer:
    """Wrap a PyTorch optimizer and add emotional feedback.

    This is intentionally lightweight for the first version:
    - forwards all calls to the underlying optimizer
    - tracks step count and previous loss
    - optionally emits text messages via a personality callable
    """

    optimizer: Optimizer
    personality: Personality = _default_personality
    enabled: bool = True
    print_fn: callable = print  # allows tests / users to override output

    def __post_init__(self) -> None:
        self._step: int = 0
        self._prev_loss: Optional[float] = None

    def step(self, loss: Optional[float] = None, *args, **kwargs):
        """Perform an optimization step and optionally emit emotional feedback.

        Parameters
        ----------
        loss:
            Current scalar loss value (e.g., loss.item()).
            If None, no emotional message is generated for this step.
        *args, **kwargs:
            Forwarded to the underlying optimizer's step() method.
        """
        result = self.optimizer.step(*args, **kwargs)
        self._step += 1

        if self.enabled and loss is not None:
            try:
                message = self.personality(loss, self._prev_loss, self._step)
            except Exception:
                # Personality logic should never break training.
                message = None

            if message:
                self.print_fn(message)

        if loss is not None:
            self._prev_loss = loss

        return result

    def zero_grad(self, *args, **kwargs):
        """Forward zero_grad to the underlying optimizer."""
        return self.optimizer.zero_grad(*args, **kwargs)

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def previous_loss(self) -> Optional[float]:
        return self._prev_loss
