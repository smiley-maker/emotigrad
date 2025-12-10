from typing import Optional, Protocol, runtime_checkable


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
