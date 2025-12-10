from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from torch.optim import Optimizer

from .types import Personality

PersonalityLike = Union[str, Personality]


@dataclass
class EmotionalOptimizer:
    """Wrap a PyTorch optimizer and add emotional feedback.

    This is intentionally lightweight for the first version:
    - forwards all calls to the underlying optimizer
    - tracks step count and previous loss
    - optionally emits text messages via a personality callable
    """

    optimizer: Optimizer
    personality: PersonalityLike = "wholesome"
    enabled: bool = True
    print_fn: callable = print  # allows tests / users to override output
    message_every: int = 1  # Number of steps between messages

    def __post_init__(self) -> None:
        self._step: int = 0
        self._prev_loss: Optional[float] = None

        self._step: int = 0
        self._prev_avg_loss: Optional[float] = None

        # For averaging over the last N steps
        self._block_loss_sum: float = 0.0
        self._block_loss_count: int = 0

        # Resolve personality if given as a string
        if isinstance(self.personality, str):
            # Lazy import to avoid circular imports
            from .personalities import get_personality

            self.personality = get_personality(self.personality)

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

        # Track losses for averaging
        if loss is not None:
            self._block_loss_sum += float(loss)
            self._block_loss_count += 1

        # Decide whether to emit feedback
        if (
            self.enabled
            and loss is not None
            and self.message_every > 0
            and (self._step % self.message_every == 0)
            and self._block_loss_count > 0
        ):
            current_avg = self._block_loss_sum / self._block_loss_count

            try:
                # message = self.personality(loss, self._prev_loss, self._step)
                message = self.personality(
                    current_avg,
                    self._prev_avg_loss,
                    self._step,
                )
            except Exception:
                # Personality logic should never break training.
                message = None

            if message:
                self.print_fn(message)

            # Prepare for next block
            self._prev_avg_loss = current_avg
            self._block_loss_sum = 0.0
            self._block_loss_count = 0

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
