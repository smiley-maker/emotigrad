"""
Basic usage example for EmotiGrad.

This script shows how to wrap a PyTorch optimizer with an EmotionalOptimizer
and get emotionally-enhanced training feedback. It uses a tiny synthetic dataset
so it runs instantly and without external dependencies.

Run with:
    python examples/basic_usage.py
"""

import torch

from emotigrad import EmotionalOptimizer


def main():
    # Simple linear model for demonstration
    model = torch.nn.Linear(10, 1)

    # Standard optimizer
    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Wrap with EmotiGrad!
    opt = EmotionalOptimizer(
        base_opt,
        personality="wholesome",  # try "sassy" or write your own!
        message_every=5,  # emotional feedback every 5 steps
    )

    # Synthetic training loop
    for step in range(20):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        preds = model(x)
        loss = (preds - y).pow(2).mean()

        opt.zero_grad()
        loss.backward()

        # Passing loss triggers EmotiGrad's emotional feedback
        opt.step(loss=loss.item())

        if step % 5 == 0:
            print(f"[step {step}] loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
