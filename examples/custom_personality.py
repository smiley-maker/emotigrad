"""
Example: Creating a custom 'roast' personality for EmotiGrad.

Shows how to write a Personality callable and pass it to EmotionalOptimizer.
"""

import torch
from emotigrad import EmotionalOptimizer


# --- Custom Personality -------------------------------------------------------

def roast(loss, prev_loss, step):
    """A sarcastic personality that roasts the model's progress."""
    if prev_loss is None:
        return f"🔥 Step {step}: New model? Cute. Let's watch it struggle. (avg loss {loss:.4f})"

    if loss < prev_loss:
        return f"😏 Step {step}: Look at you improving! Honestly shocked. ({prev_loss:.4f} → {loss:.4f})"

    if loss > prev_loss:
        return (
            f"🙃 Step {step}: Nice, you made it worse. "
            f"({prev_loss:.4f} → {loss:.4f}). Truly groundbreaking."
        )

    return f"🤨 Step {step}: No change. Riveting."


# --- Training Loop ------------------------------------------------------------

def main():
    model = torch.nn.Linear(5, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Use the custom roast personality
    opt = EmotionalOptimizer(
        base_opt,
        personality=roast,   # <-- pass the callable directly
        message_every=3,     # roast based on averaged loss every 3 steps
    )

    for step in range(12):
        x = torch.randn(32, 5)
        y = torch.randn(32, 1)

        preds = model(x)
        loss = (preds - y).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step(loss=loss.item())


if __name__ == "__main__":
    main()
