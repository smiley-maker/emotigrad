import torch

from emotigrad import EmotionalOptimizer


def test_message_every_uses_block_average_loss():
    model = torch.nn.Linear(1, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    messages = []

    def fake_print(msg: str):
        messages.append(msg)

    # This personality simply prints the loss it's given
    def echo_personality(loss, prev_loss, step):
        return f"loss={loss:.2f}, prev={None if prev_loss is None else round(prev_loss, 2)}"

    opt = EmotionalOptimizer(
        base_opt,
        personality=echo_personality,
        print_fn=fake_print,
        message_every=3,
    )

    # Provide known loss values
    # Block 1 = [1.0, 2.0, 3.0] → average = 2.0
    # Block 2 = [4.0, 6.0, 8.0] → average = 6.0

    losses = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

    # Dummy step (no gradients needed)
    for loss in losses:
        opt.optimizer.zero_grad()
        opt.step(loss=loss)

    assert len(messages) == 2, "Should emit once per block of 3 steps."

    # First message
    assert "loss=2.00" in messages[0]
    assert "prev=None" in messages[0]

    # Second message
    assert "loss=6.00" in messages[1]
    assert "prev=2.0" in messages[1]
