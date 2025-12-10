import torch

from emotigrad import EmotionalOptimizer


def test_smoke_emotional_optimizer_wraps_optimizer():
    model = torch.nn.Linear(2, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    emo_opt = EmotionalOptimizer(base_opt)

    # Just make sure it doesn't crash
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    loss = (model(x) - y).pow(2).mean()

    loss.backward()
    emo_opt.step()


# Verify that personalities are called correctly and messages are routed through the configured print_fn.
def test_smoke_emotional_optimizer_personality():
    model = torch.nn.Linear(2, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    messages = []

    def fake_print(msg: str):
        messages.append(msg)

    emo_opt = EmotionalOptimizer(
        base_opt,
        personality="sassy",
        enabled=True,
        print_fn=fake_print,
    )

    x = torch.randn(4, 2)
    y = torch.randn(4, 1)

    # First step
    loss1 = (model(x) - y).pow(2).mean()
    loss1.backward()
    emo_opt.step(loss=loss1.item())

    # Second step with worse loss
    with torch.no_grad():
        model.weight += 0.5  # make loss worse
    loss2 = (model(x) - y).pow(2).mean()
    loss2.backward()
    emo_opt.step(loss=loss2.item())

    assert len(messages) == 2


# Confirm personality is called when loss is provided and enabled=True
def test_smoke_emotional_optimizer_personality_called():
    model = torch.nn.Linear(2, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    called = []

    def fake_personality(loss, prev_loss, step):
        called.append((loss, prev_loss, step))
        return None

    emo_opt = EmotionalOptimizer(
        base_opt,
        personality=fake_personality,
        enabled=True,
    )

    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    preds = model(x)
    loss = (preds - y).pow(2).mean()

    emo_opt.zero_grad()
    loss.backward()
    emo_opt.step(loss=loss.item())

    assert len(called) == 1


# Confirm personality is NOT called when enabled=False
def test_smoke_emotional_optimizer_personality_not_called_when_disabled():
    model = torch.nn.Linear(2, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    called = []

    def fake_personality(loss, prev_loss, step):
        called.append((loss, prev_loss, step))
        return None

    emo_opt = EmotionalOptimizer(
        base_opt,
        personality=fake_personality,
        enabled=False,
    )

    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    preds = model(x)
    loss = (preds - y).pow(2).mean()

    emo_opt.zero_grad()
    loss.backward()
    emo_opt.step(loss=loss.item())

    assert len(called) == 0
