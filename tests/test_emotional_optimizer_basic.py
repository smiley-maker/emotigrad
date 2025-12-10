import torch

from emotigrad import EmotionalOptimizer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def test_emotional_optimizer_wraps_optimizer_and_steps():
    model = DummyModel()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    emo_opt = EmotionalOptimizer(base_opt, enabled=False)

    x = torch.randn(4, 2)
    y = torch.randn(4, 1)

    # compute a loss
    preds = model(x)
    loss = (preds - y).pow(2).mean()

    # standard training step using EmotionalOptimizer
    emo_opt.zero_grad()
    loss.backward()
    emo_opt.step(loss=loss.item())

    # parameters should have changed
    with torch.no_grad():
        preds_after = model(x)
    new_loss = (preds_after - y).pow(2).mean()

    assert emo_opt.step_count == 1
    # We don't assert that loss strictly decreases, but
    # we at least verify that something happened.
    assert new_loss != loss


def test_emotional_optimizer_calls_personality_when_enabled():
    model = DummyModel()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    messages = []

    def fake_print(msg: str):
        messages.append(msg)

    def fake_personality(loss, prev_loss, step):
        return f"step={step}, loss={loss}, prev={prev_loss}"

    emo_opt = EmotionalOptimizer(
        base_opt,
        personality=fake_personality,
        enabled=True,
        print_fn=fake_print,
    )

    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    preds = model(x)
    loss = (preds - y).pow(2).mean()

    emo_opt.zero_grad()
    loss.backward()
    emo_opt.step(loss=loss.item())

    assert len(messages) == 1
    assert "step=1" in messages[0]


# Confirm personality is NOT called when enabled=False
def test_emotional_optimizer_personality_not_called_when_disabled():
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


# Confirm exceptions inside personalities do not crash step() and are safely swallowed
def test_emotional_optimizer_personality_exceptions_handled():
    model = torch.nn.Linear(2, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def faulty_personality(loss, prev_loss, step):
        raise RuntimeError("Deliberate failure inside personality")

    emo_opt = EmotionalOptimizer(
        base_opt,
        personality=faulty_personality,
        enabled=True,
    )

    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    preds = model(x)
    loss = (preds - y).pow(2).mean()

    emo_opt.zero_grad()
    loss.backward()

    # This should not raise, despite the personality raising internally
    emo_opt.step(loss=loss.item())
