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
