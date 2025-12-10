import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from emotigrad import EmotionalOptimizer

# --- Data ---
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True,
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Model ---
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# --- Optimizer ---
base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt = EmotionalOptimizer(base_opt, personality="wholesome", message_every=100)

criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
for epoch in range(1, 3):
    for step, (x, y) in enumerate(train_loader):
        preds = model(x)
        loss = criterion(preds, y)

        opt.zero_grad()
        loss.backward()
        opt.step(loss=loss.item())  # <-- triggers emotional output!

        if step % 200 == 0:
            print(f"[Epoch {epoch}] step={step}, loss={loss.item():.4f}")
