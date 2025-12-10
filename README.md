# EmotiGrad

EmotiGrad is a tiny Python library that wraps your PyTorch optimizers and gives you emotionally-charged feedback during training, from wholesome encouragement to unhinged sass.

It aims to be:
- **Drop-in friendly** – keep your usual `torch.optim` code
- **Fun but useful** – emotional logs + basic training insights
- **Extensible** – easily add new "personalities" and behaviors

> ### Because sometimes you need more than just `.step()`, you need support.


## Status

> ⚠️ EmotiGrad is under active early development (pre-release).  
> The API may change before `0.1.0`. Feedback and ideas are very welcome!

---

## Installation

For now, install from source:

```bash
git clone git@github.com:smiley-maker/emotigrad.git
cd emotigrad
pip install -e .
```

(PyPI support will come in a later release.)

## Quick Start

Basic Usage: 

```python
import torch
from emotigrad import EmotionalOptimizer

model = torch.nn.Linear(10, 1)
base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Wrap your optimizer with a personality
optimizer = EmotionalOptimizer(
    base_optimizer,
    personality="wholesome",  # (planned) "sassy", "chaotic", etc.
)

for step in range(10):
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    preds = model(x)
    loss = (preds - y).pow(2).mean()

    optimizer.optimizer.zero_grad()
    loss.backward()

    # In a future version, this call will emit emotional messages
    optimizer.step()
```

In upcoming versions, `optimizer.step(loss=loss.item())` will trigger personality-specific messages based on how training is going.

## Roadmap (high level)

Planned features:

- Emotional personas: 
    - wholesome – positive, encouraging
    - sassy – mildly offended by your gradients
    - chaotic – unhelpful but entertaining
- Basic training trend detection (loss going up/down, plateauing)
- Configurable verbosity and logging destinations
- Easy hooks for custom personalities
- Longer-term ideas:
    - Integration with PyTorch Lightning / HuggingFace Trainer
    - Optional LLM-based "training advisor" for suggestions

## Contributions

Contributions are very welcome, even at this early stage!

Some helpful ways to contribute:

- Try EmotiGrad in a toy project and open issues for:
    - bugs
    - confusing APIs
    - personality ideas
    - Add or improve a personality preset
- Add tests or docs

To setup for development: 

```bash
git clone git@github.com:smiley-maker/emotigrad.git
cd emotigrad
pip install -e ".[dev]"
pytest
```

We also suggest using a virtual environment or conda to manage dependencies. Development dependencies and more detailed instructions will come as the project evolves. 

## Current Project Structure

```
emotigrad/
  emotigrad/
    __init__.py
    base.py           # will hold EmotionalOptimizer
  tests/
    test_smoke.py     # tiny test so CI has something
  README.md
  LICENSE
  pyproject.toml
  .gitignore
```

## License

EmotiGrad is open source and under the MIT License. Learn more in the LICENSE file. 