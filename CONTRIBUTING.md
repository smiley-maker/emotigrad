# Contributing to EmotiGrad

Thanks for your interest in contributing! 🎉  
EmotiGrad is a small, friendly project, and contributions of all kinds are welcome, whether that’s code, docs, tests, ideas, or new personalities.

## Ways to Contribute

You can help by:
- Adding new personalities
- Improving tests or documentation
- Fixing bugs
- Suggesting ideas or enhancements
- Improving examples or formatting

## Development Setup

```bash
git clone https://github.com/smiley-maker/emotigrad.git
cd emotigrad
pip install -e ".[dev]"
```

We recommend using a conda or virtual environment to manage dependencies. 

To create a virtual environment on mac/linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On windows:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Formatting & Linting

We use:
- `black` for formatting
- `ruff` for linting

Please run:
```bash
black .
ruff check .
```

## Running Tests

```bash
pytest
```


## Adding a Personality

Personalities are simple callables with the signature:

```python
def personality(loss: float, prev_loss: float | None, step: int) -> str | None:
    ...
```
### Guidelines:

- Keep messages short and readable
- Avoid excessive logging
- Return `None` when no message should be shown
- Add a small test when possible
- Make it fun and personality driven


## Pull Requests

- Small, focused PRs are preferred
- Please include tests when adding new behavior
- Feel free to open a PR early and ask for feedback
- First-time contributors are very welcome! 🌱

Thanks for helping make EmotiGrad better — we’re glad you’re here! 💖
