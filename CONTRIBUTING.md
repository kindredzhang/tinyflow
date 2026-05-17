# Contributing to TinyFlow

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/kindredzhang/tinyflow.git
cd tinyflow
uv sync
```

## Running Tests

```bash
uv run pytest
uv run pytest tests/test_factories.py -v
```

## Code Quality

```bash
uv run ruff check .   # lint
uv run ruff format .   # format
uv run pyright         # type check
```

## How to Contribute

### Report Bugs

Open an [issue](https://github.com/kindredzhang/tinyflow/issues) with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python version

### Suggest Features

Open an [issue](https://github.com/kindredzhang/tinyflow/issues) with the `enhancement` label.

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes with tests
4. Run `uv run pytest` to verify
5. Run linting: `uv run ruff check . && uv run ruff format .`
6. Open a PR against `main`

## Project Structure

```
tinyflow/
  config/       # Configuration helpers
  core/          # Agent, Tools, Types
  providers/     # OpenAI, Anthropic, Gemini
  embeddings/    # OpenAI + local embeddings
  vector/        # ChromaDB, Qdrant adapters
  memory/        # Memory implementations
```

## Questions?

Open an issue or start a discussion.