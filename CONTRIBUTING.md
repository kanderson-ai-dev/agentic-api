# Contributing

Thanks for your interest in contributing. This project is a portfolio piece, but
issues and pull requests are welcome.

## Development Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate # macOS/Linux

# Install runtime + dev dependencies (ruff, mypy, pytest)
pip install -r requirements-dev.txt

# Copy and fill in environment variables
cp .env.example .env
```

## Lint, Type-Check, and Test

```bash
ruff check .          # lint
mypy app              # static type checking
pytest -v             # full test suite
```

All three must pass before submitting a PR. Tests that need a real
`OPENAI_API_KEY` / `LANGCHAIN_API_KEY` are automatically skipped when those
secrets are not configured, so the suite runs green with zero secrets.

## Conventions

- Type hints on all public functions.
- Follow the existing code style; `ruff` enforces it in CI.
- Commit messages use conventional prefixes (`feat:`, `fix:`, `refactor:`,
  `chore:`, `docs:`, `test:`).
