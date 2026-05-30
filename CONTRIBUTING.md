# Contributing

Thanks for improving `verbx`.

## Development setup

1. Use Python 3.11+.
2. Create or use a virtual environment.
3. Install in editable mode with dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

Or use the installer script:

```bash
./scripts/install.sh --dev --editable
```

## Quality gates

Run these before opening a PR (mirrors CI exactly):

```bash
hatch run lint
hatch run typecheck
hatch run test
```

Or if you prefer uv directly:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests
```

## Coding expectations

- Keep DSP and control paths deterministic unless a feature is explicitly
  stochastic and seed-controlled.
- Preserve `float64` internal processing.
- Add tests for behavior changes.
- Update docs/man pages when CLI behavior changes.

## Docs synchronization

When the CLI surface or user-facing workflows change, refresh the generated docs too:

```bash
uv run python scripts/generate_cli_reference.py
python3 scripts_generate_docs_pdf.py
```

This regenerates [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md),
[`docs/USERGUIDE.md`](docs/USERGUIDE.md), and `USERGUIDE.pdf`.

## Pull requests

- Keep PRs focused and scoped.
- Include a concise summary of:
  - what changed,
  - why it changed,
  - how it was validated.
- Reference relevant issues/tasks when applicable.

## Reporting bugs

Please include:

- OS and Python version,
- exact CLI command(s),
- expected vs actual behavior,
- and, if possible, a minimal reproducible input.
