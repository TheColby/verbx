from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from verbx.cli import app

runner = CliRunner()


def test_readme_includes_key_batch_corpus_generate_flags() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    result = runner.invoke(app, ["batch", "corpus-generate", "--help"])
    assert result.exit_code == 0, result.stdout
    help_text = result.stdout
    for flag in ["--retries", "--num-shards", "--shard-index", "--checkpoint-file", "--resume"]:
        assert flag in help_text
        assert flag in readme


def test_readme_includes_key_quickstart_flags() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    result = runner.invoke(app, ["quickstart", "--help"])
    assert result.exit_code == 0, result.stdout
    help_text = result.stdout
    for flag in ["--verify", "--strict", "--smoke-test", "--json-out"]:
        assert flag in help_text
        assert flag in readme


def test_readme_includes_large_bus_layout_examples_and_guides() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for token in [
        "--output-layout 7.2.4",
        "--output-layout 8.0",
        "--output-layout 16.0",
        "--output-layout 64.4",
        "docs/notebooks/README.md",
        "docs/SOFA_FEASIBILITY.md",
    ]:
        assert token in readme
