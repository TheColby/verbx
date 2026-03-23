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
