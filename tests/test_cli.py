from typer.testing import CliRunner
from verbx.cli import app

runner = CliRunner()

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "verbx v0.1.0" in result.stdout

def test_process():
    result = runner.invoke(app, ["process", "input.wav", "output.wav"])
    assert result.exit_code == 0
    assert "Processing input.wav -> output.wav" in result.stdout
