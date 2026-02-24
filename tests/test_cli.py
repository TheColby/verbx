from typer.testing import CliRunner
from verbx.cli import app

runner = CliRunner()

def test_app():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: " in result.stdout
    assert "Apply extreme reverberation to an audio file." in result.stdout

def test_process_missing_args():
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    # Typer/Click prints errors to stderr, which is captured in result.stdout
    # if mix_stderr=True (default for CliRunner usually, but let's check result.output)
    # Actually, let's just use result.stdout because Click 8.x behavior might vary.
    # But usually help/error output is in result.stdout for CliRunner unless separate_streams is set.
    # Let's inspect what is in result.stdout
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    assert "Missing argument" in result.stdout or "Missing argument" in result.stderr
