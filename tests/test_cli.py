import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from verbx.cli import app

runner = CliRunner()


@pytest.fixture
def dummy_wav(tmp_path):
    path = tmp_path / "test.wav"
    sr = 44100
    audio = np.random.rand(44100, 2).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path


def test_cli_help():
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout
    assert "analyze" in result.stdout


def test_cli_analyze_real(dummy_wav):
    """Test analyze with real file."""
    result = runner.invoke(app, ["analyze", str(dummy_wav)])
    assert result.exit_code == 0
    assert "Analyzing" in result.stdout
    assert "duration_s" in result.stdout


def test_cli_render_algo(dummy_wav, tmp_path):
    """Test render with algo engine."""
    outfile = tmp_path / "out_algo.wav"
    result = runner.invoke(
        app,
        [
            "render",
            str(dummy_wav),
            str(outfile),
            "--engine",
            "algo",
            "--rt60",
            "0.5",
            "--silent",
        ],
    )
    assert result.exit_code == 0
    assert outfile.exists()

    # Check output valid
    out_audio, sr = sf.read(str(outfile))
    assert len(out_audio) > 0
    assert sr == 44100


def test_cli_render_conv(dummy_wav, tmp_path):
    """Test render with conv engine."""
    outfile = tmp_path / "out_conv.wav"
    # Create dummy IR
    ir_path = tmp_path / "ir.wav"
    sf.write(str(ir_path), np.random.rand(1024, 2).astype(np.float32), 44100)

    result = runner.invoke(
        app,
        [
            "render",
            str(dummy_wav),
            str(outfile),
            "--engine",
            "conv",
            "--impulse-response",
            str(ir_path),
            "--silent",
        ],
    )
    assert result.exit_code == 0
    assert outfile.exists()
