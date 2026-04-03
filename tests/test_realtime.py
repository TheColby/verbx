from __future__ import annotations

from click.testing import Result as ClickResult
from pytest import MonkeyPatch
from typer.testing import CliRunner

from verbx.cli import app
from verbx.commands import realtime as realtime_cmd
from verbx.io import realtime as realtime_io

runner = CliRunner()


def _combined_output(result: ClickResult) -> str:
    parts = [result.output]
    stdout_text = getattr(result, "stdout", "")
    stderr_text = getattr(result, "stderr", "")
    if isinstance(stdout_text, str):
        parts.append(stdout_text)
    if isinstance(stderr_text, str):
        parts.append(stderr_text)
    return "\n".join(parts)


def test_resolve_audio_device_supports_index_and_substring() -> None:
    devices = [
        realtime_io.RealtimeDeviceInfo(
            index=0,
            name="Built-in Mic",
            max_input_channels=2,
            max_output_channels=0,
            default_samplerate=48_000.0,
            hostapi="CoreAudio",
        ),
        realtime_io.RealtimeDeviceInfo(
            index=1,
            name="Studio Output",
            max_input_channels=0,
            max_output_channels=2,
            default_samplerate=48_000.0,
            hostapi="CoreAudio",
        ),
    ]

    by_index = realtime_io.resolve_audio_device(selector="0", kind="input", devices=devices)
    assert by_index.name == "Built-in Mic"

    by_name = realtime_io.resolve_audio_device(
        selector="studio",
        kind="output",
        devices=devices,
    )
    assert by_name.index == 1


def test_realtime_list_devices_uses_fake_backend(monkeypatch: MonkeyPatch) -> None:
    fake_devices = [
        realtime_io.RealtimeDeviceInfo(
            index=3,
            name="Loopback 3",
            max_input_channels=2,
            max_output_channels=2,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        )
    ]
    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)

    result = runner.invoke(app, ["realtime", "--list-devices"])
    assert result.exit_code == 0, result.stdout
    assert "Realtime Audio Devices" in result.stdout
    assert "Loopback 3" in result.stdout


def test_realtime_forced_convolution_requires_ir(monkeypatch: MonkeyPatch) -> None:
    fake_devices = [
        realtime_io.RealtimeDeviceInfo(
            index=0,
            name="Mic",
            max_input_channels=2,
            max_output_channels=0,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        realtime_io.RealtimeDeviceInfo(
            index=1,
            name="Speakers",
            max_input_channels=0,
            max_output_channels=2,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
    ]
    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)

    result = runner.invoke(app, ["realtime", "--engine", "conv", "--duration", "0.01"])
    assert result.exit_code != 0
    assert "Realtime convolution requires --ir PATH" in _combined_output(result)
