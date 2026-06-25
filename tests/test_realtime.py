from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import Result as ClickResult
from pytest import MonkeyPatch
from typer.testing import CliRunner

from verbx.cli import app
from verbx.commands import realtime as realtime_cmd
from verbx.config import RenderConfig
from verbx.core.dereverb import LiveDereverbConfig, create_live_dereverb_processor
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


def test_resolve_audio_device_error_lists_viable_devices() -> None:
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
            index=3,
            name="Studio Output",
            max_input_channels=0,
            max_output_channels=8,
            default_samplerate=96_000.0,
            hostapi="CoreAudio",
        ),
    ]

    try:
        realtime_io.resolve_audio_device(selector="not-there", kind="output", devices=devices)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing output device to fail")

    assert "No realtime output device matching" in message
    assert "3:Studio Output" in message
    assert "verbx realtime --list-devices" in message


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


def test_realtime_json_out_writes_session_report(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    class _FakeProcessor:
        output_channels = 2

    def fake_build_live_processor(
        *,
        config: RenderConfig,
        sample_rate: int,
        input_channels: int,
    ) -> tuple[_FakeProcessor, dict[str, object]]:
        return _FakeProcessor(), {
            "engine_resolved": "algo_proxy_live",
            "compute_backend": "cpu",
            "rt60": config.rt60,
            "fdn_lines": config.fdn_lines,
            "proxy_ir_path": tmp_path / "proxy.wav",
            "sample_rate": sample_rate,
            "input_channels": input_channels,
        }

    def fake_run_realtime_duplex(**_: object) -> realtime_io.RealtimeSessionSummary:
        return realtime_io.RealtimeSessionSummary(
            sample_rate=48_000,
            block_size=256,
            input_channels=2,
            output_channels=2,
            input_device="Mic",
            output_device="Speakers",
            input_channel_map=(1, 2),
            output_channel_map=(1, 2),
            processed_blocks=4,
            processed_seconds=0.021333,
            clipped_blocks=1,
            reported_input_latency_seconds=0.003,
            reported_output_latency_seconds=0.004,
        )

    json_out = tmp_path / "realtime.json"
    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)
    monkeypatch.setattr(realtime_cmd, "_build_live_processor", fake_build_live_processor)
    monkeypatch.setattr(realtime_cmd, "run_realtime_duplex", fake_run_realtime_duplex)

    result = runner.invoke(
        app,
        [
            "realtime",
            "--engine",
            "algo",
            "--duration",
            "0.01",
            "--block-size",
            "256",
            "--json-out",
            str(json_out),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "realtime-report-v1"
    assert payload["command"] == "realtime"
    assert payload["status"] == "ok"
    assert payload["input"]["device"] == "Mic"
    assert payload["output"]["device"] == "Speakers"
    assert payload["engine"]["proxy_ir_path"].endswith("proxy.wav")
    assert payload["latency"]["reported_backend_latency_ms"] == 7.0
    assert payload["processed"]["blocks"] == 4


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


def test_realtime_algo_accepts_extended_proxy_options(
    monkeypatch: MonkeyPatch,
) -> None:
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
    captured: dict[str, object] = {}

    class _FakeProcessor:
        output_channels = 2

    def fake_build_live_processor(
        *,
        config: RenderConfig,
        sample_rate: int,
        input_channels: int,
    ) -> tuple[_FakeProcessor, dict[str, object]]:
        captured["config"] = config
        captured["sample_rate"] = sample_rate
        captured["input_channels"] = input_channels
        return _FakeProcessor(), {
            "engine_resolved": "algo_proxy_live",
            "compute_backend": "cpu",
            "rt60": config.rt60,
            "fdn_matrix": config.fdn_matrix,
            "fdn_lines": config.fdn_lines,
            "freeze": config.freeze,
            "shimmer": config.shimmer,
            "tv": True,
            "proxy_eq": "lowcut=120Hz, highcut=9000Hz, tilt=+1.5dB/oct",
        }

    def fake_run_realtime_duplex(**_: object) -> realtime_io.RealtimeSessionSummary:
        return realtime_io.RealtimeSessionSummary(
            sample_rate=48_000,
            block_size=256,
            input_channels=2,
            output_channels=2,
            input_device="Mic",
            output_device="Speakers",
            input_channel_map=(1, 2),
            output_channel_map=(1, 2),
            processed_blocks=2,
            processed_seconds=0.01,
            clipped_blocks=0,
        )

    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)
    monkeypatch.setattr(realtime_cmd, "_build_live_processor", fake_build_live_processor)
    monkeypatch.setattr(realtime_cmd, "run_realtime_duplex", fake_run_realtime_duplex)

    result = runner.invoke(
        app,
        [
            "realtime",
            "--engine",
            "algo",
            "--duration",
            "0.01",
            "--block-size",
            "256",
            "--rt60",
            "18",
            "--pre-delay-ms",
            "45",
            "--damping",
            "0.58",
            "--width",
            "1.2",
            "--mod-depth-ms",
            "3.0",
            "--mod-rate-hz",
            "0.35",
            "--fdn-lines",
            "12",
            "--fdn-matrix",
            "tv-unitary",
            "--fdn-tv-rate-hz",
            "0.4",
            "--fdn-tv-depth",
            "0.15",
            "--fdn-dfm-delays-ms",
            "0.8,1.2,1.6,2.0",
            "--fdn-sparse",
            "--fdn-sparse-degree",
            "3",
            "--fdn-cascade",
            "--fdn-cascade-mix",
            "0.4",
            "--fdn-rt60-low",
            "24",
            "--fdn-rt60-mid",
            "18",
            "--fdn-rt60-high",
            "9",
            "--fdn-rt60-tilt",
            "-0.2",
            "--fdn-link-filter",
            "lowpass",
            "--fdn-link-filter-hz",
            "1800",
            "--fdn-graph-topology",
            "star",
            "--fdn-matrix-morph-to",
            "householder",
            "--fdn-spatial-coupling-mode",
            "all_to_all",
            "--fdn-spatial-coupling-strength",
            "0.4",
            "--fdn-nonlinearity",
            "tanh",
            "--fdn-nonlinearity-amount",
            "0.25",
            "--fdn-nonlinearity-drive",
            "1.7",
            "--room-size-macro",
            "0.3",
            "--clarity-macro",
            "-0.1",
            "--warmth-macro",
            "0.25",
            "--envelopment-macro",
            "0.5",
            "--algo-decorrelation-front",
            "0.2",
            "--algo-decorrelation-rear",
            "0.35",
            "--algo-decorrelation-top",
            "0.1",
            "--allpass-stages",
            "3",
            "--allpass-gain",
            "0.65,0.6,0.55",
            "--allpass-delays-ms",
            "4,7,11",
            "--comb-delays-ms",
            "29,31,37,41",
            "--freeze",
            "--shimmer",
            "--shimmer-semitones",
            "7",
            "--shimmer-mix",
            "0.2",
            "--shimmer-feedback",
            "0.45",
            "--shimmer-highcut",
            "8000",
            "--shimmer-lowcut",
            "250",
            "--shimmer-spatial",
            "--shimmer-spread-cents",
            "12",
            "--shimmer-decorrelation-ms",
            "2.5",
            "--unsafe-self-oscillate",
            "--unsafe-loop-gain",
            "1.01",
            "--algo-proxy-ir-max-seconds",
            "240",
            "--lowcut",
            "120",
            "--highcut",
            "9000",
            "--tilt",
            "1.5",
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    config = captured["config"]
    assert isinstance(config, RenderConfig)
    assert config.fdn_matrix == "tv_unitary"
    assert config.fdn_matrix_morph_to == "householder"
    assert config.fdn_graph_topology == "star"
    assert config.fdn_link_filter == "lowpass"
    assert config.fdn_nonlinearity == "tanh"
    assert config.fdn_spatial_coupling_mode == "all_to_all"
    assert config.freeze is True
    assert config.shimmer is True
    assert config.lowcut == 120.0
    assert config.highcut == 9000.0
    assert config.tilt == 1.5
    assert config.allpass_gains == (0.65, 0.6, 0.55)
    assert config.allpass_delays_ms == (4.0, 7.0, 11.0)
    assert config.comb_delays_ms == (29.0, 31.0, 37.0, 41.0)
    assert config.fdn_dfm_delays_ms == (0.8, 1.2, 1.6, 2.0)
    assert "proxy_freeze" in result.stdout
    assert "proxy_eq" in result.stdout
    assert "Realtime Preflight" in result.stdout
    assert "proxy_ir_budget" in result.stdout


def test_realtime_conv_rejects_algo_only_switches(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    dummy_ir = tmp_path / "dummy.wav"
    dummy_ir.write_bytes(b"not-really-a-wav-but-it-exists")
    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)

    result = runner.invoke(
        app,
        [
            "realtime",
            "--engine",
            "conv",
            "--ir",
            str(dummy_ir),
            "--rt60",
            "18",
            "--freeze",
            "--duration",
            "0.01",
        ],
    )
    assert result.exit_code != 0
    combined = _combined_output(result)
    assert "only apply to realtime --engine algo" in combined
    assert "--rt60" in combined
    assert "--freeze" in combined


def test_realtime_freeze_rejects_unbounded_proxy_startup(
    monkeypatch: MonkeyPatch,
) -> None:
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

    result = runner.invoke(
        app,
        [
            "realtime",
            "--engine",
            "algo",
            "--rt60",
            "3600",
            "--freeze",
            "--duration",
            "0.01",
        ],
    )
    assert result.exit_code != 0
    combined = _combined_output(result)
    assert "startup that looks hung" in combined
    assert "--algo-proxy-ir-max-seconds" in combined


def test_realtime_channel_map_sets_processor_widths(monkeypatch: MonkeyPatch) -> None:
    fake_devices = [
        realtime_io.RealtimeDeviceInfo(
            index=0,
            name="Mic",
            max_input_channels=8,
            max_output_channels=0,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        realtime_io.RealtimeDeviceInfo(
            index=1,
            name="Speakers",
            max_input_channels=0,
            max_output_channels=8,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
    ]
    captured: dict[str, object] = {}

    class _FakeProcessor:
        output_channels = 4

    def fake_build_live_processor(
        *,
        config: RenderConfig,
        sample_rate: int,
        input_channels: int,
    ) -> tuple[_FakeProcessor, dict[str, object]]:
        captured["input_channels"] = input_channels
        return _FakeProcessor(), {
            "engine_resolved": "algo_proxy_live",
            "compute_backend": "cpu",
            "rt60": config.rt60,
            "fdn_matrix": config.fdn_matrix,
            "fdn_lines": config.fdn_lines,
            "freeze": config.freeze,
            "shimmer": config.shimmer,
            "tv": False,
            "proxy_eq": "flat",
        }

    def fake_run_realtime_duplex(**kwargs: object) -> realtime_io.RealtimeSessionSummary:
        captured.update(kwargs)
        return realtime_io.RealtimeSessionSummary(
            sample_rate=48_000,
            block_size=256,
            input_channels=2,
            output_channels=4,
            input_device="Mic",
            output_device="Speakers",
            input_channel_map=(1, 3),
            output_channel_map=(1, 2, 5, 6),
            processed_blocks=1,
            processed_seconds=0.01,
            clipped_blocks=0,
        )

    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)
    monkeypatch.setattr(realtime_cmd, "_build_live_processor", fake_build_live_processor)
    monkeypatch.setattr(realtime_cmd, "run_realtime_duplex", fake_run_realtime_duplex)

    result = runner.invoke(
        app,
        [
            "realtime",
            "--engine",
            "algo",
            "--duration",
            "0.01",
            "--input-channel-map",
            "1,3",
            "--output-channel-map",
            "1,2,5,6",
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    assert captured["input_channels"] == 2
    assert captured["output_channels"] == 4
    assert captured["input_channel_map"] == (1, 3)
    assert captured["output_channel_map"] == (1, 2, 5, 6)
    assert "input_map" in result.stdout
    assert "1,3" in result.stdout
    assert "1,2,5,6" in result.stdout


def test_realtime_channel_map_rejects_count_mismatch(monkeypatch: MonkeyPatch) -> None:
    fake_devices = [
        realtime_io.RealtimeDeviceInfo(
            index=0,
            name="Mic",
            max_input_channels=8,
            max_output_channels=0,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        realtime_io.RealtimeDeviceInfo(
            index=1,
            name="Speakers",
            max_input_channels=0,
            max_output_channels=8,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
    ]
    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)

    result = runner.invoke(
        app,
        [
            "realtime",
            "--engine",
            "algo",
            "--duration",
            "0.01",
            "--input-channels",
            "2",
            "--input-channel-map",
            "1,3,5",
        ],
    )
    assert result.exit_code != 0
    combined = _combined_output(result)
    assert "must match the number of entries in" in combined
    assert "--input-channel-map" in combined


def test_realtime_dereverb_accepts_extended_options(monkeypatch: MonkeyPatch) -> None:
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
    captured: dict[str, object] = {}

    class _FakeProcessor:
        input_channels = 2
        output_channels = 2

        def process_block(self, block: np.ndarray) -> np.ndarray:
            return np.asarray(block, dtype=np.float64)

    def fake_build_live_dereverb_processor(
        *,
        config: LiveDereverbConfig,
        sample_rate: int,
        block_size: int,
        input_channels: int,
    ) -> tuple[_FakeProcessor, dict[str, object]]:
        captured["config"] = config
        captured["sample_rate"] = sample_rate
        captured["block_size"] = block_size
        captured["input_channels"] = input_channels
        return _FakeProcessor(), {
            "engine_resolved": "dereverb_live",
            "compute_backend": "stft_cpu",
            "dereverb_mode": config.mode,
            "dereverb_window_type": config.analysis_window,
            "dereverb_synthesis_window_type": (
                config.analysis_window
                if config.synthesis_window is None
                else config.synthesis_window
            ),
            "dereverb_window_ms": config.window_ms,
            "dereverb_hop_ms": config.hop_ms,
            "dereverb_tail_ms": config.tail_ms,
            "dereverb_mix": config.mix,
            "dereverb_stereo_link": config.stereo_link,
            "dereverb_latency_ms": 24.0,
        }

    def fake_run_realtime_duplex(**_: object) -> realtime_io.RealtimeSessionSummary:
        return realtime_io.RealtimeSessionSummary(
            sample_rate=48_000,
            block_size=384,
            input_channels=2,
            output_channels=2,
            input_device="Mic",
            output_device="Speakers",
            input_channel_map=(1, 2),
            output_channel_map=(1, 2),
            processed_blocks=2,
            processed_seconds=0.02,
            clipped_blocks=0,
        )

    monkeypatch.setattr(realtime_cmd, "list_audio_devices", lambda: fake_devices)
    monkeypatch.setattr(
        realtime_cmd,
        "_build_live_dereverb_processor",
        fake_build_live_dereverb_processor,
    )
    monkeypatch.setattr(realtime_cmd, "run_realtime_duplex", fake_run_realtime_duplex)

    result = runner.invoke(
        app,
        [
            "realtime",
            "--live-mode",
            "dereverb",
            "--duration",
            "0.01",
            "--block-size",
            "384",
            "--dereverb-mode",
            "spectral_sub",
            "--dereverb-strength",
            "0.9",
            "--dereverb-floor",
            "0.05",
            "--dereverb-window-ms",
            "12",
            "--dereverb-hop-ms",
            "4",
            "--dereverb-tail-ms",
            "90",
            "--dereverb-pre-emphasis",
            "0.25",
            "--dereverb-mix",
            "0.85",
            "--dereverb-max-atten-db",
            "21",
            "--no-dereverb-stereo-link",
            "--dereverb-input-gain-db",
            "3",
            "--dereverb-output-gain-db",
            "-2",
            "--dereverb-window-type",
            "kaiser",
            "--dereverb-synthesis-window-type",
            "tukey",
            "--dereverb-window-beta",
            "9",
            "--dereverb-window-alpha",
            "0.3",
            "--dereverb-window-std",
            "2.1",
            "--dereverb-window-power",
            "1.7",
            "--dereverb-window-atten-db",
            "85",
            "--dereverb-window-nbar",
            "5",
            "--dereverb-window-nw",
            "3.1",
            "--dereverb-window-tau",
            "2.8",
            "--dereverb-window-weights",
            "0.42,0.5,0.08",
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    config = captured["config"]
    assert isinstance(config, LiveDereverbConfig)
    assert config.mode == "spectral_sub"
    assert config.strength == 0.9
    assert config.floor == 0.05
    assert config.window_ms == 12.0
    assert config.hop_ms == 4.0
    assert config.tail_ms == 90.0
    assert config.pre_emphasis == 0.25
    assert config.mix == 0.85
    assert config.max_atten_db == 21.0
    assert config.stereo_link is False
    assert config.input_gain_db == 3.0
    assert config.output_gain_db == -2.0
    assert config.analysis_window == "kaiser"
    assert config.synthesis_window == "tukey"
    assert config.window_beta == 9.0
    assert config.window_alpha == 0.3
    assert config.window_std == 2.1
    assert config.window_power == 1.7
    assert config.window_atten_db == 85.0
    assert config.window_nbar == 5
    assert config.window_nw == 3.1
    assert config.window_tau == 2.8
    assert config.window_weights == (0.42, 0.5, 0.08)
    assert "dereverb_mode" in result.stdout
    assert "block_latency" in result.stdout
    assert "dereverb_latency" in result.stdout
    assert "software_latency_est" in result.stdout
    assert "end_to_end_est" in result.stdout
    assert "dereverb_windows" in result.stdout


def test_realtime_dereverb_rejects_block_size_hop_mismatch(monkeypatch: MonkeyPatch) -> None:
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

    result = runner.invoke(
        app,
        [
            "realtime",
            "--live-mode",
            "dereverb",
            "--duration",
            "0.01",
            "--block-size",
            "256",
            "--dereverb-hop-ms",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "divisible by the resolved hop size" in _combined_output(result)


def test_run_realtime_duplex_applies_explicit_channel_maps(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProcessor:
        def __init__(self) -> None:
            self.output_channels = 2

        def process_block(self, input_block: np.ndarray) -> np.ndarray:
            captured["input_block"] = np.array(input_block, copy=True)
            output = np.zeros((input_block.shape[0], 2), dtype=np.float64)
            output[:, 0] = input_block[:, 0] + 10.0
            output[:, 1] = input_block[:, 1] + 20.0
            return output

    class _FakeCallbackError(Exception):
        pass

    class _FakeStream:
        def __init__(self, **kwargs: object) -> None:
            captured["stream_kwargs"] = kwargs
            self._callback: Any = kwargs["callback"]

        def __enter__(self) -> _FakeStream:
            indata = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                dtype=np.float32,
            )
            outdata = np.full((2, 6), -99.0, dtype=np.float32)
            try:
                self._callback(indata, outdata, 2, None, None)
            except _FakeCallbackError:
                pass
            captured["outdata"] = np.array(outdata, copy=True)
            raise KeyboardInterrupt

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    class _FakeSoundDevice:
        CallbackStop = _FakeCallbackError

        def __init__(self) -> None:
            self.Stream = _stream_factory

    def _stream_factory(**kwargs: object) -> _FakeStream:
        return _FakeStream(**kwargs)

    fake_sounddevice = _FakeSoundDevice()
    monkeypatch.setattr(realtime_io, "require_sounddevice", lambda: fake_sounddevice)

    summary = realtime_io.run_realtime_duplex(
        processor=_FakeProcessor(),  # type: ignore[arg-type]
        sample_rate=48_000,
        block_size=2,
        input_device=realtime_io.RealtimeDeviceInfo(
            index=0,
            name="Mic",
            max_input_channels=4,
            max_output_channels=0,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        output_device=realtime_io.RealtimeDeviceInfo(
            index=1,
            name="Speakers",
            max_input_channels=0,
            max_output_channels=6,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        input_channels=2,
        output_channels=2,
        input_channel_map=(1, 3),
        output_channel_map=(2, 5),
        duration_seconds=0.01,
    )
    input_block = captured["input_block"]
    assert isinstance(input_block, np.ndarray)
    np.testing.assert_allclose(input_block, np.array([[1.0, 3.0], [5.0, 7.0]]))
    outdata = captured["outdata"]
    assert isinstance(outdata, np.ndarray)
    np.testing.assert_allclose(outdata[:, 1], np.array([11.0, 15.0], dtype=np.float32))
    np.testing.assert_allclose(outdata[:, 4], np.array([23.0, 27.0], dtype=np.float32))
    np.testing.assert_allclose(outdata[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(outdata[:, 2], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(outdata[:, 3], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(outdata[:, 5], np.zeros(2, dtype=np.float32))
    assert summary.input_channel_map == (1, 3)
    assert summary.output_channel_map == (2, 5)


def test_run_realtime_duplex_reports_stream_latency(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProcessor:
        output_channels = 1

        def process_block(self, input_block: np.ndarray) -> np.ndarray:
            return np.asarray(input_block, dtype=np.float64)

    class _FakeCallbackError(Exception):
        pass

    class _FakeStream:
        latency = (0.006, 0.009)

        def __init__(self, **kwargs: object) -> None:
            captured["stream_kwargs"] = kwargs
            self._callback: Any = kwargs["callback"]

        def __enter__(self) -> _FakeStream:
            indata = np.array([[0.25]], dtype=np.float32)
            outdata = np.zeros((1, 1), dtype=np.float32)
            try:
                self._callback(indata, outdata, 1, None, None)
            except _FakeCallbackError:
                pass
            captured["outdata"] = np.array(outdata, copy=True)
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    class _FakeSoundDevice:
        CallbackStop = _FakeCallbackError

        def __init__(self) -> None:
            self.Stream = _stream_factory

    def _stream_factory(**kwargs: object) -> _FakeStream:
        return _FakeStream(**kwargs)

    def _interrupt_sleep(seconds: float) -> None:
        _ = seconds
        raise KeyboardInterrupt

    fake_sounddevice = _FakeSoundDevice()
    monkeypatch.setattr(realtime_io, "require_sounddevice", lambda: fake_sounddevice)
    monkeypatch.setattr(realtime_io.time, "sleep", _interrupt_sleep)

    started: list[tuple[float | None, float | None]] = []
    summary = realtime_io.run_realtime_duplex(
        processor=_FakeProcessor(),  # type: ignore[arg-type]
        sample_rate=48_000,
        block_size=1,
        input_device=realtime_io.RealtimeDeviceInfo(
            index=0,
            name="Mic",
            max_input_channels=1,
            max_output_channels=0,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        output_device=realtime_io.RealtimeDeviceInfo(
            index=1,
            name="Speakers",
            max_input_channels=0,
            max_output_channels=1,
            default_samplerate=48_000.0,
            hostapi="TestAPI",
        ),
        input_channels=1,
        output_channels=1,
        duration_seconds=0.01,
        on_stream_started=lambda input_latency, output_latency: started.append(
            (input_latency, output_latency)
        ),
    )

    assert started == [(0.006, 0.009)]
    assert summary.reported_input_latency_seconds == 0.006
    assert summary.reported_output_latency_seconds == 0.009


def test_run_realtime_duplex_wraps_stream_open_errors(monkeypatch: MonkeyPatch) -> None:
    class _FakeProcessor:
        input_channels = 1
        output_channels = 1

        def process_block(self, block: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return block

    class _CallbackStopError(Exception):
        pass

    class _FakeSoundDevice:
        CallbackStop = _CallbackStopError

        def __init__(self) -> None:
            self.Stream = _stream_factory

    def _stream_factory(**_: object) -> object:
        raise OSError("backend rejected stream")

    monkeypatch.setattr(realtime_io, "require_sounddevice", lambda: _FakeSoundDevice())

    try:
        realtime_io.run_realtime_duplex(
            processor=_FakeProcessor(),
            sample_rate=96_000,
            block_size=128,
            input_device=realtime_io.RealtimeDeviceInfo(
                index=0,
                name="Mic",
                max_input_channels=2,
                max_output_channels=0,
                default_samplerate=48_000.0,
                hostapi="TestAPI",
            ),
            output_device=realtime_io.RealtimeDeviceInfo(
                index=1,
                name="Speakers",
                max_input_channels=0,
                max_output_channels=2,
                default_samplerate=48_000.0,
                hostapi="TestAPI",
            ),
            input_channels=1,
            output_channels=1,
            duration_seconds=0.01,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected stream-open failure to be wrapped")

    assert "Failed to open realtime duplex stream" in message
    assert "sample_rate=96000" in message
    assert "block_size=128" in message
    assert "verbx realtime --list-devices" in message


def test_apply_realtime_freeze_proxy_uses_safe_defaults() -> None:
    config = RenderConfig(freeze=True, rt60=12.0)
    freeze_proxy = realtime_cmd._apply_realtime_freeze_proxy  # pyright: ignore[reportPrivateUsage]
    frozen = freeze_proxy(config)
    assert frozen.unsafe_self_oscillate is True
    assert frozen.unsafe_loop_gain == 1.002
    assert frozen.algo_proxy_ir_max_seconds >= 120.0


def test_live_dereverb_processor_preserves_shape_and_state() -> None:
    processor = create_live_dereverb_processor(
        sample_rate=48_000,
        input_channels=1,
        config=LiveDereverbConfig(
            mode="wiener",
            window_ms=8.0,
            hop_ms=4.0,
            tail_ms=80.0,
            mix=1.0,
        ),
    )
    hop = int(processor.hop_samples)
    t = np.linspace(0.0, 1.0, hop * 2, endpoint=False)
    block_a = np.asarray((0.2 * np.sin(2.0 * np.pi * 220.0 * t)).reshape(-1, 1), dtype=np.float64)
    block_b = np.asarray((0.2 * np.sin(2.0 * np.pi * 330.0 * t)).reshape(-1, 1), dtype=np.float64)

    out_a = processor.process_block(block_a)
    out_b = processor.process_block(block_b)

    assert out_a.shape == block_a.shape
    assert out_b.shape == block_b.shape
    assert np.isfinite(out_a).all()
    assert np.isfinite(out_b).all()
    assert processor.frame_cursor == int(block_a.shape[0] + block_b.shape[0])
