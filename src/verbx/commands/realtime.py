"""Realtime command for live duplex monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from verbx.config import RenderConfig
from verbx.core.algo_proxy import render_algo_proxy_ir
from verbx.core.convolution_reverb import (
    ConvolutionReverbConfig,
    ConvolutionReverbEngine,
    LiveConvolutionProcessor,
)
from verbx.io.realtime import (
    RealtimeDeviceInfo,
    RealtimeSessionSummary,
    list_audio_devices,
    resolve_audio_device,
    run_realtime_duplex,
)

console = Console()


def realtime(
    engine: Literal["auto", "conv", "algo"] = typer.Option(
        "auto",
        "--engine",
        help=(
            "Realtime engine: convolution IR, or algorithmic proxy rendered "
            "into a live convolver."
        ),
    ),
    ir: Path | None = typer.Option(
        None,
        "--ir",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Impulse response path for realtime convolution mode.",
    ),
    input_device: str | None = typer.Option(
        None,
        "--input-device",
        help="Input device index or case-insensitive name substring.",
    ),
    output_device: str | None = typer.Option(
        None,
        "--output-device",
        help="Output device index or case-insensitive name substring.",
    ),
    list_devices: bool = typer.Option(
        False,
        "--list-devices",
        help="List available realtime audio devices and exit.",
    ),
    sample_rate: int = typer.Option(48_000, "--sample-rate", min=8_000),
    block_size: int = typer.Option(512, "--block-size", min=64),
    partition_size: int = typer.Option(2_048, "--partition-size", min=256),
    input_channels: int | None = typer.Option(
        None,
        "--input-channels",
        min=1,
        help="Requested live input channel count. Defaults to mono or stereo depending on device.",
    ),
    output_channels: int | None = typer.Option(
        None,
        "--output-channels",
        min=1,
        help=(
            "Requested live output channel count. Defaults to the processor's "
            "natural output width."
        ),
    ),
    duration: float | None = typer.Option(
        None,
        "--duration",
        min=0.0,
        help="Optional duration in seconds. Omit to run until Ctrl-C.",
    ),
    wet: float = typer.Option(0.8, "--wet", min=0.0, max=1.0),
    dry: float = typer.Option(0.2, "--dry", min=0.0, max=1.0),
    rt60: float = typer.Option(6.0, "--rt60", min=0.1),
    pre_delay_ms: float = typer.Option(20.0, "--pre-delay-ms", min=0.0),
    damping: float = typer.Option(0.45, "--damping", min=0.0, max=1.0),
    width: float = typer.Option(1.0, "--width", min=0.0, max=2.0),
    fdn_lines: int = typer.Option(8, "--fdn-lines", min=1, max=64),
    fdn_matrix: str = typer.Option(
        "hadamard",
        "--fdn-matrix",
        help="Algorithmic proxy matrix for realtime --engine algo.",
    ),
    allpass_stages: int = typer.Option(6, "--allpass-stages", min=0, max=64),
    allpass_gain: float = typer.Option(0.7, "--allpass-gain", min=-0.99, max=0.99),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce console output."),
) -> None:
    """Run realtime duplex monitoring with selectable input/output devices."""
    try:
        devices = list_audio_devices()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if list_devices:
        _print_device_table(devices)
        return

    try:
        input_info = resolve_audio_device(selector=input_device, kind="input", devices=devices)
        output_info = resolve_audio_device(selector=output_device, kind="output", devices=devices)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    resolved_input_channels = _resolve_input_channels(input_channels, input_info)
    realtime_config = RenderConfig(
        engine=engine,
        rt60=float(rt60),
        pre_delay_ms=float(pre_delay_ms),
        damping=float(damping),
        width=float(width),
        wet=float(wet),
        dry=float(dry),
        fdn_lines=int(fdn_lines),
        fdn_matrix=str(fdn_matrix),
        allpass_stages=int(allpass_stages),
        allpass_gain=float(allpass_gain),
        block_size=int(block_size),
        partition_size=int(partition_size),
        ir=None if ir is None else str(ir),
        target_sr=int(sample_rate),
        output_subtype="float32",
        output_container="wav",
        progress=False,
        silent=True,
    )

    processor, engine_summary = _build_live_processor(
        config=realtime_config,
        sample_rate=int(sample_rate),
        input_channels=resolved_input_channels,
    )
    resolved_output_channels = _resolve_output_channels(
        requested=output_channels,
        output_device=output_info,
        processor_channels=int(processor.output_channels),
    )

    if not quiet:
        _print_realtime_start_table(
            input_device=input_info,
            output_device=output_info,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_channels=resolved_input_channels,
            output_channels=resolved_output_channels,
            engine_summary=engine_summary,
            duration=duration,
        )

    try:
        summary = run_realtime_duplex(
            processor=processor,
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            input_device=input_info,
            output_device=output_info,
            input_channels=resolved_input_channels,
            output_channels=resolved_output_channels,
            duration_seconds=duration,
        )
    finally:
        proxy_path = engine_summary.get("proxy_ir_path")
        if isinstance(proxy_path, Path):
            proxy_path.unlink(missing_ok=True)

    if not quiet:
        _print_realtime_end_table(summary)


def _build_live_processor(
    *,
    config: RenderConfig,
    sample_rate: int,
    input_channels: int,
) -> tuple[LiveConvolutionProcessor, dict[str, str | Path | None]]:
    """Build the live convolution processor and reproducibility metadata."""
    resolved_engine = "conv" if config.ir is not None else "algo"
    if str(config.engine).strip().lower() == "conv" and config.ir is None:
        msg = "Realtime convolution requires --ir PATH. Use --engine algo for proxy mode."
        raise typer.BadParameter(msg)
    if str(config.engine).strip().lower() == "algo":
        resolved_engine = "algo"

    proxy_ir_path: Path | None = None
    ir_path: Path | None = None if config.ir is None else Path(config.ir)
    if resolved_engine == "algo":
        proxy_ir_path, _ = render_algo_proxy_ir(
            config=config,
            sr=int(sample_rate),
            input_channels=int(input_channels),
        )
        ir_path = proxy_ir_path

    if ir_path is None:
        msg = "Realtime mode needs an IR or an algorithmic proxy configuration."
        raise typer.BadParameter(msg)

    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=float(config.wet),
            dry=float(config.dry),
            ir_path=str(ir_path),
            partition_size=int(config.partition_size),
            input_layout="auto",
            output_layout="auto",
            device="cpu",
        )
    )
    processor = engine.create_live_processor(
        input_channels=int(input_channels),
        sr=int(sample_rate),
    )
    return processor, {
        "engine_requested": config.engine,
        "engine_resolved": "algo_proxy_live" if resolved_engine == "algo" else "conv_live",
        "compute_backend": engine.backend_name(),
        "proxy_ir_path": proxy_ir_path,
    }


def _resolve_input_channels(requested: int | None, device: RealtimeDeviceInfo) -> int:
    """Choose an input width that the selected device can actually supply."""
    available = int(device.max_input_channels)
    if available <= 0:
        raise typer.BadParameter(f"Input device '{device.name}' has no input channels.")
    if requested is None:
        return 2 if available >= 2 else 1
    if requested > available:
        raise typer.BadParameter(
            f"Input device '{device.name}' exposes {available} channels, not {requested}."
        )
    return int(requested)


def _resolve_output_channels(
    *,
    requested: int | None,
    output_device: RealtimeDeviceInfo,
    processor_channels: int,
) -> int:
    """Choose an output width that fits both the processor and hardware."""
    available = int(output_device.max_output_channels)
    if available <= 0:
        raise typer.BadParameter(f"Output device '{output_device.name}' has no output channels.")
    if requested is not None:
        if requested > available:
            raise typer.BadParameter(
                f"Output device '{output_device.name}' exposes {available} "
                f"channels, not {requested}."
            )
        return int(requested)
    return max(1, min(int(processor_channels), available))


def _print_device_table(devices: list[RealtimeDeviceInfo]) -> None:
    table = Table(title="Realtime Audio Devices")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Host API", style="magenta")
    table.add_column("In", style="green", justify="right")
    table.add_column("Out", style="green", justify="right")
    table.add_column("Default SR", style="yellow", justify="right")
    for device in devices:
        table.add_row(
            str(device.index),
            device.name,
            device.hostapi or "",
            str(device.max_input_channels),
            str(device.max_output_channels),
            f"{device.default_samplerate:.1f}",
        )
    console.print(table)


def _print_realtime_start_table(
    *,
    input_device: RealtimeDeviceInfo,
    output_device: RealtimeDeviceInfo,
    sample_rate: int,
    block_size: int,
    input_channels: int,
    output_channels: int,
    engine_summary: dict[str, str | Path | None],
    duration: float | None,
) -> None:
    table = Table(title="Realtime Session")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("engine", str(engine_summary.get("engine_resolved", "")))
    table.add_row("backend", str(engine_summary.get("compute_backend", "")))
    table.add_row("sample_rate", str(sample_rate))
    table.add_row("block_size", str(block_size))
    table.add_row("input_device", input_device.name)
    table.add_row("output_device", output_device.name)
    table.add_row("input_channels", str(input_channels))
    table.add_row("output_channels", str(output_channels))
    table.add_row("duration", "until Ctrl-C" if duration is None else f"{duration:.2f}s")
    console.print(table)


def _print_realtime_end_table(summary: RealtimeSessionSummary) -> None:
    table = Table(title="Realtime Session Complete")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("input_device", str(summary.input_device))
    table.add_row("output_device", str(summary.output_device))
    table.add_row("sample_rate", str(summary.sample_rate))
    table.add_row("processed_blocks", str(summary.processed_blocks))
    table.add_row("processed_seconds", f"{summary.processed_seconds:.3f}")
    table.add_row("clipped_blocks", str(summary.clipped_blocks))
    console.print(table)
