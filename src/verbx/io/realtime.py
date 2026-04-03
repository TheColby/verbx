"""Realtime audio I/O helpers with lazy ``sounddevice`` loading."""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from verbx.core.convolution_reverb import LiveConvolutionProcessor

AudioArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class RealtimeDeviceInfo:
    """Compact description of an audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: str | None = None


@dataclass(slots=True)
class RealtimeSessionSummary:
    """Human-readable summary returned after a realtime run ends."""

    sample_rate: int
    block_size: int
    input_channels: int
    output_channels: int
    input_device: str
    output_device: str
    processed_blocks: int
    processed_seconds: float
    clipped_blocks: int


def require_sounddevice() -> Any:
    """Import ``sounddevice`` only when realtime functionality is actually used."""
    try:
        return importlib.import_module("sounddevice")
    except ModuleNotFoundError as exc:
        msg = (
            "Realtime audio requires the optional 'sounddevice' backend. "
            "Install with: `pip install sounddevice` or `uv pip install sounddevice`."
        )
        raise RuntimeError(msg) from exc


def list_audio_devices() -> list[RealtimeDeviceInfo]:
    """Return available input/output devices from the sounddevice backend."""
    sd = require_sounddevice()
    hostapis_raw = sd.query_hostapis()
    hostapis: list[str | None] = []
    for entry in hostapis_raw:
        if isinstance(entry, dict):
            hostapis.append(str(entry.get("name", "")) or None)
        else:
            hostapis.append(None)

    devices: list[RealtimeDeviceInfo] = []
    for idx, entry in enumerate(sd.query_devices()):
        if not isinstance(entry, dict):
            continue
        hostapi_idx = entry.get("hostapi")
        hostapi_name = None
        if isinstance(hostapi_idx, int) and 0 <= hostapi_idx < len(hostapis):
            hostapi_name = hostapis[hostapi_idx]
        devices.append(
            RealtimeDeviceInfo(
                index=idx,
                name=str(entry.get("name", f"device-{idx}")),
                max_input_channels=int(entry.get("max_input_channels", 0)),
                max_output_channels=int(entry.get("max_output_channels", 0)),
                default_samplerate=float(entry.get("default_samplerate", 0.0)),
                hostapi=hostapi_name,
            )
        )
    return devices


def resolve_audio_device(
    *,
    selector: str | int | None,
    kind: str,
    devices: Sequence[RealtimeDeviceInfo] | None = None,
) -> RealtimeDeviceInfo:
    """Resolve an audio device by index or case-insensitive substring."""
    available = list(devices if devices is not None else list_audio_devices())
    capability_key = "max_input_channels" if kind == "input" else "max_output_channels"
    capable = [device for device in available if int(getattr(device, capability_key)) > 0]
    if len(capable) == 0:
        raise RuntimeError(f"No realtime {kind} devices are available.")

    if selector is None:
        return capable[0]

    if isinstance(selector, str):
        stripped = selector.strip()
        if stripped == "":
            return capable[0]
        if stripped.isdigit():
            target = int(stripped)
            for device in capable:
                if int(device.index) == target:
                    return device
            raise ValueError(f"No realtime {kind} device with index {target}.")
        needle = stripped.lower()
    else:
        target = int(selector)
        for device in capable:
            if int(device.index) == target:
                return device
        raise ValueError(f"No realtime {kind} device with index {target}.")

    for device in capable:
        haystack = f"{device.name} {device.hostapi or ''}".lower()
        if needle in haystack:
            return device
    raise ValueError(f"No realtime {kind} device matching '{selector}'.")


def run_realtime_duplex(
    *,
    processor: LiveConvolutionProcessor,
    sample_rate: int,
    block_size: int,
    input_device: RealtimeDeviceInfo,
    output_device: RealtimeDeviceInfo,
    input_channels: int,
    output_channels: int,
    duration_seconds: float | None = None,
) -> RealtimeSessionSummary:
    """Run a duplex realtime session until Ctrl-C or ``duration_seconds`` expires."""
    sd = require_sounddevice()
    stop_event = threading.Event()
    stats = {
        "processed_blocks": 0,
        "processed_frames": 0,
        "clipped_blocks": 0,
    }

    def callback(indata: Any, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        _ = time_info
        if status:
            # Real-time code should be boringly explicit when the backend hiccups.
            # If the driver is unhappy, zero the block rather than blasting garbage.
            outdata.fill(0.0)
            stats["clipped_blocks"] += 1
            return
        input_block = np.asarray(indata, dtype=np.float64)
        rendered = processor.process_block(input_block)
        rendered = np.asarray(rendered, dtype=np.float32)
        outdata.fill(0.0)
        copy_channels = min(int(outdata.shape[1]), int(rendered.shape[1]))
        outdata[:, :copy_channels] = rendered[:, :copy_channels]
        if np.max(np.abs(rendered)) > 0.999:
            stats["clipped_blocks"] += 1
        stats["processed_blocks"] += 1
        stats["processed_frames"] += int(frames)
        if stop_event.is_set():
            raise sd.CallbackStop()

    try:
        with sd.Stream(
            samplerate=float(sample_rate),
            blocksize=int(block_size),
            device=(int(input_device.index), int(output_device.index)),
            channels=(int(input_channels), int(output_channels)),
            dtype="float32",
            callback=callback,
        ):
            if duration_seconds is None:
                while True:
                    time.sleep(0.1)
            else:
                time.sleep(max(0.0, float(duration_seconds)))
                stop_event.set()
                time.sleep(max(0.05, float(block_size) / float(sample_rate)))
    except KeyboardInterrupt:
        stop_event.set()
    return RealtimeSessionSummary(
        sample_rate=int(sample_rate),
        block_size=int(block_size),
        input_channels=int(input_channels),
        output_channels=int(output_channels),
        input_device=input_device.name,
        output_device=output_device.name,
        processed_blocks=int(stats["processed_blocks"]),
        processed_seconds=float(stats["processed_frames"]) / float(max(1, sample_rate)),
        clipped_blocks=int(stats["clipped_blocks"]),
    )
