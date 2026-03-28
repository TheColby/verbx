"""SOFA interoperability helpers (MVP for info + FIR extraction workflows)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

AudioArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class SofaInfo:
    """Compact SOFA metadata summary returned by ``read_sofa_info``."""

    path: str
    conventions: str
    version: str
    data_ir_shape: tuple[int, ...]
    sample_rate_hz: int
    source_position_shape: tuple[int, ...] | None
    listener_position_shape: tuple[int, ...] | None
    receiver_position_shape: tuple[int, ...] | None
    emitter_position_shape: tuple[int, ...] | None
    dimension_labels: tuple[str, ...]


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        msg = (
            "SOFA support requires optional dependency 'h5py'. "
            "Install with: pip install \"verbx[sofa]\""
        )
        raise RuntimeError(msg) from exc
    return h5py


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        item = value.reshape(()).item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _tuple_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    array = np.asarray(value)
    return tuple(int(dim) for dim in array.shape)


def _read_dataset(group: Any, *keys: str) -> Any:
    for key in keys:
        if key in group:
            return group[key]
    return None


def _safe_sr(value: Any) -> int:
    if value is None:
        return 48_000
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 48_000
    sr = round(float(arr.reshape(-1)[0]))
    return max(1, sr)


def _read_sofa_payload(path: Path) -> dict[str, Any]:
    h5py = _require_h5py()
    with h5py.File(str(path), mode="r") as handle:
        data_group = _read_dataset(handle, "Data")
        if data_group is None:
            msg = "SOFA file missing 'Data' group."
            raise ValueError(msg)
        ir_ds = _read_dataset(data_group, "IR")
        if ir_ds is None:
            msg = "SOFA file missing 'Data/IR' dataset."
            raise ValueError(msg)
        sr_ds = _read_dataset(data_group, "SamplingRate")
        if sr_ds is None:
            sr_ds = _read_dataset(handle, "Data.SamplingRate")
        conventions = _decode_attr(handle.attrs.get("SOFAConventions", "unknown"))
        version = _decode_attr(handle.attrs.get("SOFAConventionsVersion", "unknown"))
        dim_labels: tuple[str, ...] = ()
        labels_raw = ir_ds.attrs.get("DIMENSION_LABELS", ())
        if isinstance(labels_raw, np.ndarray):
            dim_labels = tuple(_decode_attr(v) for v in labels_raw.tolist())
        elif isinstance(labels_raw, (tuple, list)):
            dim_labels = tuple(_decode_attr(v) for v in labels_raw)
        payload: dict[str, Any] = {
            "conventions": conventions,
            "version": version,
            "data_ir": np.asarray(ir_ds, dtype=np.float64),
            "data_ir_shape": tuple(int(dim) for dim in ir_ds.shape),
            "sample_rate_hz": _safe_sr(None if sr_ds is None else sr_ds[()]),
            "source_position": (
                None if "SourcePosition" not in handle else np.asarray(handle["SourcePosition"])
            ),
            "listener_position": (
                None if "ListenerPosition" not in handle else np.asarray(handle["ListenerPosition"])
            ),
            "receiver_position": (
                None if "ReceiverPosition" not in handle else np.asarray(handle["ReceiverPosition"])
            ),
            "emitter_position": (
                None if "EmitterPosition" not in handle else np.asarray(handle["EmitterPosition"])
            ),
            "dimension_labels": dim_labels,
        }
    return payload


def read_sofa_info(path: Path) -> SofaInfo:
    """Return metadata summary for a SOFA file."""
    payload = _read_sofa_payload(path)
    return SofaInfo(
        path=str(path),
        conventions=str(payload["conventions"]),
        version=str(payload["version"]),
        data_ir_shape=tuple(int(v) for v in payload["data_ir_shape"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        source_position_shape=_tuple_shape(payload["source_position"]),
        listener_position_shape=_tuple_shape(payload["listener_position"]),
        receiver_position_shape=_tuple_shape(payload["receiver_position"]),
        emitter_position_shape=_tuple_shape(payload["emitter_position"]),
        dimension_labels=tuple(str(v) for v in payload["dimension_labels"]),
    )


def extract_sofa_ir_matrix_from_array(
    data_ir: AudioArray,
    *,
    measurement_index: int = 0,
    emitter_index: int = 0,
    strict: bool = False,
) -> AudioArray:
    """Extract a ``[taps, channels]`` IR matrix from SOFA Data.IR array."""
    ir = np.asarray(data_ir, dtype=np.float64)
    if ir.ndim == 3:
        m = int(np.clip(measurement_index, 0, max(0, ir.shape[0] - 1)))
        selected = ir[m, :, :]
        return np.asarray(selected.T, dtype=np.float64)
    if ir.ndim == 4:
        m = int(np.clip(measurement_index, 0, max(0, ir.shape[0] - 1)))
        e = int(np.clip(emitter_index, 0, max(0, ir.shape[2] - 1)))
        selected = ir[m, :, e, :]
        return np.asarray(selected.T, dtype=np.float64)

    if strict:
        msg = (
            "Unsupported Data/IR rank for strict extraction. "
            "Supported strict ranks: 3 (M,R,N) and 4 (M,R,E,N)."
        )
        raise ValueError(msg)

    if ir.ndim < 2:
        msg = "SOFA Data/IR must have at least 2 dimensions."
        raise ValueError(msg)

    flat = ir.reshape(-1, ir.shape[-1])
    return np.asarray(flat.T, dtype=np.float64)


def _normalize_ir(audio: AudioArray, mode: str) -> AudioArray:
    normalized_mode = str(mode).strip().lower()
    x = np.asarray(audio, dtype=np.float64)
    if normalized_mode == "none":
        return x
    if normalized_mode == "peak":
        peak = float(np.max(np.abs(x))) if x.size > 0 else 0.0
        if peak > 1e-12:
            return np.asarray(x / peak, dtype=np.float64)
        return x
    if normalized_mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size > 0 else 0.0
        if rms > 1e-12:
            return np.asarray(x / rms, dtype=np.float64)
        return x
    msg = f"Unsupported normalize mode: {mode}"
    raise ValueError(msg)


def _resample_audio_polyphase(audio: AudioArray, *, src_sr: int, dst_sr: int) -> AudioArray:
    if int(src_sr) == int(dst_sr):
        return np.asarray(audio, dtype=np.float64)
    gcd = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr // gcd)
    down = int(src_sr // gcd)
    y = resample_poly(np.asarray(audio, dtype=np.float64), up=up, down=down, axis=0)
    return np.asarray(y, dtype=np.float64)


def extract_sofa_ir(
    path: Path,
    *,
    measurement_index: int = 0,
    emitter_index: int = 0,
    target_sr: int | None = None,
    normalize: str = "peak",
    strict: bool = False,
) -> tuple[AudioArray, int, dict[str, Any]]:
    """Load SOFA file and extract IR matrix suitable for ``verbx render --engine conv``."""
    payload = _read_sofa_payload(path)
    ir_matrix = extract_sofa_ir_matrix_from_array(
        np.asarray(payload["data_ir"], dtype=np.float64),
        measurement_index=measurement_index,
        emitter_index=emitter_index,
        strict=strict,
    )
    src_sr = int(payload["sample_rate_hz"])
    dst_sr = int(src_sr if target_sr is None else max(1, int(target_sr)))
    if dst_sr != src_sr:
        ir_matrix = _resample_audio_polyphase(ir_matrix, src_sr=src_sr, dst_sr=dst_sr)
    ir_matrix = _normalize_ir(ir_matrix, mode=normalize)

    meta = {
        "mode": "sofa-extract",
        "source": str(path),
        "conventions": str(payload["conventions"]),
        "version": str(payload["version"]),
        "data_ir_shape": tuple(int(v) for v in payload["data_ir_shape"]),
        "measurement_index": int(measurement_index),
        "emitter_index": int(emitter_index),
        "strict": bool(strict),
        "normalize": str(normalize),
        "sample_rate_source_hz": src_sr,
        "sample_rate_output_hz": dst_sr,
        "sample_rate_action": "none" if src_sr == dst_sr else f"resample:{src_sr}->{dst_sr}",
        "output_shape": tuple(int(v) for v in ir_matrix.shape),
        "dimension_labels": tuple(str(v) for v in payload["dimension_labels"]),
    }
    return np.asarray(ir_matrix, dtype=np.float64), int(dst_sr), meta
