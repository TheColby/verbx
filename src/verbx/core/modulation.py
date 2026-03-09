"""External modulation bus for time-varying parameter control.

This module provides a small, typed control-signal engine used by render
workflows. It supports combining multiple sources and mapping the resulting
control stream onto a target parameter domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from verbx.io.audio import ensure_mono_or_stereo, read_audio

AudioArray = npt.NDArray[np.float64]
ControlArray = npt.NDArray[np.float64]
CombineMode = Literal["sum", "avg", "max"]
LFOWave = Literal["sine", "triangle", "square", "saw"]
TargetName = Literal["mix", "wet", "gain-db"]
SourceKind = Literal["lfo", "env", "audio-env", "const"]


@dataclass(slots=True)
class ModulationSource:
    """Parsed modulation source descriptor."""

    spec: str
    kind: SourceKind
    weight: float = 1.0
    shape: LFOWave = "sine"
    rate_hz: float = 0.1
    depth: float = 1.0
    phase_deg: float = 0.0
    attack_ms: float = 20.0
    release_ms: float = 350.0
    path: str | None = None
    value: float = 0.5


@dataclass(slots=True)
class ModulationRoute:
    """One modulation route mapping sources to a target parameter domain."""

    spec: str
    target: TargetName
    value_min: float
    value_max: float
    combine: CombineMode
    smooth_ms: float
    source_specs: tuple[str, ...]


def parse_mod_sources(specs: tuple[str, ...] | list[str]) -> list[ModulationSource]:
    """Parse a list of source specs into typed source descriptors.

    Supported source syntax:
    - ``lfo:<shape>:<rate_hz>[:depth[:phase_deg]][*weight]``
    - ``env[:attack_ms[:release_ms]][*weight]``
    - ``audio-env:<path>[:attack_ms[:release_ms]][*weight]``
    - ``const:<value>[*weight]``
    """
    parsed: list[ModulationSource] = []
    for spec in specs:
        parsed.append(parse_mod_source(spec))
    return parsed


def parse_mod_route_spec(spec: str) -> ModulationRoute:
    """Parse one ``--mod-route`` descriptor.

    Route syntax:
    ``<target>:<min>:<max>:<combine>:<smooth_ms>:<src1>,<src2>,...``
    """
    raw = spec.strip()
    if raw == "":
        raise ValueError("Empty modulation route spec")

    parts = raw.split(":", 5)
    if len(parts) != 6:
        msg = (
            "Invalid --mod-route format. Use "
            "<target>:<min>:<max>:<combine>:<smooth_ms>:<src1>,<src2>,..."
        )
        raise ValueError(msg)

    target = _parse_mod_target(parts[0])
    value_min = float(parts[1])
    value_max = float(parts[2])
    if value_min >= value_max:
        raise ValueError("mod-route min must be less than max")

    if target in {"mix", "wet"} and (value_min < 0.0 or value_max > 1.0):
        raise ValueError("For mix/wet routes, min/max must be in [0.0, 1.0]")

    combine = _parse_combine_mode(parts[3])
    smooth_ms = _parse_non_negative_float(parts[4], "mod-route smooth_ms")
    source_specs = tuple(item.strip() for item in parts[5].split(",") if item.strip() != "")
    if len(source_specs) == 0:
        raise ValueError("mod-route requires at least one source")

    # Validate sources early with existing parser.
    parse_mod_sources(source_specs)

    return ModulationRoute(
        spec=raw,
        target=target,
        value_min=value_min,
        value_max=value_max,
        combine=combine,
        smooth_ms=smooth_ms,
        source_specs=source_specs,
    )


def parse_mod_route_specs(specs: tuple[str, ...] | list[str]) -> list[ModulationRoute]:
    """Parse many ``--mod-route`` descriptors."""
    routes: list[ModulationRoute] = []
    for spec in specs:
        routes.append(parse_mod_route_spec(spec))
    return routes


def parse_mod_source(spec: str) -> ModulationSource:
    """Parse one modulation source spec string."""
    raw = spec.strip()
    if raw == "":
        raise ValueError("Empty modulation source spec")

    base, weight = _split_weight(raw)
    parts = [part.strip() for part in base.split(":")]
    head = parts[0].lower()

    if head == "lfo":
        if len(parts) < 3:
            msg = (
                "Invalid lfo source. Use lfo:<shape>:<rate_hz>[:depth[:phase_deg]][*weight], "
                "for example lfo:sine:0.08:1.0*0.75"
            )
            raise ValueError(msg)
        shape = _parse_lfo_shape(parts[1])
        rate_hz = _parse_non_negative_float(parts[2], "lfo rate_hz")
        depth = _parse_non_negative_float(parts[3], "lfo depth") if len(parts) >= 4 else 1.0
        phase_deg = float(parts[4]) if len(parts) >= 5 else 0.0
        if len(parts) > 5:
            msg = (
                "Invalid lfo source. Too many fields. "
                "Expected lfo:<shape>:<rate_hz>[:depth[:phase_deg]][*weight]"
            )
            raise ValueError(msg)
        return ModulationSource(
            spec=raw,
            kind="lfo",
            weight=weight,
            shape=shape,
            rate_hz=rate_hz,
            depth=depth,
            phase_deg=phase_deg,
        )

    if head == "env":
        if len(parts) > 3:
            msg = "Invalid env source. Use env[:attack_ms[:release_ms]][*weight]"
            raise ValueError(msg)
        attack = _parse_positive_float(parts[1], "env attack_ms") if len(parts) >= 2 else 20.0
        release = _parse_positive_float(parts[2], "env release_ms") if len(parts) >= 3 else 350.0
        return ModulationSource(
            spec=raw,
            kind="env",
            weight=weight,
            attack_ms=attack,
            release_ms=release,
        )

    if head == "audio-env":
        if len(parts) < 2:
            msg = (
                "Invalid audio-env source. "
                "Use audio-env:<path>[:attack_ms[:release_ms]][*weight]"
            )
            raise ValueError(msg)
        if len(parts) > 4:
            msg = (
                "Invalid audio-env source. Too many fields. "
                "Use audio-env:<path>[:attack_ms[:release_ms]][*weight]"
            )
            raise ValueError(msg)
        attack = _parse_positive_float(parts[2], "audio-env attack_ms") if len(parts) >= 3 else 20.0
        release = (
            _parse_positive_float(parts[3], "audio-env release_ms") if len(parts) >= 4 else 350.0
        )
        return ModulationSource(
            spec=raw,
            kind="audio-env",
            weight=weight,
            path=parts[1],
            attack_ms=attack,
            release_ms=release,
        )

    if head == "const":
        if len(parts) != 2:
            msg = "Invalid const source. Use const:<value>[*weight]"
            raise ValueError(msg)
        return ModulationSource(
            spec=raw,
            kind="const",
            weight=weight,
            value=float(parts[1]),
        )

    msg = (
        "Unsupported modulation source kind. "
        "Expected one of: lfo, env, audio-env, const."
    )
    raise ValueError(msg)


def apply_parameter_modulation(
    *,
    audio: AudioArray,
    dry_reference: AudioArray,
    sr: int,
    target: TargetName | Literal["none"],
    source_specs: tuple[str, ...],
    value_min: float,
    value_max: float,
    combine: CombineMode,
    smooth_ms: float,
) -> tuple[AudioArray, dict[str, Any] | None]:
    """Apply modulation-controlled parameter mapping to rendered audio."""
    if target == "none" or len(source_specs) == 0:
        return audio, None

    sources = parse_mod_sources(source_specs)
    control = build_control_signal(
        sources=sources,
        n_samples=audio.shape[0],
        sr=sr,
        dry_reference=dry_reference,
        combine=combine,
        smooth_ms=smooth_ms,
    )
    values = (value_min + ((value_max - value_min) * control)).astype(np.float64)

    if target in {"mix", "wet"}:
        dry = _fit_audio_to_shape(dry_reference, audio.shape[0], audio.shape[1])
        out = ((1.0 - values[:, np.newaxis]) * dry) + (values[:, np.newaxis] * audio)
    elif target == "gain-db":
        gains = np.power(10.0, values / 20.0).astype(np.float64)
        out = audio * gains[:, np.newaxis]
    else:
        msg = f"Unsupported modulation target: {target}"
        raise ValueError(msg)

    summary = {
        "target": target,
        "sources": [source.spec for source in sources],
        "combine": combine,
        "smooth_ms": float(smooth_ms),
        "value_min": float(value_min),
        "value_max": float(value_max),
        "control_mean": float(np.mean(control)),
        "control_min": float(np.min(control)),
        "control_max": float(np.max(control)),
        "value_mean": float(np.mean(values)),
        "value_min_observed": float(np.min(values)),
        "value_max_observed": float(np.max(values)),
    }
    return np.asarray(out, dtype=np.float64), summary


def build_control_signal(
    *,
    sources: list[ModulationSource],
    n_samples: int,
    sr: int,
    dry_reference: AudioArray,
    combine: CombineMode,
    smooth_ms: float,
) -> ControlArray:
    """Render and combine modulation sources into one 0..1 control stream."""
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.float64)

    bipolar_components: list[ControlArray] = []
    abs_weights: list[float] = []
    for source in sources:
        unit = _source_to_unit(source, n_samples=n_samples, sr=sr, dry_reference=dry_reference)
        bipolar = ((2.0 * unit) - 1.0) * np.float64(source.weight)
        bipolar_components.append(np.asarray(bipolar, dtype=np.float64))
        abs_weights.append(abs(float(source.weight)))

    stack = np.stack(bipolar_components, axis=0)

    if combine == "sum":
        combined = np.sum(stack, axis=0, dtype=np.float64)
    elif combine == "avg":
        denom = float(np.sum(np.asarray(abs_weights, dtype=np.float64)))
        if denom <= 1e-12:
            combined = np.zeros((n_samples,), dtype=np.float64)
        else:
            combined = np.sum(stack, axis=0, dtype=np.float64) / np.float64(denom)
    elif combine == "max":
        index = np.argmax(np.abs(stack), axis=0)
        combined = stack[index, np.arange(n_samples)]
    else:
        msg = f"Unsupported modulation combine mode: {combine}"
        raise ValueError(msg)

    combined = np.clip(combined, -1.0, 1.0)
    unit = np.asarray((combined + 1.0) * 0.5, dtype=np.float64)
    return _smooth_unit_signal(unit, sr=sr, smooth_ms=smooth_ms)


def _source_to_unit(
    source: ModulationSource,
    *,
    n_samples: int,
    sr: int,
    dry_reference: AudioArray,
) -> ControlArray:
    """Render one source into a 0..1 unit-range control stream."""
    if source.kind == "lfo":
        wave = _lfo_wave(
            shape=source.shape,
            n_samples=n_samples,
            sr=sr,
            rate_hz=source.rate_hz,
            phase_deg=source.phase_deg,
        )
        depth = float(np.clip(source.depth, 0.0, 2.0))
        return np.clip(0.5 + (0.5 * depth * wave), 0.0, 1.0).astype(np.float64)

    if source.kind == "env":
        env = _extract_envelope(
            dry_reference,
            sr=sr,
            attack_ms=source.attack_ms,
            release_ms=source.release_ms,
        )
        return _fit_control_length(env, n_samples)

    if source.kind == "audio-env":
        if source.path is None:
            raise ValueError("audio-env source requires a path")
        ext_audio, ext_sr = read_audio(source.path)
        env = _extract_envelope(
            ext_audio,
            sr=ext_sr,
            attack_ms=source.attack_ms,
            release_ms=source.release_ms,
        )
        return _fit_control_length(env, n_samples)

    if source.kind == "const":
        value = float(np.clip(source.value, 0.0, 1.0))
        return np.full((n_samples,), np.float64(value), dtype=np.float64)

    msg = f"Unsupported modulation source kind: {source.kind}"
    raise ValueError(msg)


def _extract_envelope(
    audio: AudioArray,
    *,
    sr: int,
    attack_ms: float,
    release_ms: float,
) -> ControlArray:
    """Compute normalized 0..1 envelope follower signal."""
    x = ensure_mono_or_stereo(audio)
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    env_in = np.abs(np.mean(x, axis=1)).astype(np.float64)
    env = np.zeros_like(env_in)

    attack_seconds = max(0.0001, float(attack_ms) / 1000.0)
    release_seconds = max(0.0001, float(release_ms) / 1000.0)
    attack_alpha = float(np.exp(-1.0 / (attack_seconds * float(sr))))
    release_alpha = float(np.exp(-1.0 / (release_seconds * float(sr))))

    last = np.float64(0.0)
    for idx, sample in enumerate(env_in):
        if sample > last:
            last = np.float64((attack_alpha * last) + ((1.0 - attack_alpha) * sample))
        else:
            last = np.float64((release_alpha * last) + ((1.0 - release_alpha) * sample))
        env[idx] = last

    normalizer = float(np.percentile(env, 95.0))
    if normalizer <= 1e-12:
        return np.zeros_like(env)
    return np.clip(env / np.float64(normalizer), 0.0, 1.0)


def _lfo_wave(
    *,
    shape: LFOWave,
    n_samples: int,
    sr: int,
    rate_hz: float,
    phase_deg: float,
) -> ControlArray:
    """Generate bipolar LFO wave in range [-1, 1]."""
    t = np.arange(n_samples, dtype=np.float64) / float(max(1, sr))
    cycles = (t * float(rate_hz)) + (float(phase_deg) / 360.0)
    frac = cycles - np.floor(cycles)

    if shape == "sine":
        wave = np.sin(2.0 * np.pi * cycles)
    elif shape == "triangle":
        wave = 1.0 - (4.0 * np.abs(frac - 0.5))
    elif shape == "square":
        wave = np.where(frac < 0.5, 1.0, -1.0)
    elif shape == "saw":
        wave = (2.0 * frac) - 1.0
    else:
        msg = f"Unsupported lfo shape: {shape}"
        raise ValueError(msg)
    return np.asarray(wave, dtype=np.float64)


def _fit_control_length(signal: ControlArray, n_samples: int) -> ControlArray:
    """Resize 1-D control signal to target sample length with linear interpolation."""
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.float64)
    if signal.shape[0] == n_samples:
        return signal.astype(np.float64, copy=False)
    if signal.shape[0] == 0:
        return np.zeros((n_samples,), dtype=np.float64)
    if signal.shape[0] == 1:
        return np.full((n_samples,), signal[0], dtype=np.float64)

    x_old = np.arange(signal.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, float(signal.shape[0] - 1), num=n_samples, dtype=np.float64)
    resized = np.interp(x_new, x_old, signal.astype(np.float64))
    return np.asarray(np.clip(resized, 0.0, 1.0), dtype=np.float64)


def _fit_audio_to_shape(audio: AudioArray, n_samples: int, n_channels: int) -> AudioArray:
    """Pad/truncate/tile audio so it matches requested shape."""
    x = ensure_mono_or_stereo(audio)

    if x.shape[0] < n_samples:
        pad = np.zeros((n_samples - x.shape[0], x.shape[1]), dtype=np.float64)
        x = np.concatenate((x, pad), axis=0)
    elif x.shape[0] > n_samples:
        x = x[:n_samples, :]

    if x.shape[1] == n_channels:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, n_channels, axis=1).astype(np.float64, copy=False)
    if x.shape[1] > n_channels:
        return x[:, :n_channels].astype(np.float64, copy=False)

    reps = int(np.ceil(float(n_channels) / float(max(1, x.shape[1]))))
    tiled = np.tile(x, (1, reps))
    return tiled[:, :n_channels].astype(np.float64, copy=False)


def _smooth_unit_signal(signal: ControlArray, *, sr: int, smooth_ms: float) -> ControlArray:
    """Apply one-pole smoothing to a 0..1 control signal."""
    if signal.shape[0] == 0 or smooth_ms <= 0.0:
        return signal.astype(np.float64, copy=False)

    tau = max(0.0001, float(smooth_ms) / 1000.0)
    alpha = float(np.exp(-1.0 / (tau * float(max(1, sr)))))
    out = np.empty_like(signal)
    last = np.float64(signal[0])
    out[0] = last
    for idx in range(1, signal.shape[0]):
        current = np.float64(signal[idx])
        last = np.float64((alpha * last) + ((1.0 - alpha) * current))
        out[idx] = last
    return np.clip(out, 0.0, 1.0).astype(np.float64, copy=False)


def _split_weight(spec: str) -> tuple[str, float]:
    """Extract trailing ``*weight`` suffix if present."""
    base = spec.strip()
    if "*" not in base:
        return base, 1.0
    left, right = base.rsplit("*", 1)
    try:
        weight = float(right.strip())
        return left.strip(), weight
    except ValueError:
        return base, 1.0


def _parse_lfo_shape(raw: str) -> LFOWave:
    """Parse LFO shape literal."""
    shape = raw.strip().lower()
    if shape not in {"sine", "triangle", "square", "saw"}:
        msg = "lfo shape must be one of: sine, triangle, square, saw"
        raise ValueError(msg)
    return shape  # type: ignore[return-value]


def _parse_mod_target(raw: str) -> TargetName:
    """Parse modulation target literal."""
    value = raw.strip().lower()
    if value not in {"mix", "wet", "gain-db"}:
        msg = "mod-route target must be one of: mix, wet, gain-db"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def _parse_combine_mode(raw: str) -> CombineMode:
    """Parse source-combination literal."""
    value = raw.strip().lower()
    if value not in {"sum", "avg", "max"}:
        msg = "mod-route combine must be one of: sum, avg, max"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def _parse_non_negative_float(raw: str, label: str) -> float:
    """Parse float and enforce non-negative domain."""
    value = float(raw)
    if value < 0.0:
        msg = f"{label} must be >= 0"
        raise ValueError(msg)
    return value


def _parse_positive_float(raw: str, label: str) -> float:
    """Parse float and enforce > 0 domain."""
    value = float(raw)
    if value <= 0.0:
        msg = f"{label} must be > 0"
        raise ValueError(msg)
    return value
