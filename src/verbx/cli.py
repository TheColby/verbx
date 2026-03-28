"""Typer CLI entrypoint for verbx.

Commands are grouped by workflow:
- top-level render/analyze/suggest/presets
- ``ir`` synthesis/inspection
- ``cache`` management
- ``batch`` orchestration
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from difflib import get_close_matches
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from threading import Lock
from typing import Any, Literal, TypedDict, cast

import numpy as np
import soundfile as sf
import typer
from click.core import ParameterSource
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from verbx import __version__
from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import write_framewise_csv
from verbx.config import (
    AmbiChannelOrder,
    AmbiDecodeTo,
    AmbiEncodeFrom,
    AmbiNormalization,
    AutomationMode,
    ChannelLayout,
    DeviceName,
    EngineName,
    FDNNonlinearityMode,
    FDNSpatialCouplingMode,
    FeatureGuidePolicy,
    IRMatrixLayout,
    IRMode,
    IRMorphMismatchPolicy,
    IRNormalize,
    ModCombine,
    ModTarget,
    NormalizeStage,
    OutputPeakNorm,
    OutputSubtype,
    RenderConfig,
)
from verbx.core.accel import (
    cuda_available,
    is_apple_silicon,
    resolve_device,
    resolve_device_for_engine,
)
from verbx.core.augmentation import (
    AugmentationBuild,
    augmentation_profile_names,
    augmentation_profiles,
    build_augmentation_manifest_template,
    build_augmentation_plans,
    render_config_snapshot,
)
from verbx.core.automation import (
    CONV_AUTOMATION_TARGETS,
    ENGINE_AUTOMATION_TARGETS,
    FEATURE_GUIDE_POLICY_CHOICES,
    collect_automation_targets,
    parse_automation_clamp_overrides,
    parse_automation_point_specs,
)
from verbx.core.batch_scheduler import (
    BatchJobResult,
    BatchJobSpec,
    BatchSchedulePolicy,
    estimate_job_cost,
    order_jobs,
    run_parallel_batch,
)
from verbx.core.control_targets import RT60_DEFAULT_SECONDS, RT60_MAX_SECONDS, RT60_MIN_SECONDS
from verbx.core.fdn_capabilities import (
    FDN_GRAPH_TOPOLOGY_CHOICES,
    FDN_LINK_FILTER_CHOICES,
    FDN_MATRIX_CHOICES,
)
from verbx.core.fdn_capabilities import (
    normalize_fdn_graph_topology_name as _shared_normalize_fdn_graph_topology_name,
)
from verbx.core.fdn_capabilities import (
    normalize_fdn_link_filter_name as _shared_normalize_fdn_link_filter_name,
)
from verbx.core.fdn_capabilities import (
    normalize_fdn_matrix_name as _shared_normalize_fdn_matrix_name,
)
from verbx.core.feature_vector import (
    parse_feature_vector_lane_specs,
)
from verbx.core.immersive import (
    LAYOUT_CHANNELS as IMMERSIVE_LAYOUT_CHANNELS,
)
from verbx.core.immersive import (
    QueueWorkerConfig,
    build_qc_gates,
    evaluate_immersive_qc,
    generate_immersive_handoff_package,
    run_file_queue_worker,
    summarize_file_queue,
    validate_layout_hint,
)
from verbx.core.modulation import parse_mod_route_spec, parse_mod_sources
from verbx.core.pipeline import run_render_pipeline
from verbx.core.schema_versions import (
    AUGMENT_QA_BUNDLE_VERSION,
    AUGMENT_SUMMARY_VERSION,
    BATCH_CHECKPOINT_VERSION,
    BATCH_MANIFEST_VERSION,
    IMMERSIVE_QUEUE_VERSION,
    IR_MORPH_SWEEP_VERSION,
)
from verbx.core.spatial import (
    ambisonic_channel_count,
    normalize_ambisonic_metadata,
)
from verbx.core.tempo import parse_pre_delay_ms
from verbx.io.audio import read_audio, validate_audio_path
from verbx.ir.fitting import (
    IRFitCandidate,
    IRFitScore,
    IRFitTarget,
    build_ir_fit_candidates,
    derive_ir_fit_target,
    score_ir_candidate,
)
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir, write_ir_artifacts
from verbx.ir.metrics import analyze_ir
from verbx.ir.morph import (
    IRMorphConfig,
    generate_or_load_cached_morphed_ir,
    normalize_ir_morph_mismatch_policy_name,
    normalize_ir_morph_mode_name,
    resolve_blend_mix_values,
    validate_ir_morph_mismatch_policy_name,
    validate_ir_morph_mode_name,
)
from verbx.ir.shaping import apply_ir_shaping
from verbx.ir.sofa import extract_sofa_ir, read_sofa_info
from verbx.ir.tuning import analyze_audio_for_tuning, parse_frequency_hz
from verbx.logging import configure_logging
from verbx.presets.default_presets import preset_names, resolve_preset

IRFileFormat = Literal["auto", "wav", "flac", "aiff", "aif", "ogg", "caf"]
_FDN_MATRIX_CHOICES = set(FDN_MATRIX_CHOICES)
_FDN_GRAPH_TOPOLOGY_CHOICES = set(FDN_GRAPH_TOPOLOGY_CHOICES)
_FDN_LINK_FILTER_CHOICES = set(FDN_LINK_FILTER_CHOICES)
_FDN_SPATIAL_COUPLING_CHOICES = {
    "none",
    "adjacent",
    "front_rear",
    "bed_top",
    "all_to_all",
}
_FDN_NONLINEARITY_CHOICES = {
    "none",
    "tanh",
    "softclip",
}
_IR_ROUTE_MAP_CHOICES = {
    "auto",
    "diagonal",
    "broadcast",
    "full",
}
_CONV_ROUTE_CURVE_CHOICES = {
    "linear",
    "equal-power",
}
_AMBI_NORMALIZATION_CHOICES = {
    "auto",
    "sn3d",
    "n3d",
    "fuma",
}
_AMBI_CHANNEL_ORDER_CHOICES = {
    "auto",
    "acn",
    "fuma",
}
_AMBI_ENCODE_CHOICES = {
    "none",
    "mono",
    "stereo",
}
_AMBI_DECODE_CHOICES = {
    "none",
    "stereo",
}
_AUTOMATION_MODE_CHOICES = {
    "auto",
    "sample",
    "block",
}
_LAYOUT_CHANNELS = dict(IMMERSIVE_LAYOUT_CHANNELS)
_IR_MORPH_MODE_CHOICES = {
    "linear",
    "equal-power",
    "spectral",
    "envelope-aware",
}
_IR_MORPH_MISMATCH_POLICY_CHOICES = {
    "coerce",
    "strict",
}


class LuckyIRProcessConfig(TypedDict):
    """Typed config payload for ``ir process --lucky`` randomization."""

    damping: float
    lowcut: float | None
    highcut: float | None
    tilt: float
    normalize: Literal["none", "peak", "rms"]
    peak_dbfs: float
    target_lufs: float | None
    true_peak: bool


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help="Extreme reverb CLI with scalable DSP architecture.",
)
ir_app = typer.Typer(help="Impulse response workflows.")
cache_app = typer.Typer(help="IR cache inspection and cleanup.")
batch_app = typer.Typer(help="Batch manifest generation and rendering.")
immersive_app = typer.Typer(help="Immersive production interoperability workflows.")
immersive_queue_app = typer.Typer(help="Distributed immersive queue workflows.")

app.add_typer(ir_app, name="ir")
app.add_typer(cache_app, name="cache")
app.add_typer(batch_app, name="batch")
app.add_typer(immersive_app, name="immersive")
immersive_app.add_typer(immersive_queue_app, name="queue")

console = Console()


@contextmanager
def _processing_status(description: str, *, enabled: bool = True) -> Any:
    """Render a single-task CLI status bar for one processing stage."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not enabled,
    )
    progress.start()
    task = progress.add_task(str(description), total=1)
    try:
        yield progress
        progress.update(task, completed=1)
    finally:
        progress.stop()


class _BatchStatusBar:
    """Compact progress/status bar for batch-style CLI workflows."""

    __slots__ = ("_done", "_enabled", "_label", "_progress", "_task", "_total")

    def __init__(self, *, total: int, label: str, enabled: bool = True) -> None:
        self._total = max(1, int(total))
        self._label = str(label)
        self._enabled = bool(enabled)
        self._done = 0
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=24),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            disable=not self._enabled,
        )
        self._task: TaskID | None = None

    def __enter__(self) -> _BatchStatusBar:
        self._progress.start()
        self._task = self._progress.add_task(
            f"{self._label} 0/{self._total}",
            total=self._total,
        )
        return self

    def advance(self, *, detail: str | None = None) -> None:
        """Advance completed job count and refresh task description."""
        self._done = min(self._total, self._done + 1)
        if self._task is None:
            return
        suffix = "" if detail is None else f" ({detail})"
        self._progress.update(
            self._task,
            completed=self._done,
            description=f"{self._label} {self._done}/{self._total}{suffix}",
        )

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        self._progress.stop()


@app.command()
def version() -> None:
    """Print CLI/package version."""
    console.print(f"verbx {__version__}")


@app.command()
def quickstart(
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Run startup readiness checks for first-run confidence.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero when --verify finds one or more failed checks.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional path to write quickstart verification/smoke JSON.",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Run a tiny end-to-end render smoke test with synthetic input audio.",
    ),
    smoke_out_dir: Path | None = typer.Option(
        None,
        "--smoke-out-dir",
        resolve_path=True,
        help="Optional output directory for smoke-test artifacts.",
    ),
) -> None:
    """Print minimal copy/paste commands for first successful renders."""
    commands = [
        (
            "Homebrew install (macOS) + extreme render",
            "brew tap thecolby/verbx && brew install thecolby/verbx/verbx && "
            "verbx render ../in.wav out.wav --engine algo --rt60 12 --wet 0.88 --dry 0.12",
        ),
        (
            "Source install + extreme algorithmic render",
            "git clone https://github.com/TheColby/verbx.git && cd verbx && "
            "./scripts/install.sh && verbx render ../in.wav out.wav "
            "--engine algo --rt60 12 --wet 0.88 --dry 0.12",
        ),
        (
            "Analyze then suggested settings",
            "verbx analyze in.wav --lufs --json-out analysis.json && verbx suggest in.wav",
        ),
        (
            "Convolution render with IR",
            "verbx render in.wav out_conv.wav --engine conv --ir hall.wav --wet 0.75 --dry 0.25",
        ),
    ]
    table = Table(title="verbx Quickstart")
    table.add_column("Workflow", style="cyan")
    table.add_column("Command", style="white")
    for name, cmd in commands:
        table.add_row(name, cmd)
    console.print(table)

    if strict and not verify and not smoke_test:
        raise typer.BadParameter("--strict requires --verify and/or --smoke-test.")
    if json_out is not None and not verify and not smoke_test:
        raise typer.BadParameter("--json-out requires --verify and/or --smoke-test.")

    report: dict[str, Any] | None = None
    smoke_report: dict[str, Any] | None = None
    if verify:
        report = _collect_runtime_diagnostics()
        _print_runtime_checks_table(report, title="verbx Quickstart Verify")
    if smoke_test:
        with _processing_status("Quickstart render smoke test"):
            smoke_report = _run_render_smoke_test(out_dir=smoke_out_dir)
        _print_render_smoke_test_table(smoke_report, title="verbx Quickstart Smoke Test")

    if json_out is not None:
        if verify and not smoke_test:
            payload = report if report is not None else {}
        elif smoke_test and not verify:
            payload = {
                "schema": "quickstart-smoke-v1",
                "smoke_test": smoke_report,
                "ready": bool(smoke_report is not None and smoke_report.get("ok", False)),
            }
        else:
            payload = {
                "schema": "quickstart-verify-smoke-v1",
                "diagnostics": report,
                "smoke_test": smoke_report,
                "ready": bool(
                    report is not None
                    and report.get("ready", False)
                    and smoke_report is not None
                    and smoke_report.get("ok", False)
                ),
            }
        _write_json_atomic(json_out.resolve(), cast(dict[str, Any], payload))

    if strict:
        verify_ok = True if report is None else bool(report.get("ready", False))
        smoke_ok = True if smoke_report is None else bool(smoke_report.get("ok", False))
        if not verify_ok or not smoke_ok:
            raise typer.Exit(code=2)


@app.command()
def doctor(
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional path to write machine-readable diagnostics JSON.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero when startup checks fail.",
    ),
    render_smoke_test: bool = typer.Option(
        False,
        "--render-smoke-test",
        help="Run a tiny end-to-end render smoke test after diagnostics.",
    ),
    smoke_out_dir: Path | None = typer.Option(
        None,
        "--smoke-out-dir",
        resolve_path=True,
        help="Optional output directory for doctor smoke-test artifacts.",
    ),
) -> None:
    """Print runtime diagnostics for launch-day troubleshooting."""
    report = _collect_runtime_diagnostics()

    table = Table(title="verbx Doctor")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("verbx_version", str(report["verbx_version"]))
    table.add_row("python_version", str(report["python_version"]))
    table.add_row("platform", str(report["platform"]))
    table.add_row("machine", str(report["machine"]))
    table.add_row("cpu_count", str(report["cpu_count"]))
    table.add_row("apple_silicon", str(report["apple_silicon"]))
    table.add_row("cuda_available", str(report["cuda_available"]))
    table.add_row("device_auto", str(report["device_auto"]))
    table.add_row(
        "auto_algo_device",
        str(report["engine_auto_resolution"]["algo"]["engine_device"]),
    )
    table.add_row(
        "auto_conv_device",
        str(report["engine_auto_resolution"]["conv"]["engine_device"]),
    )
    table.add_row("cupy_version", str(report["dependencies"].get("cupy")))
    table.add_row("status", str(report.get("status", "")))
    table.add_row("checks_total", str(report.get("checks_total", 0)))
    table.add_row("checks_failed", str(report.get("failed_checks", 0)))
    console.print(table)
    _print_runtime_checks_table(report, title="verbx Doctor Checks")
    recommendations = report.get("recommendations", [])
    if isinstance(recommendations, list) and len(recommendations) > 0:
        rec_table = Table(title="verbx Doctor Recommendations")
        rec_table.add_column("#", style="cyan", justify="right")
        rec_table.add_column("Recommendation", style="white")
        for idx, text in enumerate(recommendations, start=1):
            rec_table.add_row(str(idx), str(text))
        console.print(rec_table)

    smoke_report: dict[str, Any] | None = None
    if render_smoke_test:
        with _processing_status("Doctor render smoke test"):
            smoke_report = _run_render_smoke_test(out_dir=smoke_out_dir)
        _print_render_smoke_test_table(smoke_report, title="verbx Doctor Smoke Test")

    if json_out is not None:
        if smoke_report is None:
            _write_json_atomic(json_out.resolve(), report)
        else:
            payload = dict(report)
            payload["render_smoke_test"] = smoke_report
            payload["ready"] = bool(payload.get("ready", False) and smoke_report.get("ok", False))
            _write_json_atomic(json_out.resolve(), payload)
    if strict and (
        int(report.get("failed_checks", 0)) > 0
        or (smoke_report is not None and not bool(smoke_report.get("ok", False)))
    ):
        raise typer.Exit(code=2)


def _dependency_versions() -> dict[str, str | None]:
    """Return optional dependency versions used by doctor/quickstart checks."""
    deps: dict[str, str | None] = {}
    try:
        for package_name in ("numpy", "soundfile", "rich", "typer", "cupy"):
            try:
                deps[package_name] = str(pkg_version(package_name))
            except PackageNotFoundError:
                deps[package_name] = None
    except Exception:
        deps = {"numpy": None, "soundfile": None, "rich": None, "typer": None, "cupy": None}
    return deps


def _collect_runtime_diagnostics() -> dict[str, Any]:
    """Build a runtime diagnostics payload with startup readiness checks."""
    report: dict[str, Any] = {
        "verbx_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "apple_silicon": bool(is_apple_silicon()),
        "cuda_available": bool(cuda_available()),
        "device_auto": str(resolve_device("auto")),
    }
    algo_device, algo_platform = resolve_device_for_engine("auto", "algo")
    conv_device, conv_platform = resolve_device_for_engine("auto", "conv")
    report["engine_auto_resolution"] = {
        "algo": {"engine_device": str(algo_device), "platform_device": str(algo_platform)},
        "conv": {"engine_device": str(conv_device), "platform_device": str(conv_platform)},
    }
    report["dependencies"] = _dependency_versions()
    checks = _runtime_checks(report)
    failed_checks = [item for item in checks if not bool(item.get("ok", False))]
    report["checks"] = checks
    report["checks_total"] = len(checks)
    report["failed_checks"] = len(failed_checks)
    report["issues"] = failed_checks
    report["ready"] = len(failed_checks) == 0
    report["status"] = "ok" if len(failed_checks) == 0 else "warn"
    report["recommendations"] = _runtime_recommendations(report)
    return report


def _runtime_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute actionable startup checks for first-run readiness."""
    dependencies = report.get("dependencies", {})
    deps = dependencies if isinstance(dependencies, dict) else {}
    auto_device = str(report.get("device_auto", "cpu"))

    return [
        {
            "id": "python_min",
            "name": "Python >= 3.11",
            "ok": bool(sys.version_info >= (3, 11)),
            "value": str(report.get("python_version", "")),
            "hint": "Use Python 3.11 or newer.",
        },
        {
            "id": "numpy_present",
            "name": "numpy installed",
            "ok": deps.get("numpy") is not None,
            "value": str(deps.get("numpy")),
            "hint": "Install dependencies with ./scripts/install.sh.",
        },
        {
            "id": "soundfile_present",
            "name": "soundfile installed",
            "ok": deps.get("soundfile") is not None,
            "value": str(deps.get("soundfile")),
            "hint": "Install dependencies with ./scripts/install.sh.",
        },
        {
            "id": "wav_write",
            "name": "WAV write support",
            "ok": bool(sf.check_format("WAV")),
            "value": "WAV",
            "hint": "Install/repair libsndfile for local WAV I/O.",
        },
        {
            "id": "auto_device",
            "name": "auto device resolves",
            "ok": auto_device in {"cpu", "mps", "cuda"},
            "value": auto_device,
            "hint": "Run `verbx doctor --json-out doctor.json` and check accelerator settings.",
        },
    ]


def _runtime_recommendations(report: dict[str, Any]) -> list[str]:
    """Derive concise recommendations from diagnostics payload."""
    recs: list[str] = []
    failed = report.get("issues", [])
    if isinstance(failed, list):
        for item in failed:
            if not isinstance(item, dict):
                continue
            hint = str(item.get("hint", "")).strip()
            if hint != "" and hint not in recs:
                recs.append(hint)

    dependencies = report.get("dependencies", {})
    deps = dependencies if isinstance(dependencies, dict) else {}
    if bool(report.get("cuda_available", False)) and deps.get("cupy") is None:
        recs.append("CUDA is available; install CuPy to enable accelerated convolution.")
    if bool(report.get("apple_silicon", False)) and str(report.get("device_auto", "")) == "cpu":
        recs.append(
            "Apple Silicon host is falling back to CPU; verify MPS support in your runtime."
        )
    if len(recs) == 0:
        recs.append(
            "Runtime checks are clean. Run `verbx quickstart --verify --strict` before demos."
        )
    return recs


def _run_render_smoke_test(*, out_dir: Path | None) -> dict[str, Any]:
    """Run a tiny end-to-end render to validate practical startup readiness."""
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    root: Path
    if out_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="verbx_smoke_")
        root = Path(temp_dir.name).resolve()
    else:
        root = out_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)

    infile = root / "smoke_in.wav"
    outfile = root / "smoke_out.wav"
    sr = 24_000
    num_samples = round(0.35 * float(sr))
    timeline = np.arange(num_samples, dtype=np.float64) / float(sr)
    audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * timeline)).reshape(-1, 1).astype(np.float64)
    try:
        sf.write(str(infile), audio, sr, subtype="DOUBLE")
        config = RenderConfig(
            engine="algo",
            rt60=0.8,
            wet=0.35,
            dry=0.65,
            repeat=1,
            output_subtype="float64",
            silent=True,
            progress=False,
        )
        report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
        info = sf.info(str(outfile))
        output_frames = int(info.frames)
        ok = bool(outfile.exists() and output_frames > num_samples)
        return {
            "ok": ok,
            "infile": str(infile),
            "outfile": str(outfile),
            "sample_rate": int(info.samplerate),
            "input_frames": int(num_samples),
            "output_frames": int(output_frames),
            "engine": str(report.get("engine", "")),
            "error": "",
        }
    except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
        return {
            "ok": False,
            "infile": str(infile),
            "outfile": str(outfile),
            "sample_rate": int(sr),
            "input_frames": int(num_samples),
            "output_frames": 0,
            "engine": "algo",
            "error": str(exc),
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _print_render_smoke_test_table(report: dict[str, Any], *, title: str) -> None:
    """Print smoke-test status table for quickstart/doctor output."""
    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("status", "PASS" if bool(report.get("ok", False)) else "FAIL")
    table.add_row("engine", str(report.get("engine", "")))
    table.add_row("sample_rate", str(report.get("sample_rate", "")))
    table.add_row("input_frames", str(report.get("input_frames", "")))
    table.add_row("output_frames", str(report.get("output_frames", "")))
    table.add_row("infile", str(report.get("infile", "")))
    table.add_row("outfile", str(report.get("outfile", "")))
    error_text = str(report.get("error", "")).strip()
    if error_text != "":
        table.add_row("error", error_text)
    console.print(table)


def _print_runtime_checks_table(report: dict[str, Any], *, title: str) -> None:
    """Render startup checks table used by quickstart and doctor."""
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return
    table = Table(title=title)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Value", style="white")
    for item in checks:
        if not isinstance(item, dict):
            continue
        status = "PASS" if bool(item.get("ok", False)) else "FAIL"
        table.add_row(
            str(item.get("name", "")),
            status,
            str(item.get("value", "")),
        )
    console.print(table)


@app.command()
def render(
    ctx: typer.Context,
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    outfile: Path = typer.Argument(..., resolve_path=True),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=(
            "Named preset baseline (see `verbx presets`). Explicitly supplied CLI options "
            "override preset values."
        ),
    ),
    engine: EngineName = typer.Option("auto", "--engine", help="Engine: conv, algo, or auto."),
    rt60: float = typer.Option(
        RT60_DEFAULT_SECONDS, "--rt60", min=RT60_MIN_SECONDS, max=RT60_MAX_SECONDS
    ),
    wet: float = typer.Option(0.8, "--wet", min=0.0, max=1.0),
    dry: float = typer.Option(0.2, "--dry", min=0.0, max=1.0),
    repeat: int = typer.Option(1, "--repeat", min=1),
    freeze: bool = typer.Option(False, "--freeze", help="Enable freeze segment mode."),
    start: float | None = typer.Option(None, "--start", min=0.0),
    end: float | None = typer.Option(None, "--end", min=0.0),
    pre_delay_ms: float = typer.Option(20.0, "--pre-delay-ms", min=0.0),
    pre_delay: str | None = typer.Option(None, "--pre-delay"),
    bpm: float | None = typer.Option(None, "--bpm", min=1.0),
    damping: float = typer.Option(0.45, "--damping", min=0.0, max=1.0),
    width: float = typer.Option(1.0, "--width", min=0.0, max=2.0),
    mod_depth_ms: float = typer.Option(2.0, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.1, "--mod-rate-hz", min=0.0),
    mod_target: ModTarget = typer.Option(
        "none",
        "--mod-target",
        help="Dynamic parameter target: none, mix/wet, or gain-db.",
    ),
    mod_source: list[str] | None = typer.Option(
        None,
        "--mod-source",
        help=(
            "Repeatable modulation source spec. "
            "Examples: lfo:sine:0.08:1.0*0.7, env:20:350, "
            "audio-env:sidechain.wav:10:200, const:0.5."
        ),
    ),
    mod_route: list[str] | None = typer.Option(
        None,
        "--mod-route",
        help=(
            "Repeatable advanced route: "
            "<target>:<min>:<max>:<combine>:<smooth_ms>:<src1>,<src2>,... "
            "(target: mix|wet|gain-db)."
        ),
    ),
    mod_min: float = typer.Option(
        0.0,
        "--mod-min",
        help="Minimum mapped value for the modulation target.",
    ),
    mod_max: float = typer.Option(
        1.0,
        "--mod-max",
        help="Maximum mapped value for the modulation target.",
    ),
    mod_combine: ModCombine = typer.Option(
        "sum",
        "--mod-combine",
        help="How multiple sources are combined: sum, avg, or max.",
    ),
    mod_smooth_ms: float = typer.Option(
        20.0,
        "--mod-smooth-ms",
        min=0.0,
        help="One-pole smoothing time for modulation control signals.",
    ),
    allpass_stages: int = typer.Option(
        6,
        "--allpass-stages",
        min=0,
        max=64,
        help="Number of Schroeder allpass diffusion stages (0 disables diffusion).",
    ),
    allpass_gain: str = typer.Option(
        "0.7",
        "--allpass-gain",
        help=(
            "Allpass gain. Use one value (e.g. 0.7) for all stages, or a "
            "comma-separated list (e.g. 0.72,0.70,0.68,0.66) for per-stage gains."
        ),
    ),
    allpass_delays_ms: str | None = typer.Option(
        None,
        "--allpass-delays-ms",
        help=(
            "Optional comma-separated allpass delay list in milliseconds. Example: 5,7,11,17,23,29"
        ),
    ),
    comb_delays_ms: str | None = typer.Option(
        None,
        "--comb-delays-ms",
        help=(
            "Optional comma-separated FDN comb-like delay list in milliseconds. "
            "Example: 31,37,41,43,47,53,59,67"
        ),
    ),
    fdn_lines: int = typer.Option(
        8,
        "--fdn-lines",
        min=1,
        max=64,
        help="FDN line count used when --comb-delays-ms is not provided.",
    ),
    fdn_matrix: str = typer.Option(
        "auto",
        "--fdn-matrix",
        help=(
            "FDN matrix topology: hadamard, householder, random_orthogonal, "
            "circulant, elliptic, tv_unitary, graph, or sdn_hybrid. "
            "Default resolves to hadamard."
        ),
    ),
    fdn_tv_rate_hz: float = typer.Option(
        0.0,
        "--fdn-tv-rate-hz",
        min=0.0,
        help="Block-rate update speed for --fdn-matrix tv_unitary (Hz).",
    ),
    fdn_tv_depth: float = typer.Option(
        0.0,
        "--fdn-tv-depth",
        min=0.0,
        max=1.0,
        help="Blend depth for --fdn-matrix tv_unitary (0..1).",
    ),
    fdn_dfm_delays_ms: str | None = typer.Option(
        None,
        "--fdn-dfm-delays-ms",
        help=(
            "Optional delay-feedback-matrix delays in milliseconds. "
            "Provide one value for broadcast or one per FDN line."
        ),
    ),
    fdn_sparse: bool = typer.Option(
        False,
        "--fdn-sparse/--no-fdn-sparse",
        help="Enable sparse high-order FDN pair-mixing mode.",
    ),
    fdn_sparse_degree: int = typer.Option(
        2,
        "--fdn-sparse-degree",
        min=1,
        max=16,
        help="Number of sparse pair-mixing stages used when --fdn-sparse is enabled.",
    ),
    fdn_cascade: bool = typer.Option(
        False,
        "--fdn-cascade/--no-fdn-cascade",
        help="Enable nested/cascaded FDN mode (small fast network into late network).",
    ),
    fdn_cascade_mix: float = typer.Option(
        0.35,
        "--fdn-cascade-mix",
        min=0.0,
        max=1.0,
        help="Injection amount from nested FDN into the main late-field network (0..1).",
    ),
    fdn_cascade_delay_scale: float = typer.Option(
        0.5,
        "--fdn-cascade-delay-scale",
        min=0.2,
        max=1.0,
        help="Delay scaling for nested FDN relative to primary FDN delays (0.2..1.0).",
    ),
    fdn_cascade_rt60_ratio: float = typer.Option(
        0.55,
        "--fdn-cascade-rt60-ratio",
        min=0.1,
        max=1.0,
        help="RT60 ratio for nested FDN relative to --rt60 (0.1..1.0).",
    ),
    fdn_rt60_low: float | None = typer.Option(
        None,
        "--fdn-rt60-low",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="Low-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_mid: float | None = typer.Option(
        None,
        "--fdn-rt60-mid",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="Mid-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_high: float | None = typer.Option(
        None,
        "--fdn-rt60-high",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="High-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_tilt: float = typer.Option(
        0.0,
        "--fdn-rt60-tilt",
        min=-1.0,
        max=1.0,
        help=(
            "Jot-style low/high RT skew around mid band (-1..1). "
            "Positive extends low-band decay and shortens highs."
        ),
    ),
    fdn_tonal_correction_strength: float = typer.Option(
        0.0,
        "--fdn-tonal-correction-strength",
        min=0.0,
        max=1.0,
        help=(
            "Track C tonal correction strength for multiband/tilted FDN response (0..1). "
            "Higher values apply stronger decay-color equalization."
        ),
    ),
    fdn_xover_low_hz: float = typer.Option(
        250.0,
        "--fdn-xover-low-hz",
        min=20.0,
        help="Low/mid crossover frequency used by multiband FDN decay shaping.",
    ),
    fdn_xover_high_hz: float = typer.Option(
        4_000.0,
        "--fdn-xover-high-hz",
        min=100.0,
        help="Mid/high crossover frequency used by multiband FDN decay shaping.",
    ),
    fdn_link_filter: str = typer.Option(
        "none",
        "--fdn-link-filter",
        help=("Feedback-link filter mode inside the FDN matrix path: none, lowpass, or highpass."),
    ),
    fdn_link_filter_hz: float = typer.Option(
        2_500.0,
        "--fdn-link-filter-hz",
        min=20.0,
        help="Cutoff frequency used by --fdn-link-filter (Hz).",
    ),
    fdn_link_filter_mix: float = typer.Option(
        1.0,
        "--fdn-link-filter-mix",
        min=0.0,
        max=1.0,
        help="Wet mix of feedback-link filter processing (0..1).",
    ),
    fdn_graph_topology: str = typer.Option(
        "ring",
        "--fdn-graph-topology",
        help="Graph topology for --fdn-matrix graph: ring, path, star, or random.",
    ),
    fdn_graph_degree: int = typer.Option(
        2,
        "--fdn-graph-degree",
        min=1,
        max=32,
        help="Graph neighborhood/connectivity degree for --fdn-matrix graph.",
    ),
    fdn_graph_seed: int = typer.Option(
        2026,
        "--fdn-graph-seed",
        help="Deterministic seed used to build graph-structured FDN pairings.",
    ),
    fdn_spatial_coupling_mode: FDNSpatialCouplingMode = typer.Option(
        "none",
        "--fdn-spatial-coupling-mode",
        help=(
            "Directional wet-bus coupling mode: none, adjacent, front_rear, bed_top, all_to_all."
        ),
    ),
    fdn_spatial_coupling_strength: float = typer.Option(
        0.0,
        "--fdn-spatial-coupling-strength",
        min=0.0,
        max=1.0,
        help="Wet-bus directional coupling amount (0..1).",
    ),
    fdn_nonlinearity: FDNNonlinearityMode = typer.Option(
        "none",
        "--fdn-nonlinearity",
        help="Optional in-loop nonlinearity: none, tanh, or softclip.",
    ),
    fdn_nonlinearity_amount: float = typer.Option(
        0.0,
        "--fdn-nonlinearity-amount",
        min=0.0,
        max=1.0,
        help="Blend amount for in-loop nonlinearity shaping (0..1).",
    ),
    fdn_nonlinearity_drive: float = typer.Option(
        1.0,
        "--fdn-nonlinearity-drive",
        min=0.1,
        max=8.0,
        help="Drive multiplier for in-loop nonlinearity shaping.",
    ),
    room_size_macro: float = typer.Option(
        0.0,
        "--room-size-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual room-size macro (-1..1) mapped to decay-time and spacing behavior.",
    ),
    clarity_macro: float = typer.Option(
        0.0,
        "--clarity-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual clarity macro (-1..1) mapped to decay, damping, and wet balance.",
    ),
    warmth_macro: float = typer.Option(
        0.0,
        "--warmth-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual warmth macro (-1..1) mapped to damping and spectral decay tilt.",
    ),
    envelopment_macro: float = typer.Option(
        0.0,
        "--envelopment-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual envelopment macro (-1..1) mapped to width/decorrelation emphasis.",
    ),
    beast_mode: int = typer.Option(
        1,
        "--beast-mode",
        min=1,
        max=100,
        help=(
            "Scales core reverb parameters by an intensity multiplier (1-100) "
            "to push denser, longer, freeze-like tails."
        ),
    ),
    ir: Path | None = typer.Option(None, "--ir", exists=True, readable=True, resolve_path=True),
    ir_blend: list[Path] | None = typer.Option(
        None,
        "--ir-blend",
        exists=True,
        readable=True,
        resolve_path=True,
        help=(
            "Repeatable additional IR path for render-time convolution blending. "
            "Requires convolution render path."
        ),
    ),
    ir_blend_mix: list[float] | None = typer.Option(
        None,
        "--ir-blend-mix",
        min=0.0,
        max=1.0,
        help=(
            "Repeatable blend coefficient for each --ir-blend IR (0..1). "
            "Provide one value to broadcast to all blend IRs."
        ),
    ),
    ir_blend_mode: str = typer.Option(
        "equal-power",
        "--ir-blend-mode",
        help="IR blend morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    ir_blend_early_ms: float = typer.Option(
        80.0,
        "--ir-blend-early-ms",
        min=0.0,
        help="Early/late split time (ms) used by envelope-aware and split blending modes.",
    ),
    ir_blend_early_alpha: float | None = typer.Option(
        None,
        "--ir-blend-early-alpha",
        min=0.0,
        max=1.0,
        help="Optional override alpha for early-reflection blend region.",
    ),
    ir_blend_late_alpha: float | None = typer.Option(
        None,
        "--ir-blend-late-alpha",
        min=0.0,
        max=1.0,
        help="Optional override alpha for late-tail blend region.",
    ),
    ir_blend_align_decay: bool = typer.Option(
        True,
        "--ir-blend-align-decay/--no-ir-blend-align-decay",
        help="Enable RT60 alignment before morphing to stabilize blend trajectories.",
    ),
    ir_blend_phase_coherence: float = typer.Option(
        0.75,
        "--ir-blend-phase-coherence",
        min=0.0,
        max=1.0,
        help="Phase-coherence safeguard strength for spectral/envelope-aware blending.",
    ),
    ir_blend_spectral_smooth_bins: int = typer.Option(
        3,
        "--ir-blend-spectral-smooth-bins",
        min=0,
        max=128,
        help="Frequency smoothing radius (FFT bins) used by spectral blend modes.",
    ),
    ir_blend_mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--ir-blend-mismatch-policy",
        help=(
            "Mismatch behavior for blend-source sample-rate/channel/duration differences: "
            "coerce (resample/align) or strict (fail)."
        ),
    ),
    ir_blend_cache_dir: str = typer.Option(
        ".verbx_cache/ir_morph",
        "--ir-blend-cache-dir",
        help="Cache directory for blended/morphed IR artifacts used by render workflow.",
    ),
    self_convolve: bool = typer.Option(
        False,
        "--self-convolve",
        help=(
            "Use INFILE as its own IR and force fast partitioned convolution "
            "(equivalent to --engine conv --ir INFILE)."
        ),
    ),
    ir_route_map: str = typer.Option(
        "auto",
        "--ir-route-map",
        help="Convolution route-map mode: auto, diagonal, broadcast, or full.",
    ),
    input_layout: ChannelLayout = typer.Option(
        "auto",
        "--input-layout",
        help=(
            "Input signal channel layout: auto, mono, stereo, LCR, 5.1, 7.1, "
            "7.1.2, 7.1.4, 7.2.4, 8.0, 16.0, 64.4"
        ),
    ),
    output_layout: ChannelLayout = typer.Option(
        "auto",
        "--output-layout",
        help=(
            "Output signal channel layout: auto, mono, stereo, LCR, 5.1, 7.1, "
            "7.1.2, 7.1.4, 7.2.4, 8.0, 16.0, 64.4"
        ),
    ),
    ir_normalize: IRNormalize = typer.Option("peak", "--ir-normalize"),
    ir_matrix_layout: IRMatrixLayout = typer.Option("output-major", "--ir-matrix-layout"),
    conv_route_start: str | None = typer.Option(
        None,
        "--conv-route-start",
        help="Convolution trajectory start position (index or alias, e.g. left, rear-left).",
    ),
    conv_route_end: str | None = typer.Option(
        None,
        "--conv-route-end",
        help="Convolution trajectory end position (index or alias).",
    ),
    conv_route_curve: str = typer.Option(
        "equal-power",
        "--conv-route-curve",
        help="Convolution trajectory curve: linear or equal-power.",
    ),
    ambi_order: int = typer.Option(
        0,
        "--ambi-order",
        min=0,
        max=7,
        help="Ambisonics order (0 disables Ambisonics-specific processing).",
    ),
    ambi_normalization: AmbiNormalization = typer.Option(
        "auto",
        "--ambi-normalization",
        help="Ambisonics normalization convention: auto, sn3d, n3d, or fuma.",
    ),
    channel_order: AmbiChannelOrder = typer.Option(
        "auto",
        "--channel-order",
        help="Ambisonics channel order convention: auto, acn, or fuma.",
    ),
    ambi_encode_from: AmbiEncodeFrom = typer.Option(
        "none",
        "--ambi-encode-from",
        help="Encode input bus into FOA before render: none, mono, or stereo.",
    ),
    ambi_decode_to: AmbiDecodeTo = typer.Option(
        "none",
        "--ambi-decode-to",
        help="Decode Ambisonics output after render: none or stereo.",
    ),
    ambi_rotate_yaw_deg: float = typer.Option(
        0.0,
        "--ambi-rotate-yaw-deg",
        help="Listener yaw rotation in degrees applied in Ambisonic domain.",
    ),
    algo_decorrelation_front: float = typer.Option(
        0.0,
        "--algo-front-variance",
        min=0.0,
        max=1.0,
        help="Algorithmic surround decorrelation variance for front channels.",
    ),
    algo_decorrelation_rear: float = typer.Option(
        0.0,
        "--algo-rear-variance",
        min=0.0,
        max=1.0,
        help="Algorithmic surround decorrelation variance for rear channels.",
    ),
    algo_decorrelation_top: float = typer.Option(
        0.0,
        "--algo-top-variance",
        min=0.0,
        max=1.0,
        help="Algorithmic surround decorrelation variance for top channels.",
    ),
    tail_limit: float | None = typer.Option(None, "--tail-limit", min=0.0),
    threads: int | None = typer.Option(None, "--threads", min=1),
    device: DeviceName = typer.Option(
        "auto",
        "--device",
        help="Compute device preference: auto, cpu, cuda, or mps (Apple Silicon).",
    ),
    partition_size: int = typer.Option(16_384, "--partition-size", min=256),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional output/render sample rate (Hz). Input is resampled internally if needed.",
    ),
    ir_gen: bool = typer.Option(False, "--ir-gen"),
    ir_gen_mode: IRMode = typer.Option("hybrid", "--ir-gen-mode"),
    ir_gen_length: float = typer.Option(60.0, "--ir-gen-length", min=0.1),
    ir_gen_seed: int = typer.Option(0, "--ir-gen-seed"),
    ir_gen_cache_dir: str = typer.Option(".verbx_cache/irs", "--ir-gen-cache-dir"),
    block_size: int = typer.Option(4096, "--block-size", min=256),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    target_peak_dbfs: float | None = typer.Option(None, "--target-peak-dbfs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    limiter: bool = typer.Option(True, "--limiter/--no-limiter"),
    normalize_stage: NormalizeStage = typer.Option("post", "--normalize-stage"),
    repeat_target_lufs: float | None = typer.Option(None, "--repeat-target-lufs"),
    repeat_target_peak_dbfs: float | None = typer.Option(None, "--repeat-target-peak-dbfs"),
    out_subtype: OutputSubtype = typer.Option(
        "auto",
        "--out-subtype",
        help=(
            "Output file subtype. Internal DSP runs in float64 regardless of container subtype; "
            "use float64/float32/PCM per delivery needs."
        ),
    ),
    output_peak_norm: OutputPeakNorm = typer.Option(
        "none",
        "--output-peak-norm",
        help=(
            "Final peak normalization mode: none, input peak match, explicit target, or full-scale."
        ),
    ),
    output_peak_target_dbfs: float | None = typer.Option(
        None,
        "--output-peak-target-dbfs",
        help="Target dBFS used when --output-peak-norm target is selected.",
    ),
    shimmer: bool = typer.Option(False, "--shimmer"),
    shimmer_semitones: float = typer.Option(12.0, "--shimmer-semitones"),
    shimmer_mix: float = typer.Option(0.25, "--shimmer-mix", min=0.0, max=1.0),
    shimmer_feedback: float = typer.Option(0.35, "--shimmer-feedback", min=0.0, max=1.25),
    shimmer_highcut: float | None = typer.Option(10_000.0, "--shimmer-highcut", min=10.0),
    shimmer_lowcut: float | None = typer.Option(300.0, "--shimmer-lowcut", min=10.0),
    unsafe_self_oscillate: bool = typer.Option(
        False,
        "--unsafe-self-oscillate/--safe-no-self-oscillate",
        help=(
            "UNSAFE: permit feedback-path gains above unity in algorithmic mode for "
            "self-oscillating tails."
        ),
    ),
    unsafe_loop_gain: float = typer.Option(
        1.02,
        "--unsafe-loop-gain",
        min=0.01,
        max=1.25,
        help=(
            "UNSAFE loop-gain scale used with --unsafe-self-oscillate. "
            "Values >1.0 encourage self-oscillation."
        ),
    ),
    duck: bool = typer.Option(False, "--duck"),
    duck_attack: float = typer.Option(20.0, "--duck-attack", min=0.1),
    duck_release: float = typer.Option(350.0, "--duck-release", min=0.1),
    bloom: float = typer.Option(0.0, "--bloom", min=0.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    automation_file: Path | None = typer.Option(
        None,
        "--automation-file",
        exists=True,
        readable=True,
        resolve_path=True,
        help="JSON/CSV automation lanes used for time-varying render control.",
    ),
    automation_mode: AutomationMode = typer.Option(
        "auto",
        "--automation-mode",
        help="Automation evaluation mode: auto, sample, or block.",
    ),
    automation_block_ms: float = typer.Option(
        20.0,
        "--automation-block-ms",
        min=0.1,
        help="Control block size in milliseconds when automation mode is block.",
    ),
    automation_smoothing_ms: float = typer.Option(
        20.0,
        "--automation-smoothing-ms",
        min=0.0,
        help="Default smoothing time (ms) applied to automation lanes.",
    ),
    automation_slew_limit_per_s: float | None = typer.Option(
        None,
        "--automation-slew-limit-per-s",
        min=0.0,
        help=(
            "Optional max control slew as target-range fraction per second; "
            "0/None disables slew guard."
        ),
    ),
    automation_deadband: float = typer.Option(
        0.0,
        "--automation-deadband",
        min=0.0,
        max=1.0,
        help=(
            "Optional control deadband as target-range fraction; "
            "small changes below threshold are suppressed."
        ),
    ),
    automation_clamp: list[str] | None = typer.Option(
        None,
        "--automation-clamp",
        help="Clamp override in target:min:max format (repeatable).",
    ),
    automation_point: list[str] | None = typer.Option(
        None,
        "--automation-point",
        help=(
            "Inline automation control point in target:time_s:value[:interp] format (repeatable)."
        ),
    ),
    automation_trace_out: str | None = typer.Option(
        None,
        "--automation-trace-out",
        help="Optional CSV path for resolved sample-level automation curves.",
    ),
    feature_vector_lane: list[str] | None = typer.Option(
        None,
        "--feature-vector-lane",
        help=(
            "Feature-vector mapping lane (repeatable). "
            "Format: target=<target>,source=<feature>[,weight=<w>][,bias=<b>]"
            "[,curve=<linear|smoothstep|exp|log|tanh|power>][,curve_amount=<a>]"
            "[,hysteresis_up=<u>][,hysteresis_down=<d>][,combine=<replace|add|multiply>]"
            "[,smoothing_ms=<ms>]"
        ),
    ),
    feature_vector_frame_ms: float = typer.Option(
        40.0,
        "--feature-vector-frame-ms",
        min=1.0,
        help="Frame size used for feature-vector extraction (ms).",
    ),
    feature_vector_hop_ms: float = typer.Option(
        20.0,
        "--feature-vector-hop-ms",
        min=1.0,
        help="Hop size used for feature-vector extraction (ms).",
    ),
    feature_guide: Path | None = typer.Option(
        None,
        "--feature-guide",
        exists=True,
        readable=True,
        resolve_path=True,
        help=(
            "Optional external guide audio used for feature-vector extraction "
            "instead of INFILE (Track B external feature-guide ingest)."
        ),
    ),
    feature_guide_policy: FeatureGuidePolicy = typer.Option(
        "align",
        "--feature-guide-policy",
        help=(
            "Mismatch policy for --feature-guide relative to render context: "
            "align (deterministic resample + hold/trim + mixdown) or strict."
        ),
    ),
    feature_vector_trace_out: str | None = typer.Option(
        None,
        "--feature-vector-trace-out",
        help="Optional CSV path for feature+parameter trace exports.",
    ),
    frames_out: str | None = typer.Option(None, "--frames-out"),
    analysis_out: str | None = typer.Option(None, "--analysis-out"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N wild random renders from one input using randomized parameters. "
            "Outputs are written to --lucky-out-dir (or OUTFILE parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky render generation.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress console summary tables while still writing output and analysis artifacts.",
    ),
    verbosity: int = typer.Option(
        1,
        "--verbosity",
        min=0,
        max=2,
        help=(
            "Console detail level: 0=minimal summary, 1=summary + output features (default), "
            "2=also include input feature table."
        ),
    ),
    silent: bool = typer.Option(
        False,
        "--silent",
        help="Disable analysis JSON generation and console output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and print resolved render plan without writing audio.",
    ),
    repro_bundle: bool = typer.Option(
        False,
        "--repro-bundle",
        help="Write a reproducibility/support JSON bundle next to OUTFILE.",
    ),
    repro_bundle_out: Path | None = typer.Option(
        None,
        "--repro-bundle-out",
        resolve_path=True,
        help="Optional explicit path for reproducibility/support JSON bundle.",
    ),
    failure_report_out: Path | None = typer.Option(
        None,
        "--failure-report-out",
        resolve_path=True,
        help="Optional JSON report path populated when render execution fails.",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Render input audio with algorithmic or convolution reverb."""
    resolved_pre_delay_ms = parse_pre_delay_ms(pre_delay, bpm, pre_delay_ms)
    parsed_allpass_delays = _parse_delay_list_ms(
        allpass_delays_ms,
        option_name="--allpass-delays-ms",
    )
    parsed_allpass_gain_values = _parse_gain_list(
        allpass_gain,
        option_name="--allpass-gain",
        min_value=-0.99,
        max_value=0.99,
    )
    parsed_comb_delays = _parse_delay_list_ms(
        comb_delays_ms,
        option_name="--comb-delays-ms",
    )
    parsed_dfm_delays = _parse_delay_list_ms(
        fdn_dfm_delays_ms,
        option_name="--fdn-dfm-delays-ms",
    )

    config = RenderConfig(
        engine=engine,
        rt60=rt60,
        pre_delay_ms=resolved_pre_delay_ms,
        pre_delay_note=pre_delay,
        bpm=bpm,
        damping=damping,
        width=width,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        mod_target=mod_target,
        mod_sources=tuple(mod_source or []),
        mod_routes=tuple(mod_route or []),
        mod_min=mod_min,
        mod_max=mod_max,
        mod_combine=mod_combine,
        mod_smooth_ms=mod_smooth_ms,
        allpass_stages=allpass_stages,
        allpass_gain=float(parsed_allpass_gain_values[0]),
        allpass_gains=parsed_allpass_gain_values if len(parsed_allpass_gain_values) > 1 else (),
        allpass_delays_ms=parsed_allpass_delays,
        comb_delays_ms=parsed_comb_delays,
        fdn_lines=fdn_lines,
        fdn_matrix=(
            "hadamard"
            if fdn_matrix.strip().lower() == "auto"
            else _normalize_fdn_matrix_name(fdn_matrix)
        ),
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
        fdn_tv_seed=2026,
        fdn_dfm_delays_ms=parsed_dfm_delays,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=fdn_rt60_tilt,
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
        fdn_link_filter=_normalize_fdn_link_filter_name(fdn_link_filter),
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
        fdn_graph_topology=_normalize_fdn_graph_topology_name(fdn_graph_topology),
        fdn_graph_degree=fdn_graph_degree,
        fdn_graph_seed=fdn_graph_seed,
        fdn_spatial_coupling_mode=cast(
            FDNSpatialCouplingMode,
            str(fdn_spatial_coupling_mode).strip().lower().replace("-", "_"),
        ),
        fdn_spatial_coupling_strength=float(fdn_spatial_coupling_strength),
        fdn_nonlinearity=cast(
            FDNNonlinearityMode,
            str(fdn_nonlinearity).strip().lower().replace("-", "_"),
        ),
        fdn_nonlinearity_amount=float(fdn_nonlinearity_amount),
        fdn_nonlinearity_drive=float(fdn_nonlinearity_drive),
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
        algo_decorrelation_front=algo_decorrelation_front,
        algo_decorrelation_rear=algo_decorrelation_rear,
        algo_decorrelation_top=algo_decorrelation_top,
        beast_mode=beast_mode,
        wet=wet,
        dry=dry,
        repeat=repeat,
        freeze=freeze,
        start=start,
        end=end,
        block_size=block_size,
        ir=None if ir is None else str(ir),
        ir_blend=tuple(str(path) for path in (ir_blend or [])),
        ir_blend_mix=tuple(float(value) for value in (ir_blend_mix or [])),
        ir_blend_mode=normalize_ir_morph_mode_name(ir_blend_mode),
        ir_blend_early_ms=float(ir_blend_early_ms),
        ir_blend_early_alpha=(
            None if ir_blend_early_alpha is None else float(ir_blend_early_alpha)
        ),
        ir_blend_late_alpha=(None if ir_blend_late_alpha is None else float(ir_blend_late_alpha)),
        ir_blend_align_decay=bool(ir_blend_align_decay),
        ir_blend_phase_coherence=float(ir_blend_phase_coherence),
        ir_blend_spectral_smooth_bins=int(ir_blend_spectral_smooth_bins),
        ir_blend_mismatch_policy=cast(
            IRMorphMismatchPolicy,
            normalize_ir_morph_mismatch_policy_name(ir_blend_mismatch_policy),
        ),
        ir_blend_cache_dir=ir_blend_cache_dir,
        input_layout=input_layout,
        output_layout=output_layout,
        self_convolve=self_convolve,
        ir_normalize=ir_normalize,
        ir_matrix_layout=ir_matrix_layout,
        ir_route_map=_normalize_ir_route_map_name(ir_route_map),
        conv_route_start=conv_route_start,
        conv_route_end=conv_route_end,
        conv_route_curve=_normalize_conv_route_curve_name(conv_route_curve),
        ambi_order=int(ambi_order),
        ambi_normalization=cast(AmbiNormalization, str(ambi_normalization).strip().lower()),
        channel_order=cast(AmbiChannelOrder, str(channel_order).strip().lower()),
        ambi_encode_from=cast(AmbiEncodeFrom, str(ambi_encode_from).strip().lower()),
        ambi_decode_to=cast(AmbiDecodeTo, str(ambi_decode_to).strip().lower()),
        ambi_rotate_yaw_deg=float(ambi_rotate_yaw_deg),
        tail_limit=tail_limit,
        threads=threads,
        device=device,
        partition_size=partition_size,
        target_sr=target_sr,
        ir_gen=ir_gen,
        ir_gen_mode=ir_gen_mode,
        ir_gen_length=ir_gen_length,
        ir_gen_seed=ir_gen_seed,
        ir_gen_cache_dir=ir_gen_cache_dir,
        target_lufs=target_lufs,
        target_peak_dbfs=target_peak_dbfs,
        use_true_peak=true_peak,
        limiter=limiter,
        normalize_stage=normalize_stage,
        repeat_target_lufs=repeat_target_lufs,
        repeat_target_peak_dbfs=repeat_target_peak_dbfs,
        output_subtype=out_subtype,
        output_peak_norm=output_peak_norm,
        output_peak_target_dbfs=output_peak_target_dbfs,
        shimmer=shimmer,
        shimmer_semitones=shimmer_semitones,
        shimmer_mix=shimmer_mix,
        shimmer_feedback=shimmer_feedback,
        shimmer_highcut=shimmer_highcut,
        shimmer_lowcut=shimmer_lowcut,
        unsafe_self_oscillate=unsafe_self_oscillate,
        unsafe_loop_gain=unsafe_loop_gain,
        duck=duck,
        duck_attack=duck_attack,
        duck_release=duck_release,
        bloom=bloom,
        lowcut=lowcut,
        highcut=highcut,
        tilt=tilt,
        automation_file=None if automation_file is None else str(automation_file),
        automation_mode=cast(AutomationMode, str(automation_mode).strip().lower()),
        automation_block_ms=float(automation_block_ms),
        automation_smoothing_ms=float(automation_smoothing_ms),
        automation_slew_limit_per_s=(
            None
            if automation_slew_limit_per_s is None
            else float(automation_slew_limit_per_s)
        ),
        automation_deadband=float(automation_deadband),
        automation_clamp=tuple(automation_clamp or ()),
        automation_points=tuple(automation_point or ()),
        automation_trace_out=automation_trace_out,
        feature_vector_lanes=tuple(feature_vector_lane or ()),
        feature_vector_frame_ms=float(feature_vector_frame_ms),
        feature_vector_hop_ms=float(feature_vector_hop_ms),
        feature_guide=None if feature_guide is None else str(feature_guide),
        feature_guide_policy=cast(
            FeatureGuidePolicy,
            str(feature_guide_policy).strip().lower(),
        ),
        feature_vector_trace_out=feature_vector_trace_out,
        frames_out=frames_out,
        analysis_out=analysis_out,
        silent=silent,
        progress=progress,
    )

    preset_summary: dict[str, Any] | None = None
    if preset is not None:
        try:
            preset_summary = _apply_render_preset(
                ctx=ctx,
                config=config,
                preset_name=preset,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    repro_bundle_path = _resolve_repro_bundle_path(
        outfile=outfile,
        repro_bundle=repro_bundle,
        repro_bundle_out=repro_bundle_out,
    )
    _validate_repro_bundle_path(
        infile=infile,
        outfile=outfile,
        analysis_out=config.analysis_out,
        repro_bundle_path=repro_bundle_path,
    )
    _validate_failure_report_path(
        infile=infile,
        outfile=outfile,
        analysis_out=config.analysis_out,
        repro_bundle_path=repro_bundle_path,
        failure_report_out=failure_report_out,
    )
    _validate_render_call(infile, outfile, config)
    _validate_lucky_call(
        config,
        lucky,
        lucky_out_dir,
        repro_bundle_path=repro_bundle_path,
        failure_report_out=failure_report_out,
    )
    configure_logging(verbose=not config.silent)

    if dry_run:
        _print_render_dry_run_plan(
            infile=infile,
            outfile=outfile,
            config=config,
            lucky=lucky,
            lucky_out_dir=lucky_out_dir,
            preset_summary=preset_summary,
            repro_bundle_path=repro_bundle_path,
        )
        return

    if lucky is not None:
        try:
            info = sf.info(str(infile))
            duration_seconds = (
                float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        out_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = _resolve_lucky_seed(lucky_seed)

        lucky_rows: list[dict[str, str]] = []
        with _BatchStatusBar(
            total=lucky,
            label="Lucky render batch",
            enabled=bool(config.progress and not config.silent),
        ) as status:
            for idx in range(lucky):
                rng = np.random.default_rng(seed + idx)
                lucky_config = _build_lucky_config(
                    base=config,
                    rng=rng,
                    input_duration_seconds=duration_seconds,
                )
                lucky_config.progress = config.progress
                lucky_out = out_dir / f"{outfile.stem}.lucky_{idx + 1:03d}{outfile.suffix}"

                try:
                    report = run_render_pipeline(
                        infile=infile,
                        outfile=lucky_out,
                        config=lucky_config,
                    )
                except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
                    raise typer.BadParameter(str(exc)) from exc

                lucky_rows.append(
                    {
                        "index": str(idx + 1),
                        "outfile": str(lucky_out),
                        "engine": str(report.get("engine", "unknown")),
                        "rt60": f"{float(lucky_config.rt60):.2f}",
                        "repeat": str(int(lucky_config.repeat)),
                        "beast": str(int(lucky_config.beast_mode)),
                    }
                )
                status.advance(detail=lucky_out.name)

        if not config.silent and not quiet and verbosity > 0:
            summary = Table(title=f"Lucky Render Batch ({lucky} outputs)")
            summary.add_column("#", style="cyan", justify="right")
            summary.add_column("outfile", style="white")
            summary.add_column("engine", style="green")
            summary.add_column("rt60", justify="right")
            summary.add_column("repeat", justify="right")
            summary.add_column("beast", justify="right")
            for row in lucky_rows:
                summary.add_row(
                    row["index"],
                    row["outfile"],
                    row["engine"],
                    row["rt60"],
                    row["repeat"],
                    row["beast"],
                )
            console.print(summary)
        return

    if config.shimmer:
        if importlib.util.find_spec("librosa") is None:
            console.print(
                "[yellow]Warning:[/yellow] --shimmer is enabled but librosa is not installed. "
                "Pitch-shift quality will be significantly lower (linear interpolation fallback). "
                "Install with: pip install librosa"
            )

    try:
        with _processing_status(
            "Render audio",
            enabled=bool(config.progress and not config.silent),
        ):
            report = run_render_pipeline(infile=infile, outfile=outfile, config=config)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        if failure_report_out is not None:
            try:
                payload = _build_render_failure_report(
                    infile=infile,
                    outfile=outfile,
                    config=config,
                    preset_name=(
                        None
                        if preset_summary is None
                        else str(preset_summary.get("name", "")).strip() or None
                    ),
                    error=exc,
                )
                _write_json_atomic(failure_report_out.resolve(), payload)
                msg = f"{exc} (failure report: {failure_report_out.resolve()})"
                raise typer.BadParameter(msg) from exc
            except (OSError, RuntimeError, ValueError) as write_exc:
                msg = f"{exc} (also failed to write failure report: {write_exc})"
                raise typer.BadParameter(msg) from exc
        raise typer.BadParameter(str(exc)) from exc

    if repro_bundle_path is not None:
        try:
            repro_payload = _build_render_repro_bundle(
                infile=infile,
                outfile=outfile,
                report=report,
                config=config,
                preset_name=(
                    None
                    if preset_summary is None
                    else str(preset_summary.get("name", "")).strip() or None
                ),
            )
            _write_json_atomic(repro_bundle_path, repro_payload)
        except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
            raise typer.BadParameter(f"Failed to write repro bundle: {exc}") from exc
        report["repro_bundle_path"] = str(repro_bundle_path.resolve())

    if config.silent or quiet:
        return

    _print_render_summary(
        report,
        verbosity=verbosity,
        preset_name=(
            None
            if preset_summary is None
            else str(preset_summary.get("name", "")).strip() or None
        ),
    )


@app.command()
def analyze(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    lufs: bool = typer.Option(False, "--lufs", help="Include LUFS/true-peak/LRA metrics."),
    edr: bool = typer.Option(
        False,
        "--edr",
        help="Include EDR (Energy Decay Relief) summary metrics.",
    ),
    frames_out: Path | None = typer.Option(None, "--frames-out", resolve_path=True),
    ambi_order: int = typer.Option(
        0,
        "--ambi-order",
        min=0,
        max=7,
        help="Enable Ambisonics spatial metrics for the given order.",
    ),
    ambi_normalization: AmbiNormalization = typer.Option(
        "auto",
        "--ambi-normalization",
        help="Ambisonics normalization convention for analysis mode.",
    ),
    channel_order: AmbiChannelOrder = typer.Option(
        "auto",
        "--channel-order",
        help="Ambisonics channel order convention for analysis mode.",
    ),
) -> None:
    """Analyze an audio file and print a summary table."""
    _validate_analyze_call(infile, json_out, frames_out)
    try:
        with _processing_status("Analyze audio"):
            validate_audio_path(str(infile))
            audio, sr = read_audio(str(infile))
            analyzer = AudioAnalyzer()
            metrics = analyzer.analyze(
                audio,
                sr,
                include_loudness=lufs,
                include_edr=edr,
                ambi_order=int(ambi_order) if int(ambi_order) > 0 else None,
                ambi_normalization=str(ambi_normalization).strip().lower(),
                ambi_channel_order=str(channel_order).strip().lower(),
            )
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Analysis: {infile.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        table.add_row(key, f"{metrics[key]:.6f}")
    console.print(table)

    if json_out is not None:
        payload = {"sample_rate": sr, "channels": audio.shape[1], "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if frames_out is not None:
        write_framewise_csv(frames_out, audio, sr)


@app.command()
def suggest(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    """Suggest practical render defaults from input analysis."""
    try:
        with _processing_status("Analyze for suggestions"):
            validate_audio_path(str(infile))
            audio, sr = read_audio(str(infile))
            analyzer = AudioAnalyzer()
            metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    duration = metrics["duration"]
    dynamic = metrics["dynamic_range"]
    flatness = metrics["spectral_flatness"]

    suggested_rt60 = float(np.clip(duration * 1.8, 25.0, 120.0))
    suggested_wet = float(np.clip(0.55 + (dynamic / 60.0), 0.4, 0.95))
    suggested_dry = float(np.clip(1.0 - (suggested_wet * 0.85), 0.05, 0.85))
    suggested_engine = "conv" if flatness < 0.12 else "algo"

    table = Table(title=f"Suggested Parameters: {infile.name}")
    table.add_column("Parameter", style="green")
    table.add_column("Suggested Value", style="white")
    table.add_row("engine", suggested_engine)
    table.add_row("rt60", f"{suggested_rt60:.2f}")
    table.add_row("wet", f"{suggested_wet:.3f}")
    table.add_row("dry", f"{suggested_dry:.3f}")
    table.add_row("repeat", "2" if duration < 15.0 else "1")
    table.add_row("target-lufs", "-18.0")
    table.add_row("target-peak-dbfs", "-1.0")
    table.add_row("normalize-stage", "post")
    table.add_row("shimmer", "off")
    table.add_row("duck", "off")
    console.print(table)


@app.command(name="presets")
def list_presets(
    show: str | None = typer.Option(
        None,
        "--show",
        help="Show resolved values for one preset.",
    ),
) -> None:
    """Print available presets or one preset payload."""
    if show is not None:
        try:
            resolved_name, payload = resolve_preset(show)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        table = Table(title=f"Preset: {resolved_name}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        for key in sorted(payload.keys()):
            table.add_row(key, str(payload[key]))
        console.print(table)
        return

    names = preset_names()
    table = Table(title="Available Presets")
    table.add_column("Preset", style="green")
    for name in names:
        table.add_row(name)
    console.print(table)


@ir_app.command("gen")
def ir_gen(
    out_ir: Path = typer.Argument(..., resolve_path=True),
    out_format: IRFileFormat = typer.Option("auto", "--format"),
    mode: IRMode = typer.Option("hybrid", "--mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    sr: int = typer.Option(48_000, "--sr", min=8_000),
    channels: int = typer.Option(2, "--channels", min=1),
    seed: int = typer.Option(0, "--seed"),
    rt60: float | None = typer.Option(None, "--rt60", min=RT60_MIN_SECONDS, max=RT60_MAX_SECONDS),
    rt60_low: float | None = typer.Option(
        None,
        "--rt60-low",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
    ),
    rt60_high: float | None = typer.Option(
        None,
        "--rt60-high",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
    ),
    damping: float = typer.Option(0.4, "--damping", min=0.0, max=1.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    normalize: Literal["none", "peak", "rms"] = typer.Option("peak", "--normalize"),
    peak_dbfs: float = typer.Option(-1.0, "--peak-dbfs"),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    er_count: int = typer.Option(24, "--er-count", min=0),
    er_max_delay_ms: float = typer.Option(90.0, "--er-max-delay-ms", min=1.0),
    er_decay_shape: str = typer.Option("exp", "--er-decay-shape"),
    er_stereo_width: float = typer.Option(1.0, "--er-stereo-width", min=0.0, max=2.0),
    er_room: float = typer.Option(1.0, "--er-room", min=0.1),
    diffusion: float = typer.Option(0.5, "--diffusion", min=0.0, max=1.0),
    mod_depth_ms: float = typer.Option(1.5, "--mod-depth-ms", min=0.0),
    mod_rate_hz: float = typer.Option(0.12, "--mod-rate-hz", min=0.0),
    density: float = typer.Option(1.0, "--density", min=0.01),
    tuning: str = typer.Option("A4=440", "--tuning"),
    modal_count: int = typer.Option(48, "--modal-count", min=1),
    modal_q_min: float = typer.Option(5.0, "--modal-q-min", min=0.5),
    modal_q_max: float = typer.Option(60.0, "--modal-q-max", min=0.5),
    modal_spread_cents: float = typer.Option(5.0, "--modal-spread-cents", min=0.0),
    modal_low_hz: float = typer.Option(80.0, "--modal-low-hz", min=20.0),
    modal_high_hz: float = typer.Option(12_000.0, "--modal-high-hz", min=50.0),
    fdn_lines: int = typer.Option(8, "--fdn-lines", min=1),
    fdn_matrix: str = typer.Option(
        "hadamard",
        "--fdn-matrix",
        help=(
            "FDN matrix topology: hadamard, householder, random_orthogonal, "
            "circulant, elliptic, tv_unitary, graph, or sdn_hybrid."
        ),
    ),
    fdn_tv_rate_hz: float = typer.Option(
        0.0,
        "--fdn-tv-rate-hz",
        min=0.0,
        help="Block-rate update speed for --fdn-matrix tv_unitary (Hz).",
    ),
    fdn_tv_depth: float = typer.Option(
        0.0,
        "--fdn-tv-depth",
        min=0.0,
        max=1.0,
        help="Blend depth for --fdn-matrix tv_unitary (0..1).",
    ),
    fdn_dfm_delays_ms: str | None = typer.Option(
        None,
        "--fdn-dfm-delays-ms",
        help=(
            "Optional delay-feedback-matrix delays in milliseconds. "
            "Provide one value for broadcast or one per FDN line."
        ),
    ),
    fdn_sparse: bool = typer.Option(
        False,
        "--fdn-sparse/--no-fdn-sparse",
        help="Enable sparse high-order FDN pair-mixing mode.",
    ),
    fdn_sparse_degree: int = typer.Option(
        2,
        "--fdn-sparse-degree",
        min=1,
        max=16,
        help="Number of sparse pair-mixing stages used when --fdn-sparse is enabled.",
    ),
    fdn_cascade: bool = typer.Option(
        False,
        "--fdn-cascade/--no-fdn-cascade",
        help="Enable nested/cascaded FDN mode (small fast network into late network).",
    ),
    fdn_cascade_mix: float = typer.Option(
        0.35,
        "--fdn-cascade-mix",
        min=0.0,
        max=1.0,
        help="Injection amount from nested FDN into the main late-field network (0..1).",
    ),
    fdn_cascade_delay_scale: float = typer.Option(
        0.5,
        "--fdn-cascade-delay-scale",
        min=0.2,
        max=1.0,
        help="Delay scaling for nested FDN relative to primary FDN delays (0.2..1.0).",
    ),
    fdn_cascade_rt60_ratio: float = typer.Option(
        0.55,
        "--fdn-cascade-rt60-ratio",
        min=0.1,
        max=1.0,
        help="RT60 ratio for nested FDN relative to --rt60 (0.1..1.0).",
    ),
    fdn_rt60_low: float | None = typer.Option(
        None,
        "--fdn-rt60-low",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="Low-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_mid: float | None = typer.Option(
        None,
        "--fdn-rt60-mid",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="Mid-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_high: float | None = typer.Option(
        None,
        "--fdn-rt60-high",
        min=RT60_MIN_SECONDS,
        max=RT60_MAX_SECONDS,
        help="High-band RT60 target for multiband FDN decay shaping (seconds).",
    ),
    fdn_rt60_tilt: float = typer.Option(
        0.0,
        "--fdn-rt60-tilt",
        min=-1.0,
        max=1.0,
        help=(
            "Jot-style low/high RT skew around mid band (-1..1). "
            "Positive extends low-band decay and shortens highs."
        ),
    ),
    fdn_tonal_correction_strength: float = typer.Option(
        0.0,
        "--fdn-tonal-correction-strength",
        min=0.0,
        max=1.0,
        help=(
            "Track C tonal correction strength for multiband/tilted FDN response (0..1). "
            "Higher values apply stronger decay-color equalization."
        ),
    ),
    fdn_xover_low_hz: float = typer.Option(
        250.0,
        "--fdn-xover-low-hz",
        min=20.0,
        help="Low/mid crossover frequency used by multiband FDN decay shaping.",
    ),
    fdn_xover_high_hz: float = typer.Option(
        4_000.0,
        "--fdn-xover-high-hz",
        min=100.0,
        help="Mid/high crossover frequency used by multiband FDN decay shaping.",
    ),
    fdn_link_filter: str = typer.Option(
        "none",
        "--fdn-link-filter",
        help=("Feedback-link filter mode inside the FDN matrix path: none, lowpass, or highpass."),
    ),
    fdn_link_filter_hz: float = typer.Option(
        2_500.0,
        "--fdn-link-filter-hz",
        min=20.0,
        help="Cutoff frequency used by --fdn-link-filter (Hz).",
    ),
    fdn_link_filter_mix: float = typer.Option(
        1.0,
        "--fdn-link-filter-mix",
        min=0.0,
        max=1.0,
        help="Wet mix of feedback-link filter processing (0..1).",
    ),
    fdn_graph_topology: str = typer.Option(
        "ring",
        "--fdn-graph-topology",
        help="Graph topology for --fdn-matrix graph: ring, path, star, or random.",
    ),
    fdn_graph_degree: int = typer.Option(
        2,
        "--fdn-graph-degree",
        min=1,
        max=32,
        help="Graph neighborhood/connectivity degree for --fdn-matrix graph.",
    ),
    fdn_graph_seed: int = typer.Option(
        2026,
        "--fdn-graph-seed",
        help="Deterministic seed used to build graph-structured FDN pairings.",
    ),
    fdn_spatial_coupling_mode: FDNSpatialCouplingMode = typer.Option(
        "none",
        "--fdn-spatial-coupling-mode",
        help=(
            "Directional wet-bus coupling mode: none, adjacent, front_rear, bed_top, all_to_all."
        ),
    ),
    fdn_spatial_coupling_strength: float = typer.Option(
        0.0,
        "--fdn-spatial-coupling-strength",
        min=0.0,
        max=1.0,
        help="Wet-bus directional coupling amount (0..1).",
    ),
    fdn_nonlinearity: FDNNonlinearityMode = typer.Option(
        "none",
        "--fdn-nonlinearity",
        help="Optional in-loop nonlinearity: none, tanh, or softclip.",
    ),
    fdn_nonlinearity_amount: float = typer.Option(
        0.0,
        "--fdn-nonlinearity-amount",
        min=0.0,
        max=1.0,
        help="Blend amount for in-loop nonlinearity shaping (0..1).",
    ),
    fdn_nonlinearity_drive: float = typer.Option(
        1.0,
        "--fdn-nonlinearity-drive",
        min=0.1,
        max=8.0,
        help="Drive multiplier for in-loop nonlinearity shaping.",
    ),
    room_size_macro: float = typer.Option(
        0.0,
        "--room-size-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual room-size macro (-1..1) mapped to decay-time and spacing behavior.",
    ),
    clarity_macro: float = typer.Option(
        0.0,
        "--clarity-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual clarity macro (-1..1) mapped to decay, damping, and wet balance.",
    ),
    warmth_macro: float = typer.Option(
        0.0,
        "--warmth-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual warmth macro (-1..1) mapped to damping and spectral decay tilt.",
    ),
    envelopment_macro: float = typer.Option(
        0.0,
        "--envelopment-macro",
        min=-1.0,
        max=1.0,
        help="Perceptual envelopment macro (-1..1) mapped to width/decorrelation emphasis.",
    ),
    fdn_stereo_inject: float = typer.Option(1.0, "--fdn-stereo-inject", min=0.0, max=1.0),
    f0: str | None = typer.Option(None, "--f0", help="e.g. 64, 64Hz, or 64 Hz"),
    analyze_input: Path | None = typer.Option(
        None,
        "--analyze-input",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Input audio to estimate fundamentals/harmonics for IR tuning",
    ),
    harmonic_align_strength: float = typer.Option(
        0.75, "--harmonic-align-strength", min=0.0, max=1.0
    ),
    resonator: bool = typer.Option(
        False,
        "--resonator/--no-resonator",
        help="Enable Modalys-inspired physical modal-bank late-tail coloration.",
    ),
    resonator_mix: float = typer.Option(0.35, "--resonator-mix", min=0.0, max=1.0),
    resonator_modes: int = typer.Option(32, "--resonator-modes", min=1),
    resonator_q_min: float = typer.Option(8.0, "--resonator-q-min", min=0.5),
    resonator_q_max: float = typer.Option(90.0, "--resonator-q-max", min=0.5),
    resonator_low_hz: float = typer.Option(50.0, "--resonator-low-hz", min=20.0),
    resonator_high_hz: float = typer.Option(9000.0, "--resonator-high-hz", min=30.0),
    resonator_late_start_ms: float = typer.Option(80.0, "--resonator-late-start-ms", min=0.0),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N randomized IR files from one base setup. "
            "Outputs are written to --lucky-out-dir (or OUT_IR parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky IR generation.",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Generate an IR file with deterministic caching."""
    _validate_ir_gen_call(
        out_ir=out_ir,
        out_format=out_format,
        rt60=rt60,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
        modal_q_min=modal_q_min,
        modal_q_max=modal_q_max,
        modal_low_hz=modal_low_hz,
        modal_high_hz=modal_high_hz,
        resonator_q_min=resonator_q_min,
        resonator_q_max=resonator_q_max,
        resonator_low_hz=resonator_low_hz,
        resonator_high_hz=resonator_high_hz,
        fdn_lines=fdn_lines,
        fdn_matrix=fdn_matrix,
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=fdn_rt60_tilt,
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
        fdn_link_filter=fdn_link_filter,
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
        fdn_graph_topology=fdn_graph_topology,
        fdn_graph_degree=fdn_graph_degree,
        fdn_spatial_coupling_mode=fdn_spatial_coupling_mode,
        fdn_spatial_coupling_strength=fdn_spatial_coupling_strength,
        fdn_nonlinearity=fdn_nonlinearity,
        fdn_nonlinearity_amount=fdn_nonlinearity_amount,
        fdn_nonlinearity_drive=fdn_nonlinearity_drive,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
    )
    _validate_generic_lucky_call(lucky, lucky_out_dir)

    parsed_fdn_dfm_delays = _parse_delay_list_ms(
        fdn_dfm_delays_ms,
        option_name="--fdn-dfm-delays-ms",
    )
    if len(parsed_fdn_dfm_delays) not in {0, 1, fdn_lines}:
        msg = f"--fdn-dfm-delays-ms must include either 1 value or exactly {fdn_lines} values."
        raise typer.BadParameter(msg)

    f0_hz: float | None = None
    harmonic_targets_hz: tuple[float, ...] = ()

    if f0 is not None:
        try:
            f0_hz = parse_frequency_hz(f0)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if analyze_input is not None:
        est_f0, harmonics = analyze_audio_for_tuning(analyze_input)
        if f0_hz is None:
            f0_hz = est_f0
        harmonic_targets_hz = tuple(harmonics)

    cfg = IRGenConfig(
        mode=mode,
        length=length,
        sr=sr,
        channels=channels,
        seed=seed,
        rt60=rt60,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
        damping=damping,
        lowcut=lowcut,
        highcut=highcut,
        tilt=tilt,
        normalize=normalize,
        peak_dbfs=peak_dbfs,
        target_lufs=target_lufs,
        true_peak=true_peak,
        er_count=er_count,
        er_max_delay_ms=er_max_delay_ms,
        er_decay_shape=er_decay_shape,
        er_stereo_width=er_stereo_width,
        er_room=er_room,
        diffusion=diffusion,
        mod_depth_ms=mod_depth_ms,
        mod_rate_hz=mod_rate_hz,
        density=density,
        tuning=tuning,
        modal_count=modal_count,
        modal_q_min=modal_q_min,
        modal_q_max=modal_q_max,
        modal_spread_cents=modal_spread_cents,
        modal_low_hz=modal_low_hz,
        modal_high_hz=modal_high_hz,
        fdn_lines=fdn_lines,
        fdn_matrix=_normalize_fdn_matrix_name(fdn_matrix),
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
        fdn_tv_seed=seed,
        fdn_dfm_delays_ms=parsed_fdn_dfm_delays,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_rt60_tilt=fdn_rt60_tilt,
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
        fdn_link_filter=_normalize_fdn_link_filter_name(fdn_link_filter),
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
        fdn_graph_topology=_normalize_fdn_graph_topology_name(fdn_graph_topology),
        fdn_graph_degree=fdn_graph_degree,
        fdn_graph_seed=fdn_graph_seed,
        fdn_spatial_coupling_mode=str(fdn_spatial_coupling_mode).strip().lower().replace("-", "_"),
        fdn_spatial_coupling_strength=float(fdn_spatial_coupling_strength),
        fdn_nonlinearity=str(fdn_nonlinearity).strip().lower().replace("-", "_"),
        fdn_nonlinearity_amount=float(fdn_nonlinearity_amount),
        fdn_nonlinearity_drive=float(fdn_nonlinearity_drive),
        fdn_stereo_inject=fdn_stereo_inject,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonic_targets_hz,
        harmonic_align_strength=harmonic_align_strength,
        resonator=resonator,
        resonator_mix=resonator_mix,
        resonator_modes=resonator_modes,
        resonator_q_min=resonator_q_min,
        resonator_q_max=resonator_q_max,
        resonator_low_hz=resonator_low_hz,
        resonator_high_hz=resonator_high_hz,
        resonator_late_start_ms=resonator_late_start_ms,
    )

    resolved_out_ir = _resolve_ir_output_path(out_ir, out_format)

    if lucky is not None:
        out_dir = resolved_out_ir.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_value = _resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        with _BatchStatusBar(
            total=lucky,
            label="Lucky IR generation",
            enabled=not silent,
        ) as status:
            for idx in range(lucky):
                rng = np.random.default_rng(seed_value + idx)
                lucky_cfg = _build_lucky_ir_gen_config(cfg, rng=rng)
                lucky_out = (
                    out_dir / f"{resolved_out_ir.stem}.lucky_{idx + 1:03d}{resolved_out_ir.suffix}"
                )
                try:
                    audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
                        lucky_cfg,
                        cache_dir=Path(cache_dir),
                    )
                    write_ir_artifacts(lucky_out, audio, out_sr, meta, silent=silent)
                except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
                    raise typer.BadParameter(str(exc)) from exc

                rows.append(
                    {
                        "index": str(idx + 1),
                        "out_ir": str(lucky_out),
                        "mode": lucky_cfg.mode,
                        "length_s": f"{float(lucky_cfg.length):.2f}",
                        "rt60": (
                            f"{float(lucky_cfg.rt60):.2f}"
                            if lucky_cfg.rt60 is not None
                            else (
                                f"{float(lucky_cfg.rt60_low or 0.0):.2f}-"
                                f"{float(lucky_cfg.rt60_high or 0.0):.2f}"
                            )
                        ),
                        "cache_hit": str(cache_hit),
                        "cache_path": str(cache_path),
                    }
                )
                status.advance(detail=f"seed={seed_value + idx}")

        if not silent:
            table = Table(title=f"Lucky IR Generation Batch ({lucky} outputs)")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("out_ir", style="white")
            table.add_column("mode", style="green")
            table.add_column("length_s", justify="right")
            table.add_column("rt60", justify="right")
            table.add_column("cache_hit", justify="right")
            for row in rows:
                table.add_row(
                    row["index"],
                    row["out_ir"],
                    row["mode"],
                    row["length_s"],
                    row["rt60"],
                    row["cache_hit"],
                )
            console.print(table)
        return

    try:
        with _processing_status("Generate IR", enabled=not silent):
            audio, out_sr, meta, cache_path, cache_hit = generate_or_load_cached_ir(
                cfg,
                cache_dir=Path(cache_dir),
            )
            write_ir_artifacts(resolved_out_ir, audio, out_sr, meta, silent=silent)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title="IR Generation")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", mode)
    table.add_row("out_ir", str(resolved_out_ir))
    table.add_row("format", out_format)
    table.add_row("cache_path", str(cache_path))
    table.add_row("cache_hit", str(cache_hit))
    table.add_row("duration_s", f"{audio.shape[0] / out_sr:.2f}")
    table.add_row("channels", str(audio.shape[1]))
    if f0_hz is not None:
        table.add_row("f0_hz", f"{f0_hz:.3f}")
    if analyze_input is not None:
        table.add_row("analyze_input", str(analyze_input))
        table.add_row("harmonics_detected", str(len(harmonic_targets_hz)))
    table.add_row("resonator", str(resonator))
    if resonator:
        table.add_row("resonator_mix", f"{resonator_mix:.3f}")
        table.add_row("resonator_modes", str(resonator_modes))
        table.add_row(
            "resonator_band_hz",
            f"{resonator_low_hz:.1f}-{resonator_high_hz:.1f}",
        )
        table.add_row("resonator_q", f"{resonator_q_min:.2f}-{resonator_q_max:.2f}")
        table.add_row("resonator_late_start_ms", f"{resonator_late_start_ms:.2f}")
    console.print(table)


@ir_app.command("analyze")
def ir_analyze(
    ir_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
) -> None:
    """Analyze an impulse response."""
    _validate_ir_analyze_call(ir_file, json_out)
    try:
        with _processing_status("Analyze IR"):
            audio, sr = sf.read(str(ir_file), always_2d=True, dtype="float64")
            metrics = analyze_ir(np.asarray(audio, dtype=np.float64), int(sr))
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"IR Analysis: {ir_file.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for key in [
        "duration_seconds",
        "peak_dbfs",
        "rms_dbfs",
        "rt60_estimate_seconds",
        "early_late_ratio_db",
        "stereo_coherence",
    ]:
        value = metrics.get(key)
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
    decay_points = metrics.get("decay_curve_db", [])
    point_count = len(decay_points) if isinstance(decay_points, list) else 0
    table.add_row("decay_curve_points", str(point_count))
    console.print(table)

    if json_out is not None:
        payload = {"file": str(ir_file), "sample_rate": int(sr), "metrics": metrics}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@ir_app.command("sofa-info")
def ir_sofa_info(
    sofa_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    json_out: Path | None = typer.Option(None, "--json-out", resolve_path=True),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Inspect SOFA metadata and dimensions."""
    try:
        with _processing_status("Read SOFA metadata", enabled=not silent):
            info = read_sofa_info(sofa_file)
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = {
        "path": info.path,
        "conventions": info.conventions,
        "version": info.version,
        "data_ir_shape": list(info.data_ir_shape),
        "sample_rate_hz": int(info.sample_rate_hz),
        "source_position_shape": (
            None if info.source_position_shape is None else list(info.source_position_shape)
        ),
        "listener_position_shape": (
            None if info.listener_position_shape is None else list(info.listener_position_shape)
        ),
        "receiver_position_shape": (
            None if info.receiver_position_shape is None else list(info.receiver_position_shape)
        ),
        "emitter_position_shape": (
            None if info.emitter_position_shape is None else list(info.emitter_position_shape)
        ),
        "dimension_labels": list(info.dimension_labels),
    }

    if not silent:
        table = Table(title=f"SOFA Info: {Path(info.path).name}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("conventions", info.conventions)
        table.add_row("version", info.version)
        table.add_row("sample_rate_hz", str(int(info.sample_rate_hz)))
        table.add_row("data_ir_shape", str(info.data_ir_shape))
        if len(info.dimension_labels) > 0:
            table.add_row("dimension_labels", ", ".join(info.dimension_labels))
        if info.source_position_shape is not None:
            table.add_row("source_position_shape", str(info.source_position_shape))
        if info.listener_position_shape is not None:
            table.add_row("listener_position_shape", str(info.listener_position_shape))
        if info.receiver_position_shape is not None:
            table.add_row("receiver_position_shape", str(info.receiver_position_shape))
        if info.emitter_position_shape is not None:
            table.add_row("emitter_position_shape", str(info.emitter_position_shape))
        console.print(table)

    if json_out is not None:
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@ir_app.command("sofa-extract")
def ir_sofa_extract(
    sofa_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    measurement_index: int = typer.Option(
        0,
        "--measurement-index",
        min=0,
        help="Measurement index for SOFA Data/IR extraction (first axis in strict modes).",
    ),
    emitter_index: int = typer.Option(
        0,
        "--emitter-index",
        min=0,
        help="Emitter index for rank-4 Data/IR extraction (strict mode).",
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional output sample rate target for extracted IR.",
    ),
    normalize: Literal["none", "peak", "rms"] = typer.Option(
        "peak",
        "--normalize",
        help="Normalization for extracted IR matrix.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict/--best-effort",
        help="Strict expects Data/IR rank 3 (M,R,N) or 4 (M,R,E,N).",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Extract SOFA FIR data to a WAV matrix for convolution workflows."""
    _validate_output_audio_path(out_ir, "auto")
    try:
        with _processing_status("Extract SOFA IR", enabled=not silent):
            audio, sr, meta = extract_sofa_ir(
                sofa_file,
                measurement_index=int(measurement_index),
                emitter_index=int(emitter_index),
                target_sr=None if target_sr is None else int(target_sr),
                normalize=str(normalize),
                strict=bool(strict),
            )
            write_ir_artifacts(out_ir, audio, sr, meta, silent=silent)
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title=f"SOFA Extract: {out_ir.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("source", str(sofa_file))
    table.add_row("out_ir", str(out_ir))
    table.add_row("sample_rate_hz", str(int(sr)))
    table.add_row("shape", str(tuple(int(v) for v in audio.shape)))
    table.add_row("normalize", str(normalize))
    table.add_row("strict", str(bool(strict)))
    sample_rate_action = str(meta.get("sample_rate_action", "none"))
    table.add_row("sample_rate_action", sample_rate_action)
    console.print(table)


@ir_app.command("process")
def ir_process(
    in_ir: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    damping: float = typer.Option(0.4, "--damping", min=0.0, max=1.0),
    lowcut: float | None = typer.Option(None, "--lowcut", min=10.0),
    highcut: float | None = typer.Option(None, "--highcut", min=10.0),
    tilt: float = typer.Option(0.0, "--tilt"),
    normalize: Literal["none", "peak", "rms"] = typer.Option("peak", "--normalize"),
    peak_dbfs: float = typer.Option(-1.0, "--peak-dbfs"),
    target_lufs: float | None = typer.Option(None, "--target-lufs"),
    true_peak: bool = typer.Option(True, "--true-peak/--sample-peak"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "Generate N randomized processed IR files from one input IR. "
            "Outputs are written to --lucky-out-dir (or OUT_IR parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky IR processing.",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Process an existing IR through shaping/targeting chain."""
    _validate_ir_process_call(in_ir, out_ir)
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    try:
        with _processing_status("Load IR for processing", enabled=not silent):
            audio, sr = sf.read(str(in_ir), always_2d=True, dtype="float64")
            base_audio = np.asarray(audio, dtype=np.float64)
            sr_i = int(sr)
        if lucky is None:
            with _processing_status("Process IR", enabled=not silent):
                processed = apply_ir_shaping(
                    base_audio,
                    sr=sr_i,
                    damping=damping,
                    lowcut=lowcut,
                    highcut=highcut,
                    tilt=tilt,
                    normalize=normalize,
                    peak_dbfs=peak_dbfs,
                    target_lufs=target_lufs,
                    use_true_peak=true_peak,
                )

                meta = {"source": str(in_ir), "metrics": analyze_ir(processed, sr_i)}
                write_ir_artifacts(out_ir, processed, sr_i, meta, silent=silent)
            return

        out_dir = out_ir.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_value = _resolve_lucky_seed(lucky_seed)

        rows: list[dict[str, str]] = []
        with _BatchStatusBar(
            total=lucky,
            label="Lucky IR processing",
            enabled=not silent,
        ) as status:
            for idx in range(lucky):
                rng = np.random.default_rng(seed_value + idx)
                cfg = _build_lucky_ir_process_config(
                    damping=damping,
                    lowcut=lowcut,
                    highcut=highcut,
                    tilt=tilt,
                    normalize=normalize,
                    peak_dbfs=peak_dbfs,
                    target_lufs=target_lufs,
                    true_peak=true_peak,
                    rng=rng,
                    sr=sr_i,
                )
                lucky_out = out_dir / f"{out_ir.stem}.lucky_{idx + 1:03d}{out_ir.suffix}"
                processed = apply_ir_shaping(
                    base_audio,
                    sr=sr_i,
                    damping=cfg["damping"],
                    lowcut=cfg["lowcut"],
                    highcut=cfg["highcut"],
                    tilt=cfg["tilt"],
                    normalize=cfg["normalize"],
                    peak_dbfs=cfg["peak_dbfs"],
                    target_lufs=cfg["target_lufs"],
                    use_true_peak=cfg["true_peak"],
                )

                meta = {
                    "source": str(in_ir),
                    "lucky": {"index": idx + 1, **cfg},
                    "metrics": analyze_ir(processed, sr_i),
                }
                write_ir_artifacts(lucky_out, processed, sr_i, meta, silent=silent)
                rows.append(
                    {
                        "index": str(idx + 1),
                        "out_ir": str(lucky_out),
                        "normalize": cfg["normalize"],
                        "tilt": f"{float(cfg['tilt']):.2f}",
                        "damping": f"{float(cfg['damping']):.2f}",
                        "target_lufs": (
                            f"{float(cfg['target_lufs']):.2f}"
                            if cfg["target_lufs"] is not None
                            else "none"
                        ),
                    }
                )
                status.advance(detail=f"seed={seed_value + idx}")

        if not silent:
            table = Table(title=f"Lucky IR Process Batch ({lucky} outputs)")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("out_ir", style="white")
            table.add_column("normalize", style="green")
            table.add_column("tilt", justify="right")
            table.add_column("damping", justify="right")
            table.add_column("target_lufs", justify="right")
            for row in rows:
                table.add_row(
                    row["index"],
                    row["out_ir"],
                    row["normalize"],
                    row["tilt"],
                    row["damping"],
                    row["target_lufs"],
                )
            console.print(table)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@ir_app.command("morph")
def ir_morph(
    ir_a: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    ir_b: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    mode: str = typer.Option(
        "equal-power",
        "--mode",
        help="Morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0),
    early_ms: float = typer.Option(
        80.0,
        "--early-ms",
        min=0.0,
        help="Early/late split used by split/envelope-aware morphing (ms).",
    ),
    early_alpha: float | None = typer.Option(
        None,
        "--early-alpha",
        min=0.0,
        max=1.0,
        help="Optional alpha override for early-reflection region.",
    ),
    late_alpha: float | None = typer.Option(
        None,
        "--late-alpha",
        min=0.0,
        max=1.0,
        help="Optional alpha override for late-tail region.",
    ),
    align_decay: bool = typer.Option(
        True,
        "--align-decay/--no-align-decay",
        help="Align decay profiles before morphing for stable RT trajectories.",
    ),
    phase_coherence: float = typer.Option(
        0.75,
        "--phase-coherence",
        min=0.0,
        max=1.0,
        help="Phase-coherence safeguard strength for spectral morphing.",
    ),
    spectral_smooth_bins: int = typer.Option(
        3,
        "--spectral-smooth-bins",
        min=0,
        max=128,
        help="Frequency smoothing radius (FFT bins) used by spectral modes.",
    ),
    mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--mismatch-policy",
        help=(
            "Mismatch behavior for sample-rate/channel/duration differences: "
            "coerce (align) or strict (fail)."
        ),
    ),
    target_sr: int | None = typer.Option(
        None,
        "--target-sr",
        min=1,
        help="Optional target sample rate for morph processing and output.",
    ),
    cache_dir: str = typer.Option(".verbx_cache/ir_morph", "--cache-dir"),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Morph two IR files with cache-backed Track D processing."""
    _validate_ir_morph_call(
        ir_a=ir_a,
        ir_b=ir_b,
        out_ir=out_ir,
        mode=mode,
        early_alpha=early_alpha,
        late_alpha=late_alpha,
        mismatch_policy=mismatch_policy,
        cache_dir=cache_dir,
    )

    cfg = IRMorphConfig(
        mode=cast(
            Literal["linear", "equal-power", "spectral", "envelope-aware"],
            validate_ir_morph_mode_name(mode),
        ),
        alpha=float(alpha),
        early_ms=float(early_ms),
        early_alpha=None if early_alpha is None else float(early_alpha),
        late_alpha=None if late_alpha is None else float(late_alpha),
        align_decay=bool(align_decay),
        phase_coherence=float(phase_coherence),
        spectral_smooth_bins=int(spectral_smooth_bins),
        mismatch_policy=cast(
            IRMorphMismatchPolicy,
            normalize_ir_morph_mismatch_policy_name(mismatch_policy),
        ),
    )

    try:
        with _processing_status("Morph IRs", enabled=not silent):
            audio, sr, meta, cache_path, cache_hit = generate_or_load_cached_morphed_ir(
                ir_a_path=ir_a,
                ir_b_path=ir_b,
                config=cfg,
                cache_dir=Path(cache_dir),
                target_sr=None if target_sr is None else int(target_sr),
            )
            write_ir_artifacts(out_ir, audio, sr, meta, silent=silent)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if silent:
        return

    table = Table(title="IR Morph")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", cfg.mode)
    table.add_row("alpha", f"{cfg.alpha:.3f}")
    table.add_row("early_ms", f"{cfg.early_ms:.2f}")
    table.add_row("mismatch_policy", str(cfg.mismatch_policy))
    table.add_row("out_ir", str(out_ir))
    table.add_row("cache_path", str(cache_path))
    table.add_row("cache_hit", str(cache_hit))
    table.add_row("sample_rate", str(int(sr)))
    table.add_row("channels", str(int(audio.shape[1])))
    table.add_row("duration_s", f"{float(audio.shape[0]) / float(sr):.3f}")
    quality = meta.get("quality", {})
    if isinstance(quality, dict):
        drift = quality.get("rt60_drift_s")
        if drift is not None:
            table.add_row("rt60_drift_s", f"{float(drift):.4f}")
        spectral = quality.get("spectral_distance_db")
        if spectral is not None:
            table.add_row("spectral_distance_db", f"{float(spectral):.4f}")
    console.print(table)


@ir_app.command("morph-sweep")
def ir_morph_sweep(
    ir_a: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    ir_b: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_dir: Path = typer.Argument(..., resolve_path=True),
    mode: str = typer.Option(
        "equal-power",
        "--mode",
        help="Morph mode: linear, equal-power, spectral, or envelope-aware.",
    ),
    alpha_points: list[float] | None = typer.Option(
        None,
        "--alpha",
        min=0.0,
        max=1.0,
        help="Explicit alpha point. Repeat to define custom sweep timeline.",
    ),
    alpha_start: float = typer.Option(0.0, "--alpha-start", min=0.0, max=1.0),
    alpha_end: float = typer.Option(1.0, "--alpha-end", min=0.0, max=1.0),
    alpha_steps: int = typer.Option(9, "--alpha-steps", min=2, max=257),
    out_prefix: str = typer.Option(
        "morph",
        "--out-prefix",
        help="Output filename prefix for generated sweep IR files.",
    ),
    early_ms: float = typer.Option(80.0, "--early-ms", min=0.0),
    early_alpha: float | None = typer.Option(None, "--early-alpha", min=0.0, max=1.0),
    late_alpha: float | None = typer.Option(None, "--late-alpha", min=0.0, max=1.0),
    align_decay: bool = typer.Option(
        True,
        "--align-decay/--no-align-decay",
        help="Align decay profiles before morphing for stable RT trajectories.",
    ),
    phase_coherence: float = typer.Option(0.75, "--phase-coherence", min=0.0, max=1.0),
    spectral_smooth_bins: int = typer.Option(3, "--spectral-smooth-bins", min=0, max=128),
    mismatch_policy: IRMorphMismatchPolicy = typer.Option(
        "coerce",
        "--mismatch-policy",
        help=(
            "Mismatch behavior for sample-rate/channel/duration differences: "
            "coerce (align) or strict (fail)."
        ),
    ),
    target_sr: int | None = typer.Option(None, "--target-sr", min=1),
    cache_dir: str = typer.Option(".verbx_cache/ir_morph", "--cache-dir"),
    workers: int = typer.Option(0, "--workers", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    fail_if_any_failed: bool = typer.Option(
        True,
        "--fail-if-any-failed/--allow-failed",
        help="Exit non-zero when any sweep step fails.",
    ),
    checkpoint_file: Path | None = typer.Option(
        None,
        "--checkpoint-file",
        resolve_path=True,
        help="Optional checkpoint JSON path for resume-safe sweep execution.",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from --checkpoint-file."),
    qa_json_out: Path | None = typer.Option(
        None,
        "--qa-json-out",
        resolve_path=True,
        help="Summary JSON output path (default: <out_dir>/morph_sweep_summary.json).",
    ),
    qa_csv_out: Path | None = typer.Option(
        None,
        "--qa-csv-out",
        resolve_path=True,
        help="Per-step QA metrics CSV path (default: <out_dir>/morph_sweep_metrics.csv).",
    ),
    silent: bool = typer.Option(False, "--silent"),
) -> None:
    """Run an alpha timeline sweep and emit Track D QA artifacts."""
    _validate_ir_morph_sweep_call(
        ir_a=ir_a,
        ir_b=ir_b,
        out_dir=out_dir,
        mode=mode,
        early_alpha=early_alpha,
        late_alpha=late_alpha,
        mismatch_policy=mismatch_policy,
        cache_dir=cache_dir,
        out_prefix=out_prefix,
        alpha_points=alpha_points,
        alpha_steps=alpha_steps,
        checkpoint_file=checkpoint_file,
        resume=resume,
    )
    resolved_mode = validate_ir_morph_mode_name(mode)
    alphas = _resolve_ir_morph_sweep_alphas(
        alpha_points=alpha_points,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_steps=alpha_steps,
    )
    out_root = out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    qa_json_path = (
        qa_json_out.resolve()
        if qa_json_out is not None
        else (out_root / "morph_sweep_summary.json").resolve()
    )
    qa_csv_path = (
        qa_csv_out.resolve()
        if qa_csv_out is not None
        else (out_root / "morph_sweep_metrics.csv").resolve()
    )
    checkpoint_path = checkpoint_file.resolve() if checkpoint_file is not None else None

    template_cfg = IRMorphConfig(
        mode=cast(
            Literal["linear", "equal-power", "spectral", "envelope-aware"],
            resolved_mode,
        ),
        alpha=0.0,
        early_ms=float(early_ms),
        early_alpha=None if early_alpha is None else float(early_alpha),
        late_alpha=None if late_alpha is None else float(late_alpha),
        align_decay=bool(align_decay),
        phase_coherence=float(phase_coherence),
        spectral_smooth_bins=int(spectral_smooth_bins),
        mismatch_policy=cast(
            IRMorphMismatchPolicy,
            normalize_ir_morph_mismatch_policy_name(mismatch_policy),
        ),
    )

    all_jobs: list[BatchJobSpec] = []
    alpha_by_index: dict[int, float] = {}
    for idx, alpha_value in enumerate(alphas, start=1):
        token = _alpha_token(alpha_value)
        outfile = out_root / f"{out_prefix}_{idx:03d}_{token}.wav"
        all_jobs.append(
            BatchJobSpec(
                index=idx,
                infile=ir_a,
                outfile=outfile,
                config=RenderConfig(progress=False),
                estimated_cost=1.0 + abs(float(alpha_value) - 0.5),
            )
        )
        alpha_by_index[idx] = float(alpha_value)

    checkpoint_payload: dict[str, Any] = {
        "version": IR_MORPH_SWEEP_VERSION,
        "mode": "ir-morph-sweep",
        "ir_a": str(ir_a.resolve()),
        "ir_b": str(ir_b.resolve()),
        "out_dir": str(out_root),
        "results": [],
    }
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_payload = _load_batch_checkpoint(checkpoint_path)
        checkpoint_payload.setdefault("results", [])
    completed_outfiles = (
        _checkpoint_success_outfiles(checkpoint_payload)
        if (resume and checkpoint_path is not None)
        else set()
    )
    prepared_jobs = [
        job for job in all_jobs if str(job.outfile.resolve()) not in completed_outfiles
    ]

    active_count = len(prepared_jobs) if prepared_jobs else 1
    if workers == 0:
        max_workers = max(1, min(int(os.cpu_count() or 1), active_count))
    else:
        max_workers = max(1, min(int(workers), active_count))

    runtime_meta: dict[int, dict[str, Any]] = {}
    checkpoint_results = checkpoint_payload.get("results", [])
    if not isinstance(checkpoint_results, list):
        checkpoint_results = []
        checkpoint_payload["results"] = checkpoint_results
    lock = Lock()

    def runner(job: BatchJobSpec) -> None:
        alpha_value = alpha_by_index[job.index]
        cfg = IRMorphConfig(
            mode=template_cfg.mode,
            alpha=float(alpha_value),
            early_ms=float(template_cfg.early_ms),
            early_alpha=template_cfg.early_alpha,
            late_alpha=template_cfg.late_alpha,
            align_decay=bool(template_cfg.align_decay),
            phase_coherence=float(template_cfg.phase_coherence),
            spectral_smooth_bins=int(template_cfg.spectral_smooth_bins),
            mismatch_policy=template_cfg.mismatch_policy,
        )
        audio, sr, meta, cache_path, cache_hit = generate_or_load_cached_morphed_ir(
            ir_a_path=ir_a,
            ir_b_path=ir_b,
            config=cfg,
            cache_dir=Path(cache_dir),
            target_sr=None if target_sr is None else int(target_sr),
        )
        write_ir_artifacts(job.outfile, audio, sr, meta, silent=True)
        quality = meta.get("quality", {})
        with lock:
            runtime_meta[job.index] = {
                "alpha": float(alpha_value),
                "cache_path": str(cache_path),
                "cache_hit": bool(cache_hit),
                "sample_rate": int(sr),
                "channels": int(audio.shape[1]),
                "duration_s": float(audio.shape[0]) / float(max(1, sr)),
                "quality": quality if isinstance(quality, dict) else {},
            }

    def on_result(result: BatchJobResult) -> None:
        meta = runtime_meta.get(result.index, {})
        row = {
            "index": int(result.index),
            "alpha": float(meta.get("alpha", alpha_by_index.get(result.index, 0.0))),
            "outfile": str(result.outfile.resolve()),
            "success": bool(result.success),
            "attempts": int(result.attempts),
            "duration_seconds": float(result.duration_seconds),
            "cache_hit": bool(meta.get("cache_hit", False)),
            "cache_path": str(meta.get("cache_path", "")),
            "sample_rate": int(meta.get("sample_rate", 0)),
            "channels": int(meta.get("channels", 0)),
            "render_duration_s": float(meta.get("duration_s", 0.0)),
            "quality": meta.get("quality", {}),
            "error": result.error,
        }
        with lock:
            _upsert_checkpoint_row(checkpoint_results, row)
            if checkpoint_path is not None:
                _write_json_atomic(checkpoint_path, checkpoint_payload)
        if silent:
            return
        status = "ok" if result.success else "failed"
        console.print(
            f"morph-sweep {status} {result.index}: {result.outfile} "
            f"(alpha={row['alpha']:.3f}, attempts={result.attempts})"
        )

    run_error: str | None = None
    if prepared_jobs:
        with _BatchStatusBar(
            total=len(prepared_jobs),
            label="IR morph sweep",
            enabled=not silent,
        ) as status:
            original_on_result = on_result

            def on_result_with_status(result: BatchJobResult) -> None:
                original_on_result(result)
                status.advance(detail=f"job={result.index}")

            try:
                run_parallel_batch(
                    jobs=prepared_jobs,
                    max_workers=max_workers,
                    schedule=schedule,
                    retries=retries,
                    continue_on_error=continue_on_error,
                    runner=runner,
                    on_result=on_result_with_status,
                )
            except RuntimeError as exc:
                run_error = str(exc)

    checkpoint_by_outfile: dict[str, dict[str, Any]] = {}
    for row in checkpoint_results:
        if not isinstance(row, dict):
            continue
        outfile = row.get("outfile")
        if not isinstance(outfile, str):
            continue
        checkpoint_by_outfile[str(Path(outfile).resolve())] = dict(row)

    qa_rows: list[dict[str, Any]] = []
    resumed_skipped = 0
    executed_outfiles = {str(job.outfile.resolve()) for job in prepared_jobs}
    for job in all_jobs:
        out_key = str(job.outfile.resolve())
        row = checkpoint_by_outfile.get(out_key, {})
        merged: dict[str, Any] = {
            "index": int(job.index),
            "alpha": float(row.get("alpha", alpha_by_index[job.index])),
            "outfile": out_key,
            "success": bool(row.get("success", False)),
            "attempts": int(row.get("attempts", 0)),
            "duration_seconds": float(row.get("duration_seconds", 0.0)),
            "cache_hit": bool(row.get("cache_hit", False)),
            "cache_path": str(row.get("cache_path", "")),
            "sample_rate": int(row.get("sample_rate", 0)),
            "channels": int(row.get("channels", 0)),
            "render_duration_s": float(row.get("render_duration_s", 0.0)),
            "error": row.get("error"),
        }
        quality_raw = row.get("quality", {})
        quality = quality_raw if isinstance(quality_raw, dict) else {}
        for key in (
            "rt60_target_s",
            "rt60_out_s",
            "rt60_drift_s",
            "early_late_target_db",
            "early_late_out_db",
            "early_late_drift_db",
            "spectral_distance_db",
            "interchannel_coherence_target",
            "interchannel_coherence_out",
            "interchannel_coherence_delta",
        ):
            value = quality.get(key)
            merged[key] = None if value is None else float(value)
        resumed = bool(out_key in completed_outfiles and out_key not in executed_outfiles)
        merged["resumed_skip"] = resumed
        if resumed:
            resumed_skipped += 1
        qa_rows.append(merged)

    _write_csv_atomic(qa_csv_path, qa_rows)
    success = int(sum(1 for row in qa_rows if bool(row.get("success", False))))
    failed = int(len(qa_rows) - success)
    summary_payload = {
        "version": IR_MORPH_SWEEP_VERSION,
        "mode": "ir-morph-sweep",
        "ir_a": str(ir_a.resolve()),
        "ir_b": str(ir_b.resolve()),
        "out_dir": str(out_root),
        "morph_mode": resolved_mode,
        "mismatch_policy": str(template_cfg.mismatch_policy),
        "alpha_values": [float(value) for value in alphas],
        "workers": int(max_workers),
        "schedule": str(schedule),
        "retries": int(retries),
        "continue_on_error": bool(continue_on_error),
        "planned": len(all_jobs),
        "executed": len(prepared_jobs),
        "resumed_skipped": int(resumed_skipped),
        "success": int(success),
        "failed": int(failed),
        "qa_csv": str(qa_csv_path),
        "checkpoint_file": None if checkpoint_path is None else str(checkpoint_path),
        "rt60_drift_s_stats": _summarize_numeric_column(qa_rows, "rt60_drift_s"),
        "spectral_distance_db_stats": _summarize_numeric_column(qa_rows, "spectral_distance_db"),
    }
    _write_json_atomic(qa_json_path, summary_payload)

    if not silent:
        table = Table(title="IR Morph Sweep Summary")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        for key in (
            "morph_mode",
            "mismatch_policy",
            "planned",
            "executed",
            "resumed_skipped",
            "success",
            "failed",
            "qa_csv",
        ):
            table.add_row(key, str(summary_payload.get(key, "")))
        table.add_row("qa_json", str(qa_json_path))
        console.print(table)

    if run_error is not None and not continue_on_error:
        if fail_if_any_failed:
            raise typer.Exit(code=2)
        raise typer.BadParameter(run_error)

    if fail_if_any_failed and failed > 0:
        raise typer.Exit(code=2)


@ir_app.command("fit")
def ir_fit(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_ir: Path = typer.Argument(..., resolve_path=True),
    top_k: int = typer.Option(3, "--top-k", min=1),
    base_mode: IRMode = typer.Option("hybrid", "--base-mode"),
    length: float = typer.Option(60.0, "--length", min=0.1),
    seed: int = typer.Option(0, "--seed"),
    candidate_pool: int = typer.Option(12, "--candidate-pool", min=1),
    fit_workers: int = typer.Option(0, "--fit-workers", min=0, help="0 = auto"),
    analyze_tuning: bool = typer.Option(True, "--analyze-tuning/--no-analyze-tuning"),
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Analyze source audio, score candidate IRs, and write top-k results."""
    _validate_output_audio_path(out_ir, "auto")
    try:
        with _processing_status("Analyze source for IR fit"):
            audio, sr = read_audio(str(infile))
            analyzer = AudioAnalyzer()
            metrics = analyzer.analyze(audio, sr)
    except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    pool_size = max(top_k, candidate_pool)
    target_profile = derive_ir_fit_target(metrics, sr)

    f0_hz: float | None = None
    harmonics: tuple[float, ...] = ()
    if analyze_tuning:
        try:
            f0_est, harmonic_est = analyze_audio_for_tuning(infile, max_harmonics=12)
            f0_hz = f0_est
            harmonics = tuple(harmonic_est)
        except (ValueError, RuntimeError, FileNotFoundError, sf.LibsndfileError):
            f0_hz = None
            harmonics = ()

    candidates = build_ir_fit_candidates(
        base_mode=base_mode,
        length=length,
        sr=sr,
        channels=max(1, min(2, audio.shape[1])),
        seed=seed,
        pool_size=pool_size,
        target=target_profile,
        f0_hz=f0_hz,
        harmonic_targets_hz=harmonics,
    )

    cache_root = Path(cache_dir)
    scored = _score_fit_candidates(
        candidates=candidates,
        target=target_profile,
        cache_dir=cache_root,
        fit_workers=fit_workers,
        show_progress=True,
    )

    selected = sorted(
        scored,
        key=lambda item: item.score.score,
        reverse=True,
    )[:top_k]

    created: list[str] = []
    with _BatchStatusBar(total=len(selected), label="Write fitted IRs", enabled=True) as status:
        for rank, item in enumerate(selected, start=1):
            target_path = (
                out_ir
                if top_k == 1
                else out_ir.with_name(f"{out_ir.stem}_{rank:02d}{out_ir.suffix}")
            )
            meta = dict(item.meta)
            meta["fit"] = {
                "rank": rank,
                "score": item.score.score,
                "strategy": item.candidate.strategy,
                "target": asdict(target_profile),
                "errors": asdict(item.score),
                "detail_metrics": item.detail_metrics,
            }
            cached_audio, _ = sf.read(str(item.cache_path), always_2d=True, dtype="float64")
            write_ir_artifacts(
                target_path,
                np.asarray(cached_audio, dtype=np.float64),
                item.sr,
                meta,
                silent=False,
            )
            created.append(str(target_path))
            status.advance(detail=f"rank={rank}")

    table = Table(title="IR Fit")
    table.add_column("Field", style="green")
    table.add_column("Value", style="white")
    table.add_row("input", str(infile))
    table.add_row("top_k", str(top_k))
    table.add_row("candidate_pool", str(pool_size))
    table.add_row("target_rt60", f"{target_profile.rt60_seconds:.2f}")
    table.add_row("target_early_late_db", f"{target_profile.early_late_ratio_db:.2f}")
    table.add_row("target_coherence", f"{target_profile.stereo_coherence:.3f}")
    if f0_hz is not None:
        table.add_row("detected_f0_hz", f"{f0_hz:.3f}")
    if selected:
        table.add_row("best_score", f"{selected[0].score.score:.5f}")
        table.add_row("best_strategy", selected[0].candidate.strategy)
    table.add_row("outputs", "\n".join(created))
    console.print(table)


@cache_app.command("info")
def cache_info(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Show cache statistics."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    wavs = sorted(root.glob("*.wav"))
    metas = sorted(root.glob("*.meta.json"))
    total_bytes = (
        sum(path.stat().st_size for path in root.glob("*") if path.is_file())
        if root.exists()
        else 0
    )

    table = Table(title="Cache Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("cache_dir", str(root))
    table.add_row("wav_files", str(len(wavs)))
    table.add_row("meta_files", str(len(metas)))
    table.add_row("size_mb", f"{total_bytes / (1024 * 1024):.3f}")
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Clear IR cache directory."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    console.print(f"Cleared cache: {root}")


@batch_app.command("template")
def batch_template() -> None:
    """Print a batch manifest template as JSON."""
    template = {
        "version": BATCH_MANIFEST_VERSION,
        "jobs": [
            {
                "infile": "input.wav",
                "outfile": "output.wav",
                "options": {
                    "engine": "auto",
                    "rt60": 60.0,
                    "wet": 0.8,
                    "dry": 0.2,
                    "repeat": 1,
                },
            }
        ],
    }
    typer.echo(json.dumps(template, indent=2))


@batch_app.command("augment-template")
def batch_augment_template() -> None:
    """Print an AI/data-augmentation manifest template as JSON."""
    template = build_augmentation_manifest_template()
    template["profiles"] = {
        name: {"description": str(data.get("description", ""))}
        for name, data in augmentation_profiles().items()
    }
    typer.echo(json.dumps(template, indent=2))


@batch_app.command("augment-profiles")
def batch_augment_profiles(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit profile definitions as JSON instead of a table.",
    ),
) -> None:
    """List built-in augmentation profiles and archetypes."""
    profiles = augmentation_profiles()
    if as_json:
        typer.echo(json.dumps(profiles, indent=2, sort_keys=True))
        return

    table = Table(title="Batch Augment Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Archetypes", style="magenta")
    table.add_column("Description", style="white")
    for name in augmentation_profile_names():
        data = profiles.get(name, {})
        archetypes = data.get("archetypes", [])
        archetype_names = ", ".join(
            str(item.get("name", "")) for item in archetypes if isinstance(item, dict)
        )
        table.add_row(
            name,
            archetype_names,
            str(data.get("description", "")),
        )
    console.print(table)


@batch_app.command("augment")
def batch_augment(
    manifest: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        resolve_path=True,
        help="Override output root directory from manifest.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Optional profile override (for quick profile A/B against one manifest).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Optional deterministic seed override.",
    ),
    variants_per_input: int | None = typer.Option(
        None,
        "--variants-per-input",
        min=1,
        max=500,
        help="Optional variants-per-source override.",
    ),
    write_analysis: bool | None = typer.Option(
        None,
        "--write-analysis/--no-write-analysis",
        help="Override manifest write_analysis behavior.",
    ),
    copy_dry: bool = typer.Option(
        False,
        "--copy-dry/--no-copy-dry",
        help="Copy clean source files into output tree (paired dry/wet dataset layout).",
    ),
    verify_split_isolation: bool = typer.Option(
        True,
        "--verify-split-isolation/--allow-split-overlap",
        help=(
            "Require one source id/input file to belong to exactly one split "
            "(prevents train/val/test leakage)."
        ),
    ),
    jobs: int = typer.Option(0, "--jobs", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    fail_if_any_failed: bool = typer.Option(
        True,
        "--fail-if-any-failed/--allow-failed",
        help="Exit non-zero when any augmentation render fails.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    jsonl_out: Path | None = typer.Option(
        None,
        "--jsonl-out",
        resolve_path=True,
        help=(
            "Path for dataset metadata JSONL "
            "(default: <output_root>/augmentation_manifest.jsonl)."
        ),
    ),
    summary_out: Path | None = typer.Option(
        None,
        "--summary-out",
        resolve_path=True,
        help="Path for run summary JSON (default: <output_root>/augmentation_summary.json).",
    ),
    dataset_card_out: Path | None = typer.Option(
        None,
        "--dataset-card-out",
        resolve_path=True,
        help="Optional Markdown dataset-card path for ML dataset documentation.",
    ),
    metrics_csv_out: Path | None = typer.Option(
        None,
        "--metrics-csv-out",
        resolve_path=True,
        help="Optional CSV path for per-output acoustic features.",
    ),
    metrics_include_loudness: bool = typer.Option(
        False,
        "--metrics-include-loudness/--metrics-fast",
        help="Include LUFS/true-peak/LRA in metrics CSV export (slower).",
    ),
    qa_bundle_out: Path | None = typer.Option(
        None,
        "--qa-bundle-out",
        resolve_path=True,
        help=(
            "Optional QA bundle JSON path (default: <output_root>/augmentation_qa_bundle.json)."
        ),
    ),
    baseline_summary: Path | None = typer.Option(
        None,
        "--baseline-summary",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Optional prior augmentation summary/QA bundle JSON used for regeneration deltas.",
    ),
    provenance_hash: bool = typer.Option(
        False,
        "--provenance-hash/--no-provenance-hash",
        help="Emit deterministic provenance hash over manifest payload + source signatures.",
    ),
) -> None:
    """Render a deterministic augmentation dataset for AI research workflows."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Augmentation manifest must be a JSON object.")

    if profile is not None:
        payload["profile"] = str(profile)
    if seed is not None:
        payload["seed"] = int(seed)
    if variants_per_input is not None:
        payload["variants_per_input"] = int(variants_per_input)
    if write_analysis is not None:
        payload["write_analysis"] = bool(write_analysis)

    try:
        build = build_augmentation_plans(
            payload=cast(dict[str, Any], payload),
            manifest_path=manifest,
            output_root_override=output_root,
            copy_dry=copy_dry,
            verify_split_isolation=verify_split_isolation,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if len(build.plans) == 0:
        raise typer.BadParameter("No augmentation plans were generated from manifest.")

    for plan in build.plans:
        if plan.infile.resolve() == plan.outfile.resolve():
            msg = (
                f"Augmentation source #{plan.source_index} resolved identical input/output path: "
                f"{plan.infile}"
            )
            raise typer.BadParameter(msg)
        if not plan.infile.exists():
            raise typer.BadParameter(f"Augmentation infile not found: {plan.infile}")
        _validate_output_audio_path(plan.outfile, str(plan.config.output_subtype))

    if dry_run:
        _print_augmentation_dry_run(build=build, schedule=schedule)
        return

    _prepare_augmentation_output_dirs(build=build)
    _copy_augmentation_dry_sources(build=build)

    prepared_jobs = [
        BatchJobSpec(
            index=plan.index,
            infile=plan.infile,
            outfile=plan.outfile,
            config=plan.config,
            estimated_cost=estimate_job_cost(plan.infile, plan.config),
        )
        for plan in build.plans
    ]

    max_workers = int(os.cpu_count() or 1) if jobs == 0 else int(jobs)
    max_workers = max(1, min(max_workers, len(prepared_jobs)))
    results_by_index: dict[int, BatchJobResult] = {}

    def runner(job: BatchJobSpec) -> None:
        run_render_pipeline(infile=job.infile, outfile=job.outfile, config=job.config)

    def on_result(result: BatchJobResult) -> None:
        results_by_index[int(result.index)] = result
        if result.success:
            console.print(
                f"augmented {result.index}: {result.outfile} "
                f"(attempts={result.attempts}, {result.duration_seconds:.2f}s)"
            )
        else:
            console.print(
                f"augment-failed {result.index}: {result.outfile} "
                f"(attempts={result.attempts}) {result.error}"
            )

    with _BatchStatusBar(total=len(prepared_jobs), label="Batch augment", enabled=True) as status:
        def on_result_with_status(result: BatchJobResult) -> None:
            on_result(result)
            status.advance(detail=f"job={result.index}")

        try:
            run_parallel_batch(
                jobs=prepared_jobs,
                max_workers=max_workers,
                schedule=schedule,
                retries=retries,
                continue_on_error=continue_on_error,
                runner=runner,
                on_result=on_result_with_status,
            )
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc

    out_root = build.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = (
        jsonl_out.resolve()
        if jsonl_out is not None
        else (out_root / "augmentation_manifest.jsonl").resolve()
    )
    summary_path = (
        summary_out.resolve()
        if summary_out is not None
        else (out_root / "augmentation_summary.json").resolve()
    )
    qa_bundle_path = (
        qa_bundle_out.resolve()
        if qa_bundle_out is not None
        else (out_root / "augmentation_qa_bundle.json").resolve()
    )
    records = _build_augmentation_records(build=build, results_by_index=results_by_index)
    _write_jsonl_atomic(jsonl_path, records)
    summary_payload = _build_augmentation_summary(
        build=build,
        records=records,
        manifest_path=manifest,
        jsonl_path=jsonl_path,
        schedule=schedule,
        jobs=max_workers,
    )
    baseline_payload: dict[str, Any] | None = None
    if baseline_summary is not None:
        baseline_payload = _load_optional_summary_payload(baseline_summary)
        summary_payload["baseline_summary"] = str(baseline_summary.resolve())

    qa_metric_rows: list[dict[str, Any]] = []
    if metrics_csv_out is not None:
        metrics_path = metrics_csv_out.resolve()
        metrics_rows = _build_augmentation_metrics_rows(
            records=records,
            include_loudness=metrics_include_loudness,
        )
        _write_csv_atomic(metrics_path, metrics_rows)
        summary_payload["metrics_csv"] = str(metrics_path)
        summary_payload["metrics_rows"] = len(metrics_rows)
        summary_payload["metrics_include_loudness"] = bool(metrics_include_loudness)
        qa_metric_rows = metrics_rows

    if len(qa_metric_rows) == 0:
        qa_metric_rows = _build_augmentation_metrics_rows(
            records=records,
            include_loudness=False,
        )
    qa_bundle_payload = _build_augmentation_qa_bundle(
        build=build,
        records=records,
        metrics_rows=qa_metric_rows,
        baseline_summary=baseline_payload,
    )
    _write_json_atomic(qa_bundle_path, qa_bundle_payload)
    summary_payload["qa_bundle"] = str(qa_bundle_path)
    summary_payload["qa_bundle_version"] = str(qa_bundle_payload.get("version", ""))

    if provenance_hash:
        provenance = _compute_augmentation_provenance_hash(
            build=build,
            manifest=manifest,
            records=records,
        )
        summary_payload["provenance_hash"] = str(provenance.get("sha256", ""))
        summary_payload["provenance"] = provenance

    if dataset_card_out is not None:
        dataset_card_path = dataset_card_out.resolve()
        card = _build_augmentation_dataset_card(
            build=build,
            summary=summary_payload,
            manifest_path=manifest,
            summary_path=summary_path,
            records=records,
        )
        _write_text_atomic(dataset_card_path, card)
        summary_payload["dataset_card"] = str(dataset_card_path)

    _write_json_atomic(summary_path, summary_payload)
    _print_augmentation_summary_table(summary_payload, summary_path=summary_path)

    failed = int(summary_payload.get("failed", 0))
    if fail_if_any_failed and failed > 0:
        raise typer.Exit(code=2)


@batch_app.command("render")
def batch_render(
    manifest: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    jobs: int = typer.Option(0, "--jobs", min=0, help="0 = auto"),
    schedule: BatchSchedulePolicy = typer.Option("longest-first", "--schedule"),
    retries: int = typer.Option(0, "--retries", min=0),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast"),
    checkpoint_file: Path | None = typer.Option(
        None,
        "--checkpoint-file",
        resolve_path=True,
        help="Optional checkpoint file used to persist per-job completion state.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from --checkpoint-file and skip already completed jobs.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    lucky: int | None = typer.Option(
        None,
        "--lucky",
        min=1,
        max=500,
        help=(
            "For each manifest job, generate N wild random render variants. "
            "Outputs are written to --lucky-out-dir (or each job OUTFILE parent by default)."
        ),
    ),
    lucky_out_dir: Path | None = typer.Option(
        None,
        "--lucky-out-dir",
        resolve_path=True,
        help="Output directory used when --lucky is enabled.",
    ),
    lucky_seed: int | None = typer.Option(
        None,
        "--lucky-seed",
        help="Optional deterministic seed for --lucky batch generation.",
    ),
) -> None:
    """Render jobs from manifest.json."""
    if resume and checkpoint_file is None:
        msg = "--resume requires --checkpoint-file."
        raise typer.BadParameter(msg)
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON manifest: {exc}"
        raise typer.BadParameter(msg) from exc
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise typer.BadParameter("Manifest must contain a top-level 'jobs' array")

    job_list = payload["jobs"]
    if not isinstance(job_list, list):
        raise typer.BadParameter("jobs must be a list")

    prepared_jobs: list[BatchJobSpec] = []
    prepared_index = 1
    lucky_seed_value = _resolve_lucky_seed(lucky_seed) if lucky is not None else 0

    for idx, job in enumerate(job_list, start=1):
        if not isinstance(job, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}] must be an object")
        infile = Path(str(job.get("infile", "")))
        outfile = Path(str(job.get("outfile", "")))
        options = job.get("options", {})
        if not isinstance(options, dict):
            raise typer.BadParameter(f"jobs[{idx - 1}].options must be an object")

        try:
            render_config = _render_config_from_options(options)
        except (TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] has invalid options: {exc}"
            raise typer.BadParameter(msg) from exc
        _validate_batch_job_paths(infile, outfile, idx)
        if lucky is None:
            prepared_jobs.append(
                BatchJobSpec(
                    index=prepared_index,
                    infile=infile,
                    outfile=outfile,
                    config=render_config,
                    estimated_cost=estimate_job_cost(infile, render_config),
                )
            )
            prepared_index += 1
            continue

        try:
            info = sf.info(str(infile))
            duration_seconds = (
                float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = f"jobs[{idx - 1}] failed to inspect infile: {exc}"
            raise typer.BadParameter(msg) from exc

        out_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant_idx in range(lucky):
            rng = np.random.default_rng(lucky_seed_value + ((idx - 1) * lucky) + variant_idx)
            lucky_config = _build_lucky_config(
                base=render_config,
                rng=rng,
                input_duration_seconds=duration_seconds,
            )
            lucky_config.progress = False
            lucky_config.analysis_out = None
            lucky_config.frames_out = None
            lucky_out = out_dir / f"{outfile.stem}.lucky_{variant_idx + 1:03d}{outfile.suffix}"
            prepared_jobs.append(
                BatchJobSpec(
                    index=prepared_index,
                    infile=infile,
                    outfile=lucky_out,
                    config=lucky_config,
                    estimated_cost=estimate_job_cost(infile, lucky_config),
                )
            )
            prepared_index += 1

    if not prepared_jobs:
        raise typer.BadParameter("Manifest has no jobs to render")

    checkpoint_payload: dict[str, Any] | None = None
    resumed_outfiles: set[str] = set()
    if checkpoint_file is not None:
        if resume:
            checkpoint_payload = _load_batch_checkpoint(checkpoint_file)
            resumed_outfiles = _checkpoint_success_outfiles(checkpoint_payload)
            if len(resumed_outfiles) > 0:
                before = len(prepared_jobs)
                prepared_jobs = [
                    job
                    for job in prepared_jobs
                    if str(job.outfile.resolve()) not in resumed_outfiles
                ]
                skipped = before - len(prepared_jobs)
                if skipped > 0:
                    console.print(f"resuming batch: skipped {skipped} completed jobs")
        else:
            checkpoint_payload = {
                "version": BATCH_CHECKPOINT_VERSION,
                "manifest": str(manifest.resolve()),
                "results": [],
            }

    if len(prepared_jobs) == 0:
        console.print("batch complete: no pending jobs after resume filtering")
        return

    if dry_run:
        ordered = order_jobs(prepared_jobs, schedule)
        for job in ordered:
            console.print(
                "[dry-run] job "
                f"{job.index}: {job.infile} -> {job.outfile} "
                f"(cost={job.estimated_cost:.2f}, schedule={schedule})"
            )
        return

    max_workers = int(os.cpu_count() or 1) if jobs == 0 else int(jobs)
    max_workers = max(1, min(max_workers, len(prepared_jobs)))

    def runner(job: BatchJobSpec) -> None:
        run_render_pipeline(infile=job.infile, outfile=job.outfile, config=job.config)

    def on_result(result: BatchJobResult) -> None:
        if checkpoint_file is not None and checkpoint_payload is not None:
            checkpoint_payload.setdefault("results", [])
            assert isinstance(checkpoint_payload["results"], list)
            checkpoint_payload["results"].append(
                {
                    "index": int(result.index),
                    "outfile": str(result.outfile.resolve()),
                    "success": bool(result.success),
                    "attempts": int(result.attempts),
                    "duration_seconds": float(result.duration_seconds),
                    "estimated_cost": float(result.estimated_cost),
                    "error": result.error,
                }
            )
            _write_json_atomic(checkpoint_file, checkpoint_payload)
        if result.success:
            console.print(
                f"rendered job {result.index}: {result.outfile} "
                f"(attempts={result.attempts}, {result.duration_seconds:.2f}s)"
            )
        else:
            console.print(
                f"failed job {result.index}: {result.outfile} "
                f"(attempts={result.attempts}) {result.error}"
            )

    with _BatchStatusBar(total=len(prepared_jobs), label="Batch render", enabled=True) as status:
        def on_result_with_status(result: BatchJobResult) -> None:
            on_result(result)
            status.advance(detail=f"job={result.index}")

        try:
            run_parallel_batch(
                jobs=prepared_jobs,
                max_workers=max_workers,
                schedule=schedule,
                retries=retries,
                continue_on_error=continue_on_error,
                runner=runner,
                on_result=on_result_with_status,
            )
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc


@immersive_app.command("template")
def immersive_template() -> None:
    """Print an immersive scene handoff template as JSON."""
    template = {
        "scene_name": "feature_episode_01",
        "sample_rate": 48_000,
        "bed": {
            "name": "bed_main",
            "path": "renders/bed_7p1p2.wav",
            "layout": "7.1.2",
            "render_options": {"wet": 0.75, "rt60": 4.5},
        },
        "objects": [
            {
                "id": "obj_001",
                "name": "lead_vox",
                "path": "renders/obj_lead_vox.wav",
                "layout": "mono",
                "start_s": 0.0,
                "gain_db": 0.0,
                "x": 0.05,
                "y": 0.0,
                "z": 0.0,
                "render_options": {"wet": 0.45, "rt60": 2.8},
            }
        ],
        "policy": {
            "mode": "bed-safe",
            "max_bed_wet": 0.85,
            "max_object_wet": 0.6,
            "max_object_rt60": 12.0,
            "downmix_max_delta_db": 4.0,
        },
        "qc_gates": {
            "target_lufs": -18.0,
            "lufs_tolerance": 3.0,
            "max_true_peak_dbfs": -1.0,
            "max_fold_down_delta_db": 4.0,
            "min_channel_occupancy": 0.34,
            "occupancy_threshold_dbfs": -45.0,
        },
        "deliverables": {
            "adm_sidecar": True,
            "object_stem_manifest": True,
            "qa_bundle": True,
        },
    }
    typer.echo(json.dumps(template, indent=2))


@immersive_app.command("handoff")
def immersive_handoff(
    scene_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    out_dir: Path = typer.Argument(..., resolve_path=True),
    strict: bool = typer.Option(
        True,
        "--strict/--warn-only",
        help="Fail if policy/QC errors are detected.",
    ),
) -> None:
    """Generate immersive handoff sidecars and deliverable manifests."""
    try:
        payload = json.loads(scene_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid scene JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Scene file must contain a JSON object.")

    try:
        with _processing_status("Build immersive handoff package"):
            summary = generate_immersive_handoff_package(
                scene=cast(dict[str, Any], payload),
                out_dir=out_dir,
                strict=strict,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    outputs_raw = summary.get("outputs", {})
    outputs = outputs_raw if isinstance(outputs_raw, dict) else {}
    policy_raw = summary.get("policy", {})
    policy = policy_raw if isinstance(policy_raw, dict) else {}
    qa_raw = summary.get("qa_summary", {})
    qa = qa_raw if isinstance(qa_raw, dict) else {}
    validation_raw = summary.get("validation", {})
    validation = validation_raw if isinstance(validation_raw, dict) else {}
    validation_errors = validation.get("errors", [])
    validation_warnings = validation.get("warnings", [])
    failed_tracks = qa.get("failed_tracks", [])

    table = Table(title="Immersive Handoff")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("scene", str(summary.get("scene_name", "")))
    table.add_row("strict", str(bool(summary.get("strict", False))))
    table.add_row("qa_all_pass", str(bool(qa.get("all_pass", False))))
    table.add_row("policy_mode", str(policy.get("mode", "")))
    table.add_row("policy_errors", str(len(policy.get("errors", []))))
    table.add_row("policy_warnings", str(len(policy.get("warnings", []))))
    table.add_row("validation_errors", str(len(validation_errors)))
    table.add_row("validation_warnings", str(len(validation_warnings)))
    table.add_row("failed_tracks", str(len(failed_tracks)))
    table.add_row(
        "outputs",
        "\n".join(str(path) for path in outputs.values()) if len(outputs) > 0 else "(none)",
    )
    console.print(table)


@immersive_app.command("qc")
def immersive_qc(
    infile: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    layout: str = typer.Option(
        "auto",
        "--layout",
        help=(
            "Channel layout hint: auto, mono, stereo, lcr, 5.1, 7.1, "
            "7.1.2, 7.1.4, 7.2.4, 8.0, 16.0, 64.4"
        ),
    ),
    target_lufs: float = typer.Option(-18.0, "--target-lufs"),
    lufs_tolerance: float = typer.Option(3.0, "--lufs-tolerance", min=0.0),
    max_true_peak_dbfs: float = typer.Option(-1.0, "--max-true-peak-dbfs"),
    max_fold_down_delta_db: float = typer.Option(4.0, "--max-fold-down-delta-db", min=0.0),
    min_channel_occupancy: float = typer.Option(
        0.34,
        "--min-channel-occupancy",
        min=0.0,
        max=1.0,
    ),
    occupancy_threshold_dbfs: float = typer.Option(
        -45.0,
        "--occupancy-threshold-dbfs",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        resolve_path=True,
        help="Optional output path for QC JSON payload.",
    ),
    fail_on_violation: bool = typer.Option(
        False,
        "--fail-on-violation",
        help="Exit with code 2 when any QC gate fails.",
    ),
) -> None:
    """Run immersive QC gates for loudness/true-peak/fold-down/occupancy."""
    try:
        with _processing_status("Run immersive QC"):
            audio, sr = read_audio(str(infile))
            resolved_layout = validate_layout_hint(layout)
            gates = build_qc_gates(
                {
                    "target_lufs": target_lufs,
                    "lufs_tolerance": lufs_tolerance,
                    "max_true_peak_dbfs": max_true_peak_dbfs,
                    "max_fold_down_delta_db": max_fold_down_delta_db,
                    "min_channel_occupancy": min_channel_occupancy,
                    "occupancy_threshold_dbfs": occupancy_threshold_dbfs,
                }
            )
            report = evaluate_immersive_qc(
                audio=audio,
                sr=sr,
                label=infile.stem,
                layout=resolved_layout,
                gates=gates,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    metrics_raw = report.get("metrics", {})
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    passes_raw = report.get("passes", {})
    passes = passes_raw if isinstance(passes_raw, dict) else {}

    table = Table(title="Immersive QC")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("file", str(infile))
    table.add_row("layout", str(report.get("layout", "")))
    table.add_row("channels", str(report.get("channels", "")))
    table.add_row("integrated_lufs", f"{float(metrics.get('integrated_lufs', 0.0)):.2f}")
    table.add_row("true_peak_dbfs", f"{float(metrics.get('true_peak_dbfs', 0.0)):.2f}")
    table.add_row("fold_down_delta_db", f"{float(metrics.get('fold_down_delta_db', 0.0)):.2f}")
    table.add_row("channel_occupancy", f"{float(metrics.get('channel_occupancy', 0.0)):.3f}")
    table.add_row("loudness_gate", str(bool(passes.get("loudness", False))))
    table.add_row("true_peak_gate", str(bool(passes.get("true_peak", False))))
    table.add_row("fold_down_gate", str(bool(passes.get("fold_down_delta", False))))
    table.add_row("occupancy_gate", str(bool(passes.get("channel_occupancy", False))))
    table.add_row("all_pass", str(bool(report.get("pass", False))))
    console.print(table)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if fail_on_violation and (not bool(report.get("pass", False))):
        raise typer.Exit(code=2)


@immersive_queue_app.command("template")
def immersive_queue_template() -> None:
    """Print a file-backed immersive queue template as JSON."""
    template = {
        "version": IMMERSIVE_QUEUE_VERSION,
        "backend": "file",
        "jobs": [
            {
                "id": "job_0001",
                "infile": "input.wav",
                "outfile": "renders/output.wav",
                "max_retries": 1,
                "options": {
                    "engine": "algo",
                    "rt60": 3.0,
                    "wet": 0.7,
                    "dry": 0.3,
                    "progress": False,
                },
            }
        ],
    }
    typer.echo(json.dumps(template, indent=2))


@immersive_queue_app.command("status")
def immersive_queue_status(
    queue_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
) -> None:
    """Show file-queue state summary."""
    try:
        status = summarize_file_queue(queue_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Immersive Queue Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("queue", str(status.get("queue_path", "")))
    table.add_row("state_root", str(status.get("state_root", "")))
    table.add_row("total_jobs", str(status.get("total_jobs", 0)))
    table.add_row("success_jobs", str(status.get("success_jobs", 0)))
    table.add_row("failed_jobs", str(status.get("failed_jobs", 0)))
    table.add_row("claimed_jobs", str(status.get("claimed_jobs", 0)))
    table.add_row("pending_jobs", str(status.get("pending_jobs", 0)))
    console.print(table)


@immersive_queue_app.command("worker")
def immersive_queue_worker(
    queue_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    worker_id: str | None = typer.Option(
        None,
        "--worker-id",
        help="Worker identifier. Defaults to host PID-based value.",
    ),
    heartbeat_dir: Path = typer.Option(
        Path(".verbx_queue_heartbeats"),
        "--heartbeat-dir",
        resolve_path=True,
        help="Directory for per-worker heartbeat JSON files.",
    ),
    poll_ms: int = typer.Option(800, "--poll-ms", min=50),
    max_jobs: int = typer.Option(0, "--max-jobs", min=0, help="0 = run until queue drain"),
    stale_claim_seconds: float = typer.Option(120.0, "--stale-claim-seconds", min=1.0),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast"),
    fail_if_any_failed: bool = typer.Option(
        False,
        "--fail-if-any-failed",
        help="Exit with code 2 if any queue jobs end in failed state.",
    ),
) -> None:
    """Run one distributed queue worker for immersive batch execution."""
    resolved_worker_id = (
        worker_id
        if worker_id is not None and str(worker_id).strip() != ""
        else f"worker_{os.getpid()}"
    )

    config = QueueWorkerConfig(
        worker_id=resolved_worker_id,
        heartbeat_dir=heartbeat_dir,
        poll_ms=poll_ms,
        max_jobs=max_jobs,
        stale_claim_seconds=stale_claim_seconds,
        continue_on_error=continue_on_error,
    )

    def runner(job: Any) -> None:
        if not isinstance(job, dict):
            raise ValueError("queue runner received invalid job payload")
        infile = Path(str(job.get("infile", "")))
        outfile = Path(str(job.get("outfile", "")))
        options_raw = job.get("options")
        options = options_raw if isinstance(options_raw, dict) else {}
        render_config = _render_config_from_options(options)
        _validate_batch_job_paths(infile, outfile, 1)
        run_render_pipeline(infile=infile, outfile=outfile, config=render_config)

    try:
        with _processing_status("Run immersive queue worker"):
            summary = run_file_queue_worker(
                queue_path=queue_file,
                runner=runner,
                config=config,
            )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Immersive Queue Worker")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("worker_id", str(summary.get("worker_id", "")))
    table.add_row("queue_path", str(summary.get("queue_path", "")))
    table.add_row("state_root", str(summary.get("state_root", "")))
    table.add_row("processed", str(summary.get("processed", 0)))
    table.add_row("success", str(summary.get("success", 0)))
    table.add_row("retried", str(summary.get("retried", 0)))
    table.add_row("failed", str(summary.get("failed", 0)))
    table.add_row("last_error", str(summary.get("last_error", "")))
    console.print(table)

    if fail_if_any_failed and int(summary.get("failed", 0)) > 0:
        raise typer.Exit(code=2)


def _render_config_from_options(options: dict[str, Any]) -> RenderConfig:
    """Build ``RenderConfig`` from manifest options with safe field filtering."""
    fields = RenderConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in options.items() if key in fields}

    pre_delay_note = filtered.get("pre_delay_note")
    bpm = filtered.get("bpm")
    if isinstance(pre_delay_note, str):
        fallback_ms = float(filtered.get("pre_delay_ms", 20.0))
        resolved_bpm = float(bpm) if isinstance(bpm, (float, int)) else None
        filtered["pre_delay_ms"] = parse_pre_delay_ms(pre_delay_note, resolved_bpm, fallback_ms)

    return RenderConfig(**filtered)


def _load_batch_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint payload with graceful fallback when missing/invalid."""
    if not path.exists():
        return {"version": BATCH_CHECKPOINT_VERSION, "results": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": BATCH_CHECKPOINT_VERSION, "results": []}
    if not isinstance(payload, dict):
        return {"version": BATCH_CHECKPOINT_VERSION, "results": []}
    results = payload.get("results")
    if not isinstance(results, list):
        payload["results"] = []
    return payload


def _checkpoint_success_outfiles(payload: dict[str, Any]) -> set[str]:
    """Return canonical outfile paths marked successful in checkpoint payload."""
    completed: set[str] = set()
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return completed
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("success", False)):
            continue
        outfile = row.get("outfile")
        if not isinstance(outfile, str):
            continue
        out_path = Path(outfile)
        if out_path.exists():
            completed.add(str(out_path.resolve()))
    return completed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row, sort_keys=True)}\n")
    tmp.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write tabular rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    if not columns:
        columns = ["note"]
    if "metrics_error" in columns:
        columns.remove("metrics_error")
        columns.append("metrics_error")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically write UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _build_augmentation_metrics_rows(
    *,
    records: list[dict[str, Any]],
    include_loudness: bool,
) -> list[dict[str, Any]]:
    """Extract per-output metrics rows from augmentation records."""
    analyzer = AudioAnalyzer()
    rows: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("success", False)):
            continue
        outfile = Path(str(record.get("outfile", "")))
        base: dict[str, Any] = {
            "augment_index": int(record.get("augment_index", 0)),
            "source_id": str(record.get("source_id", "")),
            "split": str(record.get("split", "")),
            "label": str(record.get("label", "")),
            "archetype": str(record.get("archetype", "")),
            "outfile": str(outfile),
        }
        try:
            audio, sr = read_audio(str(outfile))
            metrics = analyzer.analyze(
                audio,
                sr,
                include_loudness=include_loudness,
                include_edr=False,
            )
            base["sample_rate"] = int(sr)
            base["samples"] = int(audio.shape[0])
            base["channels"] = int(audio.shape[1])
            for key in sorted(metrics):
                base[key] = float(metrics[key])
            base["metrics_error"] = ""
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            base["metrics_error"] = str(exc)
        rows.append(base)
    return rows


def _build_augmentation_dataset_card(
    *,
    build: AugmentationBuild,
    summary: dict[str, Any],
    manifest_path: Path,
    summary_path: Path,
    records: list[dict[str, Any]],
) -> str:
    """Generate a compact markdown dataset card for AI/data workflows."""
    split_counts = cast(dict[str, Any], summary.get("split_counts", {}))
    label_counts = cast(dict[str, Any], summary.get("label_counts", {}))
    archetype_counts = cast(dict[str, Any], summary.get("archetype_counts", {}))
    lines = [
        "# Dataset Card",
        "",
        "## Identity",
        f"- dataset_name: `{build.dataset_name}`",
        f"- profile: `{build.profile}`",
        f"- seed: `{build.seed}`",
        f"- generated_utc: `{datetime.now(UTC).isoformat()}`",
        f"- manifest_path: `{manifest_path.resolve()}`",
        f"- summary_json: `{summary_path.resolve()}`",
        f"- qa_bundle_json: `{summary.get('qa_bundle', '(none)')}`",
        f"- output_root: `{build.output_root.resolve()}`",
        "",
        "## Render Summary",
        f"- planned: `{summary.get('planned', 0)}`",
        f"- success: `{summary.get('success', 0)}`",
        f"- failed: `{summary.get('failed', 0)}`",
        f"- variants_per_input: `{build.variants_per_input}`",
        f"- provenance_hash: `{summary.get('provenance_hash', '(disabled)')}`",
        "",
        "## Split Distribution",
    ]
    if split_counts:
        for split in sorted(split_counts):
            lines.append(f"- {split}: `{split_counts[split]}`")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Label Distribution",
        ]
    )
    if label_counts:
        for label in sorted(label_counts):
            lines.append(f"- {label}: `{label_counts[label]}`")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Archetype Distribution",
        ]
    )
    if archetype_counts:
        for archetype in sorted(archetype_counts):
            lines.append(f"- {archetype}: `{archetype_counts[archetype]}`")
    else:
        lines.append("- (none)")

    rt60_values: list[float] = []
    wet_values: list[float] = []
    damping_values: list[float] = []
    fdn_lines_values: list[int] = []
    fdn_matrices: set[str] = set()
    for row in records:
        config = row.get("render_config", {})
        if not isinstance(config, dict):
            continue
        rt60 = config.get("rt60")
        wet = config.get("wet")
        damping = config.get("damping")
        fdn_lines = config.get("fdn_lines")
        fdn_matrix = config.get("fdn_matrix")
        if isinstance(rt60, (int, float)):
            rt60_values.append(float(rt60))
        if isinstance(wet, (int, float)):
            wet_values.append(float(wet))
        if isinstance(damping, (int, float)):
            damping_values.append(float(damping))
        if isinstance(fdn_lines, (int, float)):
            fdn_lines_values.append(int(fdn_lines))
        if isinstance(fdn_matrix, str) and fdn_matrix != "":
            fdn_matrices.add(fdn_matrix)

    lines.extend(
        [
            "",
            "## Parameter Envelope",
        ]
    )
    if rt60_values:
        lines.append(f"- rt60_s_min_max: `{min(rt60_values):.3f} .. {max(rt60_values):.3f}`")
    if wet_values:
        lines.append(f"- wet_min_max: `{min(wet_values):.3f} .. {max(wet_values):.3f}`")
    if damping_values:
        lines.append(
            f"- damping_min_max: `{min(damping_values):.3f} .. {max(damping_values):.3f}`"
        )
    if fdn_lines_values:
        lines.append(
            f"- fdn_lines_min_max: `{min(fdn_lines_values)} .. {max(fdn_lines_values)}`"
        )
    if fdn_matrices:
        lines.append(f"- fdn_matrices: `{', '.join(sorted(fdn_matrices))}`")
    if len(lines) > 0 and lines[-1] == "## Parameter Envelope":
        lines.append("- (no parameter metadata available)")

    lines.extend(
        [
            "",
            "## Notes",
            "- This card is auto-generated by `verbx batch augment`.",
            (
                "- Keep this file with `augmentation_manifest.jsonl` and summary JSON "
                "for reproducibility."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _print_augmentation_dry_run(
    *,
    build: AugmentationBuild,
    schedule: BatchSchedulePolicy,
) -> None:
    """Print dry-run summary for augmentation plan expansion."""
    table = Table(title="Batch Augment Dry-Run")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("dataset_name", build.dataset_name)
    table.add_row("profile", build.profile)
    table.add_row("seed", str(build.seed))
    table.add_row("variants_per_input", str(build.variants_per_input))
    table.add_row("write_analysis", str(build.write_analysis))
    table.add_row("copy_dry", str(build.copy_dry))
    table.add_row("schedule", str(schedule))
    table.add_row("plans", str(len(build.plans)))
    table.add_row("output_root", str(build.output_root))
    console.print(table)

    ordered = order_jobs(
        [
            BatchJobSpec(
                index=plan.index,
                infile=plan.infile,
                outfile=plan.outfile,
                config=plan.config,
                estimated_cost=estimate_job_cost(plan.infile, plan.config),
            )
            for plan in build.plans
        ],
        schedule,
    )
    for job in ordered:
        match_plan = build.plans[job.index - 1]
        console.print(
            "[dry-run] augment "
            f"{job.index}: {job.infile} -> {job.outfile} "
            f"(profile={match_plan.profile}, archetype={match_plan.archetype}, "
            f"split={match_plan.split}, label={match_plan.label}, seed={match_plan.seed})"
        )


def _copy_augmentation_dry_sources(*, build: AugmentationBuild) -> list[Path]:
    """Copy clean source files to output tree (optional paired dataset mode)."""
    copied: list[Path] = []
    seen: set[str] = set()
    for plan in build.plans:
        target = plan.dry_copy_outfile
        if target is None:
            continue
        key = str(target.resolve())
        if key in seen:
            continue
        seen.add(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.infile, target)
        copied.append(target)
    return copied


def _prepare_augmentation_output_dirs(*, build: AugmentationBuild) -> None:
    """Ensure batch augment output directories exist before parallel rendering."""
    for plan in build.plans:
        plan.outfile.parent.mkdir(parents=True, exist_ok=True)
        if plan.config.analysis_out is not None:
            Path(plan.config.analysis_out).parent.mkdir(parents=True, exist_ok=True)


def _build_augmentation_records(
    *,
    build: AugmentationBuild,
    results_by_index: dict[int, BatchJobResult],
) -> list[dict[str, Any]]:
    """Build machine-friendly per-variant metadata rows."""
    records: list[dict[str, Any]] = []
    for plan in build.plans:
        result = results_by_index.get(plan.index)
        success = bool(result.success) if result is not None else False
        duration = float(result.duration_seconds) if result is not None else 0.0
        attempts = int(result.attempts) if result is not None else 0
        error = None if result is None else result.error
        outfile_rel = (
            str(plan.outfile.relative_to(build.output_root))
            if plan.outfile.is_relative_to(build.output_root)
            else str(plan.outfile)
        )
        record = {
            "augment_index": int(plan.index),
            "source_index": int(plan.source_index),
            "source_id": plan.source_id,
            "split": plan.split,
            "label": plan.label,
            "tags": list(plan.tags),
            "infile": str(plan.infile.resolve()),
            "outfile": str(plan.outfile.resolve()),
            "outfile_relative": outfile_rel,
            "profile": plan.profile,
            "archetype": plan.archetype,
            "variant_index": int(plan.variant_index),
            "seed": int(plan.seed),
            "render_config": render_config_snapshot(plan.config),
            "success": success,
            "attempts": attempts,
            "duration_seconds": duration,
            "error": error,
            "source_metadata": dict(plan.source_metadata),
        }
        if plan.dry_copy_outfile is not None:
            dry_rel = (
                str(plan.dry_copy_outfile.relative_to(build.output_root))
                if plan.dry_copy_outfile.is_relative_to(build.output_root)
                else str(plan.dry_copy_outfile)
            )
            record["dry_outfile"] = str(plan.dry_copy_outfile.resolve())
            record["dry_outfile_relative"] = dry_rel
        records.append(record)
    return records


def _build_augmentation_summary(
    *,
    build: AugmentationBuild,
    records: list[dict[str, Any]],
    manifest_path: Path,
    jsonl_path: Path,
    schedule: BatchSchedulePolicy,
    jobs: int,
) -> dict[str, Any]:
    """Build summary payload for augmentation run."""
    total = len(records)
    success = int(sum(1 for row in records if bool(row.get("success", False))))
    failed = int(total - success)
    split_counts: dict[str, int] = {}
    split_source_ids: dict[str, set[str]] = {}
    split_label_counts: dict[str, dict[str, int]] = {}
    label_counts: dict[str, int] = {}
    archetype_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for row in records:
        split = str(row.get("split", ""))
        label = str(row.get("label", ""))
        source_id = str(row.get("source_id", ""))
        archetype = str(row.get("archetype", ""))
        split_counts[split] = int(split_counts.get(split, 0) + 1)
        if split != "" and label != "":
            split_map = split_label_counts.setdefault(split, {})
            split_map[label] = int(split_map.get(label, 0) + 1)
        if split != "" and source_id != "":
            split_source_ids.setdefault(split, set()).add(source_id)
        if label != "":
            label_counts[label] = int(label_counts.get(label, 0) + 1)
        if archetype != "":
            archetype_counts[archetype] = int(archetype_counts.get(archetype, 0) + 1)
        tags = row.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                token = str(tag)
                if token == "":
                    continue
                tag_counts[token] = int(tag_counts.get(token, 0) + 1)
    return {
        "version": AUGMENT_SUMMARY_VERSION,
        "mode": "batch-augment",
        "dataset_name": build.dataset_name,
        "profile": build.profile,
        "seed": int(build.seed),
        "variants_per_input": int(build.variants_per_input),
        "write_analysis": bool(build.write_analysis),
        "copy_dry": bool(build.copy_dry),
        "manifest_path": str(manifest_path.resolve()),
        "output_root": str(build.output_root.resolve()),
        "metadata_jsonl": str(jsonl_path.resolve()),
        "schedule": str(schedule),
        "jobs": int(jobs),
        "planned": int(total),
        "success": int(success),
        "failed": int(failed),
        "unique_sources": len({str(row.get("source_id", "")) for row in records}),
        "source_split_counts": {
            split: len(source_ids) for split, source_ids in sorted(split_source_ids.items())
        },
        "split_counts": split_counts,
        "split_label_counts": {
            split: dict(sorted(label_map.items()))
            for split, label_map in sorted(split_label_counts.items())
        },
        "label_counts": label_counts,
        "archetype_counts": archetype_counts,
        "tag_counts": tag_counts,
    }


def _load_optional_summary_payload(path: Path) -> dict[str, Any]:
    """Load optional JSON summary payload for regeneration delta reporting."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Invalid baseline summary JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--baseline-summary must contain a JSON object.")
    return cast(dict[str, Any], payload)


def _build_augmentation_qa_bundle(
    *,
    build: AugmentationBuild,
    records: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build augmentation QA bundle with split-quality and class-balance deltas."""
    split_label_counts = _extract_split_label_counts(records)
    split_quality = _summarize_split_quality(metrics_rows)
    successful = int(sum(1 for row in records if bool(row.get("success", False))))
    failed = int(len(records) - successful)

    baseline_split_label_counts = (
        _extract_baseline_split_label_counts(baseline_summary)
        if baseline_summary is not None
        else {}
    )
    class_balance_delta = (
        _compute_class_balance_delta(
            current=split_label_counts,
            baseline=baseline_split_label_counts,
        )
        if baseline_summary is not None
        else None
    )

    return {
        "version": AUGMENT_QA_BUNDLE_VERSION,
        "mode": "batch-augment-qa",
        "dataset_name": build.dataset_name,
        "profile": build.profile,
        "seed": int(build.seed),
        "planned": len(records),
        "success": int(successful),
        "failed": int(failed),
        "split_label_counts": split_label_counts,
        "split_quality": split_quality,
        "baseline_present": baseline_summary is not None,
        "class_balance_delta": class_balance_delta,
    }


def _extract_split_label_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Extract split->label->count table from augmentation metadata rows."""
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        split = str(row.get("split", ""))
        label = str(row.get("label", ""))
        if split == "" or label == "":
            continue
        split_map = counts.setdefault(split, {})
        split_map[label] = int(split_map.get(label, 0) + 1)
    return {
        split: dict(sorted(label_map.items()))
        for split, label_map in sorted(counts.items())
    }


def _extract_baseline_split_label_counts(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Extract split-label table from baseline summary or QA bundle payload."""
    direct = payload.get("split_label_counts")
    if isinstance(direct, dict):
        return _normalize_split_label_counts(direct)
    qa_raw = payload.get("class_balance")
    if isinstance(qa_raw, dict):
        nested = qa_raw.get("split_label_counts")
        if isinstance(nested, dict):
            return _normalize_split_label_counts(nested)
    return {}


def _normalize_split_label_counts(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    normalized: dict[str, dict[str, int]] = {}
    for split, labels_raw in payload.items():
        if not isinstance(labels_raw, dict):
            continue
        labels: dict[str, int] = {}
        for label, count_raw in labels_raw.items():
            try:
                labels[str(label)] = int(count_raw)
            except (TypeError, ValueError):
                continue
        if labels:
            normalized[str(split)] = dict(sorted(labels.items()))
    return dict(sorted(normalized.items()))


def _compute_class_balance_delta(
    *,
    current: dict[str, dict[str, int]],
    baseline: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compute split-level and global class-balance deltas against baseline."""
    split_delta: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in sorted(set(current) | set(baseline)):
        cur_labels = current.get(split, {})
        base_labels = baseline.get(split, {})
        cur_total = int(sum(cur_labels.values()))
        base_total = int(sum(base_labels.values()))
        label_delta: dict[str, dict[str, float | int]] = {}
        for label in sorted(set(cur_labels) | set(base_labels)):
            cur_count = int(cur_labels.get(label, 0))
            base_count = int(base_labels.get(label, 0))
            cur_ratio = float(cur_count / cur_total) if cur_total > 0 else 0.0
            base_ratio = float(base_count / base_total) if base_total > 0 else 0.0
            label_delta[label] = {
                "current_count": cur_count,
                "baseline_count": base_count,
                "count_delta": int(cur_count - base_count),
                "current_ratio": cur_ratio,
                "baseline_ratio": base_ratio,
                "ratio_delta": float(cur_ratio - base_ratio),
            }
        split_delta[split] = label_delta

    current_global = _collapse_split_label_counts(current)
    baseline_global = _collapse_split_label_counts(baseline)
    global_total = int(sum(current_global.values()))
    baseline_total = int(sum(baseline_global.values()))
    global_delta: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(current_global) | set(baseline_global)):
        cur_count = int(current_global.get(label, 0))
        base_count = int(baseline_global.get(label, 0))
        cur_ratio = float(cur_count / global_total) if global_total > 0 else 0.0
        base_ratio = float(base_count / baseline_total) if baseline_total > 0 else 0.0
        global_delta[label] = {
            "current_count": cur_count,
            "baseline_count": base_count,
            "count_delta": int(cur_count - base_count),
            "current_ratio": cur_ratio,
            "baseline_ratio": base_ratio,
            "ratio_delta": float(cur_ratio - base_ratio),
        }
    return {
        "split_label_delta": split_delta,
        "global_label_delta": global_delta,
    }


def _collapse_split_label_counts(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for label_map in counts.values():
        for label, count in label_map.items():
            merged[label] = int(merged.get(label, 0) + int(count))
    return merged


def _summarize_split_quality(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize numeric quality metrics per split from augmentation metrics rows."""
    fields = (
        "rt60_estimate_seconds",
        "early_late_ratio_db",
        "stereo_width",
        "stereo_coherence",
        "spectral_centroid",
        "spectral_flatness",
        "dynamic_range",
        "lufs_integrated",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        split = str(row.get("split", ""))
        if split == "":
            continue
        grouped.setdefault(split, []).append(row)
    summary: dict[str, Any] = {}
    for split, rows in sorted(grouped.items()):
        split_summary: dict[str, Any] = {}
        for field in fields:
            stats = _summarize_metrics_field(rows, field)
            if stats is not None:
                split_summary[field] = stats
        errors = int(sum(1 for row in rows if str(row.get("metrics_error", "")).strip() != ""))
        split_summary["rows"] = len(rows)
        split_summary["metrics_errors"] = int(errors)
        summary[split] = split_summary
    return summary


def _summarize_metrics_field(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float] | None:
    values: list[float] = []
    for row in rows:
        raw = row.get(field)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) == 0:
        return None
    vec = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "mean": float(np.mean(vec)),
    }


def _audio_file_signature(path: Path) -> dict[str, Any]:
    """Return deterministic file signature + audio metadata for reproducibility bundles."""
    info = sf.info(str(path))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _build_render_repro_bundle(
    *,
    infile: Path,
    outfile: Path,
    report: dict[str, Any],
    config: RenderConfig,
    preset_name: str | None,
) -> dict[str, Any]:
    """Build render reproducibility/support bundle payload."""
    input_signature = _audio_file_signature(infile)
    output_signature = _audio_file_signature(outfile)
    analysis_entry: dict[str, Any] | None = None
    analysis_path_raw = report.get("analysis_path")
    if isinstance(analysis_path_raw, str) and analysis_path_raw.strip() != "":
        analysis_path = Path(analysis_path_raw)
        if analysis_path.exists():
            analysis_entry = {
                "path": str(analysis_path.resolve()),
                "sha256": _sha256_file(analysis_path),
                "bytes": int(analysis_path.stat().st_size),
            }

    signature_payload = {
        "verbx_version": __version__,
        "engine": report.get("engine"),
        "config": report.get("config"),
        "effective": report.get("effective"),
        "input_sha256": input_signature["sha256"],
        "output_sha256": output_signature["sha256"],
        "analysis_sha256": None if analysis_entry is None else analysis_entry.get("sha256"),
    }
    run_signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    payload: dict[str, Any] = {
        "schema": "render-repro-bundle-v1",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "verbx_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "run_signature": run_signature,
        "engine": str(report.get("engine", "")),
        "preset": preset_name,
        "input": input_signature,
        "output": output_signature,
        "analysis": analysis_entry,
        "effective": report.get("effective"),
        "config": asdict(config),
    }
    return payload


def _build_render_failure_report(
    *,
    infile: Path,
    outfile: Path,
    config: RenderConfig,
    preset_name: str | None,
    error: Exception,
) -> dict[str, Any]:
    """Build structured failure report payload for render support workflows."""
    diagnostics = _collect_runtime_diagnostics()
    return {
        "schema": "render-failure-report-v1",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "verbx_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "error_type": type(error).__name__,
        "error": str(error),
        "infile": str(infile.resolve()),
        "outfile": str(outfile.resolve()),
        "preset": preset_name,
        "config": asdict(config),
        "diagnostics": diagnostics,
    }


def _compute_augmentation_provenance_hash(
    *,
    build: AugmentationBuild,
    manifest: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic augmentation provenance hash for registry workflows."""
    manifest_payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            manifest_payload = cast(dict[str, Any], parsed)
    except (OSError, json.JSONDecodeError):
        manifest_payload = None
    if manifest_payload is None:
        manifest_canonical = manifest.read_bytes()
    else:
        manifest_canonical = json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    source_paths = sorted({str(plan.infile.resolve()) for plan in build.plans})
    source_signatures: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    digest.update(manifest_canonical)
    for source in source_paths:
        path = Path(source)
        info = sf.info(str(path))
        signature = {
            "path": source,
            "sha256": _sha256_file(path),
            "frames": int(info.frames),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "format": str(info.format),
            "subtype": str(info.subtype),
        }
        source_signatures.append(signature)
        digest.update(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(str(len(records)).encode("utf-8"))

    return {
        "schema": "augmentation-provenance-v1",
        "sha256": digest.hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_canonical).hexdigest(),
        "source_count": len(source_signatures),
        "sources": source_signatures,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


def _print_augmentation_summary_table(
    summary: dict[str, Any],
    *,
    summary_path: Path,
) -> None:
    """Print concise augmentation run summary."""
    table = Table(title="Batch Augment Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for key in (
        "dataset_name",
        "profile",
        "seed",
        "variants_per_input",
        "planned",
        "success",
        "failed",
        "unique_sources",
        "output_root",
        "metadata_jsonl",
    ):
        table.add_row(key, str(summary.get(key, "")))
    metrics_csv = summary.get("metrics_csv")
    if metrics_csv:
        table.add_row("metrics_csv", str(metrics_csv))
    qa_bundle = summary.get("qa_bundle")
    if qa_bundle:
        table.add_row("qa_bundle", str(qa_bundle))
    dataset_card = summary.get("dataset_card")
    if dataset_card:
        table.add_row("dataset_card", str(dataset_card))
    provenance_hash = summary.get("provenance_hash")
    if provenance_hash:
        table.add_row("provenance_hash", str(provenance_hash))
    table.add_row("summary_json", str(summary_path.resolve()))
    console.print(table)


class _ScoredFitCandidate:
    """Internal transport object for IR-fit ranking results."""

    def __init__(
        self,
        *,
        candidate: IRFitCandidate,
        score: IRFitScore,
        detail_metrics: dict[str, float],
        sr: int,
        meta: dict[str, Any],
        cache_path: Path,
    ) -> None:
        self.candidate = candidate
        self.score = score
        self.detail_metrics = detail_metrics
        self.sr = int(sr)
        self.meta = meta
        self.cache_path = cache_path


def _score_fit_candidates(
    *,
    candidates: list[IRFitCandidate],
    target: IRFitTarget,
    cache_dir: Path,
    fit_workers: int,
    show_progress: bool = True,
) -> list[_ScoredFitCandidate]:
    """Evaluate IR-fit candidates serially or in parallel."""
    worker_count = int(os.cpu_count() or 1) if fit_workers == 0 else fit_workers
    worker_count = max(1, min(worker_count, len(candidates)))

    def evaluate(candidate: IRFitCandidate) -> _ScoredFitCandidate:
        audio, sr, meta, cache_path, _ = generate_or_load_cached_ir(
            candidate.config,
            cache_dir=cache_dir,
        )
        score, detail_metrics = score_ir_candidate(ir_audio=audio, sr=sr, target=target)
        return _ScoredFitCandidate(
            candidate=candidate,
            score=score,
            detail_metrics=detail_metrics,
            sr=sr,
            meta=meta,
            cache_path=cache_path,
        )

    if worker_count == 1:
        scored_serial: list[_ScoredFitCandidate] = []
        with _BatchStatusBar(
            total=len(candidates),
            label="IR fit candidates",
            enabled=show_progress,
        ) as status:
            for candidate in candidates:
                scored_serial.append(evaluate(candidate))
                status.advance(detail="serial")
        return scored_serial

    scored: list[_ScoredFitCandidate] = []
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="verbx-fit") as pool:
        with _BatchStatusBar(
            total=len(candidates),
            label="IR fit candidates",
            enabled=show_progress,
        ) as status:
            futures: list[Future[_ScoredFitCandidate]] = [
                pool.submit(evaluate, candidate) for candidate in candidates
            ]
            for fut in as_completed(futures):
                scored.append(fut.result())
                status.advance(detail=f"workers={worker_count}")
    return scored


def _format_metric_value(value: Any) -> str:
    """Format scalar metric values for console tables."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _print_feature_table(title: str, metrics: dict[str, Any]) -> None:
    """Print a compact feature/statistics table."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    preferred_keys = (
        "duration",
        "samples",
        "channels",
        "rms",
        "rms_dbfs",
        "peak",
        "peak_dbfs",
        "sample_peak_dbfs",
        "true_peak_dbfs",
        "integrated_lufs",
        "lra",
        "dynamic_range",
        "crest_factor",
        "dc_offset",
        "sample_min",
        "sample_max",
        "silence_ratio",
        "transient_density",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "spectral_flux",
        "spectral_slope",
        "stereo_correlation",
        "stereo_width",
    )
    printed: set[str] = set()
    for key in preferred_keys:
        if key in metrics:
            table.add_row(key, _format_metric_value(metrics[key]))
            printed.add(key)

    for key in sorted(k for k in metrics.keys() if k.startswith("channel_") and k.endswith("_rms")):
        table.add_row(key, _format_metric_value(metrics[key]))
        printed.add(key)

    for key in sorted(k for k in metrics.keys() if k not in printed):
        table.add_row(key, _format_metric_value(metrics[key]))

    console.print(table)


def _print_render_summary(
    report: dict[str, Any],
    *,
    verbosity: int = 1,
    preset_name: str | None = None,
) -> None:
    """Print render summary and output feature/statistics tables."""
    table = Table(title="Render Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    if preset_name is not None and str(preset_name).strip() != "":
        table.add_row("preset", str(preset_name))
    table.add_row("requested_engine", str(report.get("effective", {}).get("engine_requested", "")))
    table.add_row("engine", str(report.get("engine", "unknown")))
    table.add_row("requested_device", str(report.get("effective", {}).get("device_requested", "")))
    table.add_row("device", str(report.get("effective", {}).get("device_resolved", "")))
    table.add_row(
        "device_platform",
        str(report.get("effective", {}).get("device_platform_resolved", "")),
    )
    table.add_row("compute_backend", str(report.get("effective", {}).get("compute_backend", "")))
    table.add_row("ir_used", str(report.get("effective", {}).get("ir_used")))
    table.add_row(
        "self_convolve",
        str(report.get("effective", {}).get("self_convolve", False)),
    )
    table.add_row("sample_rate", str(report.get("sample_rate", "")))
    table.add_row("channels", str(report.get("channels", "")))
    table.add_row("input_samples", str(report.get("input_samples", "")))
    table.add_row("output_samples", str(report.get("output_samples", "")))
    table.add_row(
        "tail_padding_s",
        str(report.get("effective", {}).get("tail_padding_seconds", "")),
    )
    table.add_row("beast_mode", str(report.get("effective", {}).get("beast_mode", 1)))
    table.add_row("out_subtype", str(report.get("effective", {}).get("output_subtype", "")))
    table.add_row("output_peak_norm", str(report.get("effective", {}).get("output_peak_norm", "")))
    table.add_row(
        "output_peak_target_dbfs",
        str(report.get("effective", {}).get("output_peak_target_dbfs", "")),
    )
    table.add_row("mod_target", str(report.get("config", {}).get("mod_target", "none")))
    mod_sources = report.get("config", {}).get("mod_sources", ())
    mod_routes = report.get("config", {}).get("mod_routes", ())
    if isinstance(mod_sources, list):
        table.add_row("mod_sources", str(len(mod_sources)))
    elif isinstance(mod_sources, tuple):
        table.add_row("mod_sources", str(len(mod_sources)))
    else:
        table.add_row("mod_sources", "0")
    if isinstance(mod_routes, list):
        table.add_row("mod_routes", str(len(mod_routes)))
    elif isinstance(mod_routes, tuple):
        table.add_row("mod_routes", str(len(mod_routes)))
    else:
        table.add_row("mod_routes", "0")
    table.add_row("streaming_mode", str(report.get("effective", {}).get("streaming_mode", "")))
    automation_payload = report.get("effective", {}).get("automation")
    if isinstance(automation_payload, dict):
        targets = automation_payload.get("targets", [])
        if isinstance(targets, list):
            table.add_row("automation_targets", ",".join(str(t) for t in targets))
        table.add_row("automation_mode", str(automation_payload.get("mode", "")))
        feature_payload = automation_payload.get("feature_vector")
        if isinstance(feature_payload, dict):
            sources = feature_payload.get("sources", [])
            if isinstance(sources, list):
                table.add_row("feature_vector_sources", ",".join(str(s) for s in sources))
            table.add_row("feature_vector_signature", str(feature_payload.get("signature", "")))
            mapping = feature_payload.get("mapping")
            if isinstance(mapping, dict):
                table.add_row("feature_map_signature", str(mapping.get("signature", "")))
                mapping_targets = mapping.get("targets", [])
                if isinstance(mapping_targets, list):
                    table.add_row(
                        "feature_map_targets",
                        ",".join(str(target) for target in mapping_targets),
                    )
            guide_alignment = feature_payload.get("guide_alignment")
            if isinstance(guide_alignment, dict):
                table.add_row("feature_guide_path", str(guide_alignment.get("path", "")))
                table.add_row("feature_guide_policy", str(guide_alignment.get("policy", "")))
                table.add_row(
                    "feature_guide_sr_action",
                    str(guide_alignment.get("sample_rate_action", "")),
                )
                table.add_row(
                    "feature_guide_ch_action",
                    str(guide_alignment.get("channel_action", "")),
                )
                table.add_row(
                    "feature_guide_dur_action",
                    str(guide_alignment.get("duration_action", "")),
                )
    config_report = report.get("config", {})
    if isinstance(config_report, dict):
        for key in (
            "rt60",
            "pre_delay_ms",
            "beast_mode",
            "wet",
            "dry",
            "repeat",
            "ir_matrix_layout",
            "self_convolve",
            "damping",
            "width",
            "allpass_stages",
            "allpass_gain",
            "fdn_lines",
            "block_size",
        ):
            if key in config_report:
                table.add_row(key, str(config_report[key]))
        for key in ("allpass_gains", "allpass_delays_ms", "comb_delays_ms"):
            if key in config_report and isinstance(config_report[key], (list, tuple)):
                table.add_row(f"{key}_count", str(len(config_report[key])))
    table.add_row("analysis_json", str(report.get("analysis_path", "")))
    if "repro_bundle_path" in report:
        table.add_row("repro_bundle_json", str(report.get("repro_bundle_path")))
    if "frames_path" in report:
        table.add_row("frames_csv", str(report.get("frames_path")))
    if "feature_vector_trace_path" in report:
        table.add_row("feature_vector_trace_csv", str(report.get("feature_vector_trace_path")))
    if "ir_runtime" in report:
        runtime = report["ir_runtime"]
        if isinstance(runtime, dict):
            table.add_row("ir_runtime_path", str(runtime.get("ir_path", "")))
            table.add_row("ir_cache_hit", str(runtime.get("cache_hit", False)))
    console.print(table)

    if verbosity <= 0:
        return

    output_metrics = report.get("output")
    if isinstance(output_metrics, dict):
        _print_feature_table("Output Audio Features and Statistics", output_metrics)

    if verbosity >= 2:
        input_metrics = report.get("input")
        if isinstance(input_metrics, dict):
            _print_feature_table("Input Audio Features and Statistics", input_metrics)


def _build_lucky_config(
    base: RenderConfig,
    rng: np.random.Generator,
    input_duration_seconds: float,
) -> RenderConfig:
    """Build one randomized high-intensity render config for ``--lucky`` mode."""
    cfg = RenderConfig(**asdict(base))

    cfg.engine = cast(EngineName, rng.choice(np.array(["algo", "conv", "auto"], dtype=object)))
    cfg.beast_mode = int(rng.integers(1, 7))
    cfg.rt60 = float(rng.uniform(0.8, 12.0))
    cfg.pre_delay_ms = float(rng.uniform(0.0, 500.0))
    cfg.damping = float(rng.uniform(0.0, 1.0))
    cfg.width = float(rng.uniform(0.0, 2.0))
    cfg.mod_depth_ms = float(rng.uniform(0.0, 25.0))
    cfg.mod_rate_hz = float(rng.uniform(0.02, 2.0))
    cfg.wet = float(rng.uniform(0.55, 1.0))
    cfg.dry = float(rng.uniform(0.0, 0.6))
    cfg.repeat = int(rng.integers(1, 4))
    cfg.partition_size = int(rng.choice(np.array([4096, 8192, 16384, 32768], dtype=np.int32)))
    cfg.block_size = int(rng.choice(np.array([1024, 2048, 4096, 8192], dtype=np.int32)))
    cfg.progress = False
    cfg.frames_out = None
    cfg.analysis_out = None
    cfg.automation_trace_out = None
    cfg.feature_vector_trace_out = None

    cfg.freeze = bool(input_duration_seconds > 1.5 and rng.random() < 0.33)
    if cfg.freeze:
        max_start = max(0.0, input_duration_seconds - 0.25)
        start = float(rng.uniform(0.0, max_start))
        max_len = max(0.2, min(4.0, input_duration_seconds * 0.45))
        end = float(min(input_duration_seconds, start + rng.uniform(0.2, max_len)))
        if end <= start:
            end = min(input_duration_seconds, start + 0.2)
        cfg.start = start
        cfg.end = end
    else:
        cfg.start = None
        cfg.end = None

    cfg.target_lufs = float(rng.uniform(-30.0, -10.0)) if rng.random() < 0.45 else None
    cfg.target_peak_dbfs = float(rng.uniform(-9.0, -0.6)) if rng.random() < 0.7 else None
    cfg.use_true_peak = bool(rng.random() < 0.8)
    cfg.normalize_stage = cast(
        NormalizeStage,
        rng.choice(np.array(["none", "post", "per-pass"], dtype=object)),
    )
    if cfg.normalize_stage == "per-pass":
        cfg.repeat_target_lufs = (
            float(rng.uniform(-30.0, -12.0)) if rng.random() < 0.55 else cfg.target_lufs
        )
        cfg.repeat_target_peak_dbfs = (
            float(rng.uniform(-10.0, -0.8)) if rng.random() < 0.6 else cfg.target_peak_dbfs
        )
    else:
        cfg.repeat_target_lufs = None
        cfg.repeat_target_peak_dbfs = None

    cfg.shimmer = bool(rng.random() < 0.42)
    cfg.shimmer_semitones = float(rng.choice(np.array([7.0, 12.0, 19.0, 24.0], dtype=np.float64)))
    cfg.shimmer_mix = float(rng.uniform(0.05, 0.85))
    cfg.shimmer_feedback = float(rng.uniform(0.05, 0.95))
    cfg.shimmer_lowcut = float(rng.uniform(80.0, 600.0))
    cfg.shimmer_highcut = float(rng.uniform(2_000.0, 16_000.0))

    cfg.duck = bool(rng.random() < 0.45)
    cfg.duck_attack = float(rng.uniform(2.0, 120.0))
    cfg.duck_release = float(rng.uniform(80.0, 1200.0))
    cfg.bloom = float(rng.uniform(0.0, 6.0))
    cfg.tilt = float(rng.uniform(-8.0, 8.0))

    if rng.random() < 0.6:
        cfg.lowcut = float(rng.uniform(20.0, 500.0))
    else:
        cfg.lowcut = None
    if rng.random() < 0.8:
        min_high = (cfg.lowcut + 300.0) if cfg.lowcut is not None else 800.0
        cfg.highcut = float(rng.uniform(min_high, 20_000.0))
    else:
        cfg.highcut = None

    if rng.random() < 0.4:
        cfg.fdn_rt60_low = float(rng.uniform(6.0, 70.0))
        cfg.fdn_rt60_mid = float(rng.uniform(3.0, 40.0))
        cfg.fdn_rt60_high = float(rng.uniform(0.8, 20.0))
        cfg.fdn_tonal_correction_strength = float(rng.uniform(0.05, 0.9))
        cfg.fdn_xover_low_hz = float(rng.uniform(100.0, 700.0))
        cfg.fdn_xover_high_hz = float(
            rng.uniform(max(cfg.fdn_xover_low_hz + 250.0, 1_000.0), 8_000.0)
        )
    else:
        cfg.fdn_rt60_low = None
        cfg.fdn_rt60_mid = None
        cfg.fdn_rt60_high = None
        cfg.fdn_tonal_correction_strength = 0.0
        cfg.fdn_xover_low_hz = 250.0
        cfg.fdn_xover_high_hz = 4_000.0
    cfg.fdn_cascade = bool(rng.random() < 0.4)
    if cfg.fdn_cascade:
        cfg.fdn_cascade_mix = float(rng.uniform(0.15, 0.85))
        cfg.fdn_cascade_delay_scale = float(rng.uniform(0.25, 0.85))
        cfg.fdn_cascade_rt60_ratio = float(rng.uniform(0.2, 0.95))
    else:
        cfg.fdn_cascade_mix = 0.35
        cfg.fdn_cascade_delay_scale = 0.5
        cfg.fdn_cascade_rt60_ratio = 0.55
    if rng.random() < 0.45:
        cfg.fdn_link_filter = cast(
            str,
            rng.choice(np.array(["lowpass", "highpass"], dtype=object)),
        )
        cfg.fdn_link_filter_hz = float(rng.uniform(80.0, 12_000.0))
        cfg.fdn_link_filter_mix = float(rng.uniform(0.2, 1.0))
    else:
        cfg.fdn_link_filter = "none"
        cfg.fdn_link_filter_hz = 2_500.0
        cfg.fdn_link_filter_mix = 1.0
    cfg.algo_decorrelation_front = float(rng.uniform(0.0, 0.8)) if rng.random() < 0.45 else 0.0
    cfg.algo_decorrelation_rear = float(rng.uniform(0.0, 0.9)) if rng.random() < 0.45 else 0.0
    cfg.algo_decorrelation_top = float(rng.uniform(0.0, 0.9)) if rng.random() < 0.3 else 0.0
    cfg.ir_route_map = cast(
        str,
        rng.choice(np.array(["auto", "diagonal", "broadcast", "full"], dtype=object)),
    )
    if cfg.ir_route_map == "auto":
        cfg.conv_route_start = None
        cfg.conv_route_end = None
    elif rng.random() < 0.35:
        cfg.conv_route_start = "left"
        cfg.conv_route_end = "right"
    else:
        cfg.conv_route_start = None
        cfg.conv_route_end = None
    cfg.conv_route_curve = cast(
        str,
        rng.choice(np.array(["linear", "equal-power"], dtype=object)),
    )

    # Ensure convolution path has a valid IR source whenever selected.
    if cfg.engine in {"conv", "auto"} and rng.random() < 0.75:
        mode = int(rng.integers(0, 3))
        if mode == 0 and base.ir is not None:
            cfg.ir = base.ir
            cfg.ir_gen = False
            cfg.self_convolve = False
        elif mode == 1:
            cfg.self_convolve = True
            cfg.ir = None
            cfg.ir_gen = False
        else:
            cfg.ir_gen = True
            cfg.ir = None
            cfg.self_convolve = False
            cfg.ir_gen_mode = cast(
                IRMode,
                rng.choice(np.array(["hybrid", "fdn", "modal", "stochastic"], dtype=object)),
            )
            cfg.ir_gen_length = float(rng.uniform(5.0, 60.0))
            cfg.ir_gen_seed = int(rng.integers(0, 2_147_483_647))
    else:
        cfg.ir = None
        cfg.ir_gen = False
        cfg.self_convolve = False
        if cfg.engine == "conv":
            cfg.engine = "algo"
        cfg.ir_route_map = "auto"
        cfg.conv_route_start = None
        cfg.conv_route_end = None

    cfg.tail_limit = float(rng.uniform(2.0, 40.0)) if rng.random() < 0.4 else None
    return cfg


def _build_lucky_ir_gen_config(
    base: IRGenConfig,
    rng: np.random.Generator,
) -> IRGenConfig:
    """Build one randomized IR generation config for ``ir gen --lucky``."""
    cfg = IRGenConfig(**asdict(base))

    cfg.mode = cast(
        IRMode,
        rng.choice(np.array(["hybrid", "fdn", "stochastic", "modal"], dtype=object)),
    )
    cfg.seed = int(rng.integers(0, 2_147_483_647))
    cfg.length = float(np.clip(base.length * rng.uniform(0.5, 2.5), 0.1, 120.0))
    cfg.rt60 = float(np.clip((base.rt60 or 12.0) * rng.uniform(0.4, 2.5), 0.1, 180.0))
    cfg.rt60_low = None
    cfg.rt60_high = None

    cfg.damping = float(rng.uniform(0.0, 1.0))
    cfg.diffusion = float(rng.uniform(0.05, 1.0))
    cfg.density = float(rng.uniform(0.1, 2.2))
    cfg.mod_depth_ms = float(rng.uniform(0.0, 18.0))
    cfg.mod_rate_hz = float(rng.uniform(0.02, 1.2))

    cfg.lowcut = float(rng.uniform(20.0, 500.0)) if rng.random() < 0.6 else None
    if rng.random() < 0.8:
        min_high = (cfg.lowcut + 300.0) if cfg.lowcut is not None else 800.0
        cfg.highcut = float(rng.uniform(min_high, (cfg.sr * 0.48)))
    else:
        cfg.highcut = None
    cfg.tilt = float(rng.uniform(-8.0, 8.0))

    cfg.normalize = cast(
        Literal["none", "peak", "rms"],
        rng.choice(np.array(["none", "peak", "rms"], dtype=object)),
    )
    cfg.peak_dbfs = float(rng.uniform(-12.0, -0.5))
    cfg.target_lufs = float(rng.uniform(-32.0, -12.0)) if rng.random() < 0.45 else None
    cfg.true_peak = bool(rng.random() < 0.7)

    cfg.er_count = int(rng.integers(0, 96))
    cfg.er_max_delay_ms = float(rng.uniform(5.0, 180.0))
    cfg.er_decay_shape = cast(str, rng.choice(np.array(["exp", "linear", "sqrt"], dtype=object)))
    cfg.er_stereo_width = float(rng.uniform(0.0, 2.0))
    cfg.er_room = float(rng.uniform(0.1, 3.0))

    cfg.modal_count = int(rng.integers(8, 128))
    cfg.modal_q_min = float(rng.uniform(0.8, 20.0))
    cfg.modal_q_max = float(rng.uniform(max(cfg.modal_q_min + 0.5, 5.0), 120.0))
    cfg.modal_spread_cents = float(rng.uniform(0.0, 40.0))
    cfg.modal_low_hz = float(rng.uniform(30.0, 400.0))
    cfg.modal_high_hz = float(rng.uniform(max(cfg.modal_low_hz + 200.0, 1200.0), cfg.sr * 0.48))

    cfg.fdn_lines = int(rng.choice(np.array([4, 6, 8, 10, 12, 16], dtype=np.int32)))
    cfg.fdn_matrix = cast(
        str,
        rng.choice(
            np.array(
                [
                    "hadamard",
                    "householder",
                    "random_orthogonal",
                    "circulant",
                    "elliptic",
                    "tv_unitary",
                    "graph",
                ],
                dtype=object,
            )
        ),
    )
    if cfg.fdn_matrix == "tv_unitary":
        cfg.fdn_tv_rate_hz = float(rng.uniform(0.03, 0.45))
        cfg.fdn_tv_depth = float(rng.uniform(0.1, 0.8))
    else:
        cfg.fdn_tv_rate_hz = 0.0
        cfg.fdn_tv_depth = 0.0
    cfg.fdn_tv_seed = int(rng.integers(0, 2_147_483_647))
    cfg.fdn_sparse = bool(rng.random() < 0.45)
    cfg.fdn_sparse_degree = int(rng.integers(1, 6))
    if cfg.fdn_sparse and cfg.fdn_matrix in {"tv_unitary", "graph"}:
        cfg.fdn_sparse = False
    if cfg.fdn_matrix == "graph":
        cfg.fdn_graph_topology = cast(
            str,
            rng.choice(np.array(["ring", "path", "star", "random"], dtype=object)),
        )
        cfg.fdn_graph_degree = int(rng.integers(1, 8))
    else:
        cfg.fdn_graph_topology = "ring"
        cfg.fdn_graph_degree = 2
    cfg.fdn_graph_seed = int(rng.integers(0, 2_147_483_647))
    cfg.fdn_cascade = bool(rng.random() < 0.45)
    if cfg.fdn_cascade:
        cfg.fdn_cascade_mix = float(rng.uniform(0.1, 0.85))
        cfg.fdn_cascade_delay_scale = float(rng.uniform(0.2, 0.85))
        cfg.fdn_cascade_rt60_ratio = float(rng.uniform(0.15, 0.9))
    else:
        cfg.fdn_cascade_mix = 0.35
        cfg.fdn_cascade_delay_scale = 0.5
        cfg.fdn_cascade_rt60_ratio = 0.55
    if rng.random() < 0.55:
        cfg.fdn_rt60_low = float(rng.uniform(8.0, 90.0))
        cfg.fdn_rt60_mid = float(rng.uniform(4.0, 50.0))
        cfg.fdn_rt60_high = float(rng.uniform(1.0, 30.0))
        cfg.fdn_tonal_correction_strength = float(rng.uniform(0.05, 0.95))
        cfg.fdn_xover_low_hz = float(rng.uniform(80.0, 800.0))
        cfg.fdn_xover_high_hz = float(
            rng.uniform(
                max(1_200.0, cfg.fdn_xover_low_hz + 200.0),
                9_000.0,
            )
        )
    else:
        cfg.fdn_rt60_low = None
        cfg.fdn_rt60_mid = None
        cfg.fdn_rt60_high = None
        cfg.fdn_tonal_correction_strength = 0.0
        cfg.fdn_xover_low_hz = 250.0
        cfg.fdn_xover_high_hz = 4_000.0
    if rng.random() < 0.5:
        cfg.fdn_link_filter = cast(
            str,
            rng.choice(np.array(["lowpass", "highpass"], dtype=object)),
        )
        cfg.fdn_link_filter_hz = float(rng.uniform(80.0, cfg.sr * 0.45))
        cfg.fdn_link_filter_mix = float(rng.uniform(0.2, 1.0))
    else:
        cfg.fdn_link_filter = "none"
        cfg.fdn_link_filter_hz = 2_500.0
        cfg.fdn_link_filter_mix = 1.0
    if rng.random() < 0.4:
        dfm_count = int(rng.choice(np.array([1, cfg.fdn_lines], dtype=np.int32)))
        cfg.fdn_dfm_delays_ms = tuple(float(rng.uniform(0.1, 8.0)) for _ in range(dfm_count))
    else:
        cfg.fdn_dfm_delays_ms = ()
    cfg.fdn_stereo_inject = float(rng.uniform(0.0, 1.0))

    cfg.harmonic_align_strength = float(rng.uniform(0.0, 1.0))
    cfg.resonator = bool(rng.random() < 0.45)
    cfg.resonator_mix = float(rng.uniform(0.0, 0.95))
    cfg.resonator_modes = int(rng.integers(8, 80))
    cfg.resonator_q_min = float(rng.uniform(0.8, 24.0))
    cfg.resonator_q_max = float(rng.uniform(max(cfg.resonator_q_min + 0.5, 6.0), 140.0))
    cfg.resonator_low_hz = float(rng.uniform(20.0, 250.0))
    cfg.resonator_high_hz = float(
        rng.uniform(max(cfg.resonator_low_hz + 500.0, 1500.0), cfg.sr * 0.48)
    )
    cfg.resonator_late_start_ms = float(rng.uniform(0.0, 400.0))
    return cfg


def _build_lucky_ir_process_config(
    *,
    damping: float,
    lowcut: float | None,
    highcut: float | None,
    tilt: float,
    normalize: Literal["none", "peak", "rms"],
    peak_dbfs: float,
    target_lufs: float | None,
    true_peak: bool,
    rng: np.random.Generator,
    sr: int,
) -> LuckyIRProcessConfig:
    """Build one randomized shaping config for ``ir process --lucky``."""
    lucky_lowcut = lowcut if lowcut is not None else float(rng.uniform(20.0, 500.0))
    lucky_lowcut = float(np.clip(lucky_lowcut * rng.uniform(0.5, 2.0), 20.0, sr * 0.45))

    if highcut is None:
        min_high = min(sr * 0.48, lucky_lowcut + 100.0)
        lucky_highcut = float(rng.uniform(max(min_high, 400.0), sr * 0.49))
    else:
        lucky_highcut = float(np.clip(highcut * rng.uniform(0.5, 1.7), 200.0, sr * 0.49))
    if lucky_highcut <= lucky_lowcut:
        lucky_highcut = min(sr * 0.49, lucky_lowcut + 200.0)

    modes = np.array(["none", "peak", "rms"], dtype=object)
    normalize_mode = (
        cast(Literal["none", "peak", "rms"], rng.choice(modes))
        if rng.random() < 0.85
        else normalize
    )
    return {
        "damping": float(np.clip(damping * rng.uniform(0.4, 2.2), 0.0, 1.0)),
        "lowcut": lucky_lowcut if rng.random() < 0.8 else None,
        "highcut": lucky_highcut if rng.random() < 0.9 else None,
        "tilt": float(np.clip(tilt + rng.uniform(-6.0, 6.0), -12.0, 12.0)),
        "normalize": normalize_mode,
        "peak_dbfs": float(np.clip(peak_dbfs + rng.uniform(-8.0, 2.0), -18.0, -0.1)),
        "target_lufs": (
            float(np.clip((target_lufs or -22.0) + rng.uniform(-10.0, 10.0), -36.0, -8.0))
            if rng.random() < 0.6
            else None
        ),
        "true_peak": bool(rng.random() < 0.7 if true_peak else rng.random() < 0.4),
    }


def _resolve_lucky_seed(lucky_seed: int | None) -> int:
    """Resolve deterministic seed for lucky-mode batches."""
    if lucky_seed is not None:
        return int(lucky_seed)
    return int(np.random.default_rng().integers(0, 2_147_483_647))


def _resolve_ir_output_path(out_ir: Path, out_format: IRFileFormat) -> Path:
    """Resolve output IR path based on explicit format switch."""
    if out_format == "auto":
        return out_ir if out_ir.suffix else out_ir.with_suffix(".wav")

    suffix = ".aiff" if out_format == "aiff" else f".{out_format}"
    return out_ir.with_suffix(suffix)


def _parse_delay_list_ms(raw: str | None, *, option_name: str) -> tuple[float, ...]:
    """Parse a comma-separated millisecond delay list for CLI options."""
    if raw is None:
        return ()
    cleaned = raw.strip()
    if cleaned == "":
        return ()
    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            delay = float(part)
        except ValueError as exc:
            msg = f"{option_name} expects a comma-separated float list in milliseconds."
            raise typer.BadParameter(msg) from exc
        if delay <= 0.0:
            msg = f"{option_name} values must be > 0 ms."
            raise typer.BadParameter(msg)
        values.append(delay)
    if len(values) == 0:
        msg = f"{option_name} must include at least one numeric value."
        raise typer.BadParameter(msg)
    return tuple(values)


def _parse_gain_list(
    raw: str,
    *,
    option_name: str,
    min_value: float,
    max_value: float,
) -> tuple[float, ...]:
    """Parse one or more comma-separated gain values for CLI options."""
    cleaned = raw.strip()
    if cleaned == "":
        msg = f"{option_name} requires at least one numeric value."
        raise typer.BadParameter(msg)

    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            gain = float(part)
        except ValueError as exc:
            msg = f"{option_name} expects float values, optionally comma-separated."
            raise typer.BadParameter(msg) from exc
        if gain < min_value or gain > max_value:
            msg = f"{option_name} values must be in [{min_value}, {max_value}]."
            raise typer.BadParameter(msg)
        values.append(gain)

    if len(values) == 0:
        msg = f"{option_name} requires at least one numeric value."
        raise typer.BadParameter(msg)
    return tuple(values)


def _did_you_mean(value: str, choices: set[str]) -> str | None:
    token = str(value).strip().lower()
    if token == "":
        return None
    matches = get_close_matches(token, sorted(choices), n=1, cutoff=0.5)
    if len(matches) == 0:
        return None
    return str(matches[0])


def _choice_error(option_name: str, choices: set[str], actual: str) -> str:
    options = ", ".join(sorted(choices))
    suggestion = _did_you_mean(actual, choices)
    if suggestion is not None:
        return (
            f"{option_name} must be one of: {options}. "
            f"Did you mean '{suggestion}'?"
        )
    return f"{option_name} must be one of: {options}."


def _param_is_default(ctx: typer.Context, param_name: str) -> bool:
    """Return True when a Click/Typer param came from default value."""
    try:
        source = ctx.get_parameter_source(param_name)
    except Exception:
        return True
    return source in {None, ParameterSource.DEFAULT}


def _apply_render_preset(
    *,
    ctx: typer.Context,
    config: RenderConfig,
    preset_name: str,
) -> dict[str, Any]:
    """Apply preset values only where user did not explicitly provide a CLI override."""
    resolved_name, preset_values = resolve_preset(preset_name)
    applied: dict[str, Any] = {}
    skipped: list[str] = []
    fields = RenderConfig.__dataclass_fields__.keys()

    for key, value in sorted(preset_values.items()):
        if key not in fields:
            continue
        if not _param_is_default(ctx, key):
            skipped.append(str(key))
            continue
        setattr(config, key, value)
        applied[str(key)] = value

    return {
        "name": resolved_name,
        "applied": applied,
        "skipped": tuple(skipped),
    }


def _estimate_render_output_duration_seconds(*, infile: Path, config: RenderConfig) -> float | None:
    """Estimate render output duration for dry-run planning."""
    try:
        info = sf.info(str(infile))
    except (RuntimeError, TypeError, ValueError):
        return None
    sr = int(info.samplerate)
    frames = int(info.frames)
    if sr <= 0:
        return None

    duration_s = float(frames) / float(sr)
    if config.engine == "algo" or (
        config.engine == "auto"
        and not (config.ir is not None or config.ir_gen or config.self_convolve)
    ):
        duration_s += max(
            0.25,
            float(config.rt60) + (max(0.0, float(config.pre_delay_ms)) / 1000.0),
        )
    return duration_s


def _estimate_output_file_size_mb(
    *,
    duration_s: float,
    sample_rate: int,
    channels: int,
    out_subtype_mode: str,
) -> float | None:
    """Estimate output file size in MiB for dry-run planning."""
    if duration_s <= 0.0 or sample_rate <= 0 or channels <= 0:
        return None
    bytes_per_sample_map = {
        "float64": 8,
        "float32": 4,
        "pcm32": 4,
        "pcm24": 3,
        "pcm16": 2,
    }
    bytes_per_sample = bytes_per_sample_map.get(str(out_subtype_mode).strip().lower(), 8)
    frames = int(np.ceil(float(duration_s) * float(sample_rate)))
    total_bytes = int(frames * int(channels) * int(bytes_per_sample))
    return float(total_bytes / float(1024 * 1024))


def _print_render_dry_run_plan(
    *,
    infile: Path,
    outfile: Path,
    config: RenderConfig,
    lucky: int | None,
    lucky_out_dir: Path | None,
    preset_summary: dict[str, Any] | None,
    repro_bundle_path: Path | None,
) -> None:
    """Print resolved render plan without writing audio."""
    table = Table(title="Render Dry-Run Plan")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", "dry-run")
    table.add_row("infile", str(infile))
    table.add_row("outfile", str(outfile))
    if preset_summary is not None:
        table.add_row("preset", str(preset_summary.get("name", "")))
        applied = preset_summary.get("applied", {})
        skipped = preset_summary.get("skipped", ())
        table.add_row(
            "preset_applied_fields",
            str(len(applied) if isinstance(applied, dict) else 0),
        )
        table.add_row(
            "preset_overridden_fields",
            str(len(skipped) if isinstance(skipped, tuple) else 0),
        )

    table.add_row("engine_requested", str(config.engine))
    table.add_row("device_requested", str(config.device))
    table.add_row("rt60", f"{float(config.rt60):.3f}")
    table.add_row("wet", f"{float(config.wet):.3f}")
    table.add_row("dry", f"{float(config.dry):.3f}")
    table.add_row("repeat", str(int(config.repeat)))
    table.add_row("fdn_matrix", str(config.fdn_matrix))
    table.add_row("fdn_lines", str(int(config.fdn_lines)))
    table.add_row("ir", str(config.ir))
    table.add_row("ir_blend_count", str(len(config.ir_blend)))
    estimated_duration_s = _estimate_render_output_duration_seconds(infile=infile, config=config)
    if estimated_duration_s is not None:
        table.add_row("estimated_output_duration_s", f"{estimated_duration_s:.3f}")
        try:
            info = sf.info(str(infile))
            sample_rate = int(info.samplerate)
            channels = int(info.channels)
            estimated_size_mb = _estimate_output_file_size_mb(
                duration_s=estimated_duration_s,
                sample_rate=sample_rate,
                channels=channels,
                out_subtype_mode=str(config.output_subtype),
            )
            if estimated_size_mb is not None:
                table.add_row("estimated_output_size_mb", f"{estimated_size_mb:.3f}")
        except (RuntimeError, TypeError, ValueError):
            pass
    table.add_row("automation_file", str(config.automation_file))
    table.add_row("automation_points", str(len(config.automation_points)))
    table.add_row("feature_vector_lanes", str(len(config.feature_vector_lanes)))
    try:
        targets = sorted(
            collect_automation_targets(
                path=(
                    None
                    if config.automation_file is None
                    else Path(config.automation_file)
                ),
                point_specs=config.automation_points,
                feature_lane_specs=config.feature_vector_lanes,
            )
        )
    except ValueError:
        targets = []
    if len(targets) > 0:
        table.add_row("automation_targets", ",".join(targets))
    if lucky is not None:
        target_dir = outfile.parent if lucky_out_dir is None else lucky_out_dir
        table.add_row("lucky_count", str(int(lucky)))
        table.add_row("lucky_out_dir", str(target_dir))
    if not config.silent:
        analysis_path = Path(config.analysis_out) if config.analysis_out is not None else Path(
            f"{outfile}.analysis.json"
        )
        table.add_row("analysis_out", str(analysis_path))
    if repro_bundle_path is not None:
        table.add_row("repro_bundle_out", str(repro_bundle_path))

    # Dry-run means no disk audio write. Yes, this is the point.
    table.add_row("audio_write", "skipped")
    console.print(table)


def _validate_fdn_matrix_name(fdn_matrix: str) -> None:
    """Validate FDN matrix topology identifier."""
    normalized = _normalize_fdn_matrix_name(fdn_matrix)
    if normalized not in _FDN_MATRIX_CHOICES:
        raise typer.BadParameter(
            _choice_error("--fdn-matrix", _FDN_MATRIX_CHOICES, normalized)
        )


def _validate_fdn_tv_settings(
    *,
    fdn_matrix: str,
    fdn_tv_rate_hz: float,
    fdn_tv_depth: float,
) -> None:
    """Validate time-varying unitary matrix controls."""
    normalized = _normalize_fdn_matrix_name(fdn_matrix)
    rate = float(fdn_tv_rate_hz)
    depth = float(fdn_tv_depth)
    if normalized == "tv_unitary":
        if rate <= 0.0 or depth <= 0.0:
            msg = (
                "--fdn-matrix tv_unitary requires both --fdn-tv-rate-hz > 0 and --fdn-tv-depth > 0."
            )
            raise typer.BadParameter(msg)
        return

    if rate > 0.0 or depth > 0.0:
        msg = "--fdn-tv-rate-hz and --fdn-tv-depth are only valid with --fdn-matrix tv_unitary."
        raise typer.BadParameter(msg)


def _validate_fdn_sparse_settings(
    *,
    fdn_matrix: str,
    fdn_sparse: bool,
    fdn_sparse_degree: int,
) -> None:
    """Validate sparse high-order FDN settings."""
    normalized = _normalize_fdn_matrix_name(fdn_matrix)
    if fdn_sparse_degree < 1:
        msg = "--fdn-sparse-degree must be >= 1."
        raise typer.BadParameter(msg)
    if fdn_sparse and normalized == "tv_unitary":
        msg = "--fdn-sparse cannot be combined with --fdn-matrix tv_unitary."
        raise typer.BadParameter(msg)
    if fdn_sparse and normalized == "graph":
        msg = "--fdn-sparse cannot be combined with --fdn-matrix graph."
        raise typer.BadParameter(msg)


def _validate_fdn_graph_settings(
    *,
    fdn_matrix: str,
    fdn_graph_topology: str,
    fdn_graph_degree: int,
) -> None:
    """Validate graph-structured FDN controls."""
    normalized_matrix = _normalize_fdn_matrix_name(fdn_matrix)
    normalized_topology = _normalize_fdn_graph_topology_name(fdn_graph_topology)

    if normalized_matrix == "graph":
        if normalized_topology not in _FDN_GRAPH_TOPOLOGY_CHOICES:
            raise typer.BadParameter(
                _choice_error(
                    "--fdn-graph-topology",
                    _FDN_GRAPH_TOPOLOGY_CHOICES,
                    normalized_topology,
                )
            )
        if int(fdn_graph_degree) < 1:
            msg = "--fdn-graph-degree must be >= 1."
            raise typer.BadParameter(msg)
        return

    # Non-default graph options are considered a configuration mismatch unless graph mode is active.
    if normalized_topology != "ring" or int(fdn_graph_degree) != 2:
        msg = "--fdn-graph-topology/--fdn-graph-degree are only valid with --fdn-matrix graph."
        raise typer.BadParameter(msg)


def _validate_fdn_cascade_settings(
    *,
    fdn_lines: int,
    fdn_cascade: bool,
    fdn_cascade_mix: float,
    fdn_cascade_delay_scale: float,
    fdn_cascade_rt60_ratio: float,
) -> None:
    """Validate nested/cascaded FDN controls."""
    if not fdn_cascade:
        return
    if fdn_lines < 2:
        msg = "--fdn-cascade requires at least 2 FDN lines."
        raise typer.BadParameter(msg)
    if float(fdn_cascade_mix) <= 0.0:
        msg = "--fdn-cascade-mix must be > 0 when --fdn-cascade is enabled."
        raise typer.BadParameter(msg)
    if float(fdn_cascade_delay_scale) <= 0.0:
        msg = "--fdn-cascade-delay-scale must be > 0."
        raise typer.BadParameter(msg)
    if float(fdn_cascade_rt60_ratio) <= 0.0:
        msg = "--fdn-cascade-rt60-ratio must be > 0."
        raise typer.BadParameter(msg)


def _validate_fdn_multiband_settings(
    *,
    fdn_rt60_low: float | None,
    fdn_rt60_mid: float | None,
    fdn_rt60_high: float | None,
    fdn_xover_low_hz: float,
    fdn_xover_high_hz: float,
) -> None:
    """Validate multiband FDN decay controls."""
    set_count = sum(value is not None for value in (fdn_rt60_low, fdn_rt60_mid, fdn_rt60_high))
    if set_count not in {0, 3}:
        msg = (
            "For multiband decay use either none of --fdn-rt60-low/mid/high "
            "or provide all three values."
        )
        raise typer.BadParameter(msg)
    if float(fdn_xover_low_hz) >= float(fdn_xover_high_hz):
        msg = "--fdn-xover-low-hz must be < --fdn-xover-high-hz."
        raise typer.BadParameter(msg)


def _validate_fdn_link_filter_settings(
    *,
    fdn_link_filter: str,
    fdn_link_filter_hz: float,
    fdn_link_filter_mix: float,
) -> None:
    """Validate feedback-link filter controls used in the FDN path."""
    normalized = _normalize_fdn_link_filter_name(fdn_link_filter)
    if normalized not in _FDN_LINK_FILTER_CHOICES:
        raise typer.BadParameter(
            _choice_error("--fdn-link-filter", _FDN_LINK_FILTER_CHOICES, normalized)
        )
    if float(fdn_link_filter_hz) <= 0.0:
        msg = "--fdn-link-filter-hz must be > 0."
        raise typer.BadParameter(msg)
    mix = float(fdn_link_filter_mix)
    if mix < 0.0 or mix > 1.0:
        msg = "--fdn-link-filter-mix must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)


def _validate_fdn_spatial_coupling_settings(
    *,
    fdn_spatial_coupling_mode: str,
    fdn_spatial_coupling_strength: float,
) -> None:
    """Validate directional wet-bus coupling controls."""
    normalized = str(fdn_spatial_coupling_mode).strip().lower().replace("-", "_")
    if normalized not in _FDN_SPATIAL_COUPLING_CHOICES:
        raise typer.BadParameter(
            _choice_error(
                "--fdn-spatial-coupling-mode",
                _FDN_SPATIAL_COUPLING_CHOICES,
                normalized,
            )
        )
    strength = float(fdn_spatial_coupling_strength)
    if strength < 0.0 or strength > 1.0:
        msg = "--fdn-spatial-coupling-strength must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)
    if normalized == "none" and strength > 0.0:
        msg = (
            "--fdn-spatial-coupling-strength must be 0 when "
            "--fdn-spatial-coupling-mode none is selected."
        )
        raise typer.BadParameter(msg)


def _validate_fdn_nonlinearity_settings(
    *,
    fdn_nonlinearity: str,
    fdn_nonlinearity_amount: float,
    fdn_nonlinearity_drive: float,
) -> None:
    """Validate bounded in-loop nonlinearity controls."""
    normalized = str(fdn_nonlinearity).strip().lower().replace("-", "_")
    if normalized not in _FDN_NONLINEARITY_CHOICES:
        raise typer.BadParameter(
            _choice_error("--fdn-nonlinearity", _FDN_NONLINEARITY_CHOICES, normalized)
        )
    amount = float(fdn_nonlinearity_amount)
    if amount < 0.0 or amount > 1.0:
        msg = "--fdn-nonlinearity-amount must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)
    drive = float(fdn_nonlinearity_drive)
    if drive < 0.1 or drive > 8.0:
        msg = "--fdn-nonlinearity-drive must be in [0.1, 8.0]."
        raise typer.BadParameter(msg)
    if normalized == "none" and amount > 0.0:
        msg = (
            "--fdn-nonlinearity-amount must be 0 when --fdn-nonlinearity none is selected."
        )
        raise typer.BadParameter(msg)


def _validate_perceptual_macro_settings(
    *,
    fdn_rt60_tilt: float,
    room_size_macro: float,
    clarity_macro: float,
    warmth_macro: float,
    envelopment_macro: float,
) -> None:
    """Validate perceptual macro controls and Jot-inspired RT tilt."""
    values = {
        "--fdn-rt60-tilt": float(fdn_rt60_tilt),
        "--room-size-macro": float(room_size_macro),
        "--clarity-macro": float(clarity_macro),
        "--warmth-macro": float(warmth_macro),
        "--envelopment-macro": float(envelopment_macro),
    }
    for option_name, value in values.items():
        if value < -1.0 or value > 1.0:
            msg = f"{option_name} must be in [-1.0, 1.0]."
            raise typer.BadParameter(msg)


def _validate_fdn_tonal_correction_settings(*, fdn_tonal_correction_strength: float) -> None:
    """Validate Track C tonal-correction controls."""
    strength = float(fdn_tonal_correction_strength)
    if strength < 0.0 or strength > 1.0:
        msg = "--fdn-tonal-correction-strength must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)


def _normalize_fdn_matrix_name(value: str) -> str:
    """Normalize FDN matrix identifier for CLI/API compatibility."""
    return _shared_normalize_fdn_matrix_name(value)


def _normalize_fdn_link_filter_name(value: str) -> str:
    """Normalize FDN feedback-link filter identifier for CLI/API compatibility."""
    return _shared_normalize_fdn_link_filter_name(value)


def _normalize_fdn_graph_topology_name(value: str) -> str:
    """Normalize graph-structured FDN topology identifier."""
    return _shared_normalize_fdn_graph_topology_name(value)


def _normalize_ir_route_map_name(value: str) -> str:
    """Normalize convolution IR route-map identifier."""
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"diag"}:
        return "diagonal"
    if normalized in {"full_matrix", "fullmatrix"}:
        return "full"
    return normalized


def _normalize_conv_route_curve_name(value: str) -> str:
    """Normalize convolution route trajectory curve name."""
    return value.strip().lower().replace("_", "-")


def _validate_ir_route_map_name(value: str) -> None:
    """Validate named convolution route-map preset."""
    normalized = _normalize_ir_route_map_name(value)
    if normalized not in _IR_ROUTE_MAP_CHOICES:
        raise typer.BadParameter(
            _choice_error("--ir-route-map", _IR_ROUTE_MAP_CHOICES, normalized)
        )


def _validate_ir_blend_settings(config: RenderConfig) -> None:
    """Validate render-time IR blending controls."""
    has_blend = len(config.ir_blend) > 0
    has_blend_args = (
        len(config.ir_blend_mix) > 0
        or config.ir_blend_mode != "equal-power"
        or abs(float(config.ir_blend_early_ms) - 80.0) > 1e-12
        or config.ir_blend_early_alpha is not None
        or config.ir_blend_late_alpha is not None
        or not bool(config.ir_blend_align_decay)
        or abs(float(config.ir_blend_phase_coherence) - 0.75) > 1e-12
        or int(config.ir_blend_spectral_smooth_bins) != 3
        or config.ir_blend_mismatch_policy != "coerce"
        or config.ir_blend_cache_dir != ".verbx_cache/ir_morph"
    )
    if not has_blend and has_blend_args:
        msg = (
            "--ir-blend-mix/--ir-blend-mode/--ir-blend-early-ms/"
            "--ir-blend-early-alpha/--ir-blend-late-alpha/"
            "--ir-blend-align-decay/--ir-blend-phase-coherence/"
            "--ir-blend-spectral-smooth-bins/--ir-blend-mismatch-policy/"
            "--ir-blend-cache-dir "
            "require at least one --ir-blend path."
        )
        raise typer.BadParameter(msg)
    if not has_blend:
        return

    if config.engine == "algo":
        msg = "--ir-blend requires convolution render path (use --engine conv or --engine auto)."
        raise typer.BadParameter(msg)

    has_base_ir_source = config.ir is not None or config.ir_gen or config.self_convolve
    if not has_base_ir_source:
        msg = "--ir-blend requires base IR source via --ir, --ir-gen, or --self-convolve."
        raise typer.BadParameter(msg)

    try:
        config.ir_blend_mode = validate_ir_morph_mode_name(config.ir_blend_mode)
        config.ir_blend_mismatch_policy = cast(
            IRMorphMismatchPolicy,
            validate_ir_morph_mismatch_policy_name(config.ir_blend_mismatch_policy),
        )
        resolve_blend_mix_values(config.ir_blend_mix, len(config.ir_blend))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if str(config.ir_blend_cache_dir).strip() == "":
        msg = "--ir-blend-cache-dir must not be empty."
        raise typer.BadParameter(msg)


def _validate_conv_route_settings(
    *,
    conv_route_start: str | None,
    conv_route_end: str | None,
    conv_route_curve: str,
) -> None:
    """Validate convolution route-trajectory controls."""
    if (conv_route_start is None) != (conv_route_end is None):
        msg = "Use both --conv-route-start and --conv-route-end together."
        raise typer.BadParameter(msg)
    normalized_curve = _normalize_conv_route_curve_name(conv_route_curve)
    if normalized_curve not in _CONV_ROUTE_CURVE_CHOICES:
        raise typer.BadParameter(
            _choice_error("--conv-route-curve", _CONV_ROUTE_CURVE_CHOICES, normalized_curve)
        )


def _validate_ambisonic_settings(infile: Path, config: RenderConfig) -> None:
    """Validate Ambisonics render options and channel/layout compatibility."""
    order = int(config.ambi_order)
    if order < 0:
        raise typer.BadParameter("--ambi-order must be >= 0.")

    if order > 1:
        console.print(
            f"[yellow]Warning:[/yellow] HOA order {order} is beyond first-order Ambisonics (FOA). "
            "Higher-order paths exist in verbx but have not been fully validated — "
            "results may be incorrect. "
            "FOA (--ambi-order 1) is the supported configuration for v0.7.x."
        )

    if config.ambi_normalization not in _AMBI_NORMALIZATION_CHOICES:
        raise typer.BadParameter(
            _choice_error(
                "--ambi-normalization",
                _AMBI_NORMALIZATION_CHOICES,
                str(config.ambi_normalization),
            )
        )
    if config.channel_order not in _AMBI_CHANNEL_ORDER_CHOICES:
        raise typer.BadParameter(
            _choice_error("--channel-order", _AMBI_CHANNEL_ORDER_CHOICES, str(config.channel_order))
        )
    if config.ambi_encode_from not in _AMBI_ENCODE_CHOICES:
        raise typer.BadParameter(
            _choice_error("--ambi-encode-from", _AMBI_ENCODE_CHOICES, str(config.ambi_encode_from))
        )
    if config.ambi_decode_to not in _AMBI_DECODE_CHOICES:
        raise typer.BadParameter(
            _choice_error("--ambi-decode-to", _AMBI_DECODE_CHOICES, str(config.ambi_decode_to))
        )

    if order == 0:
        if config.ambi_encode_from != "none":
            raise typer.BadParameter("--ambi-encode-from requires --ambi-order 1.")
        if config.ambi_decode_to != "none":
            raise typer.BadParameter("--ambi-decode-to requires --ambi-order >= 1.")
        if abs(float(config.ambi_rotate_yaw_deg)) > 1e-12:
            raise typer.BadParameter("--ambi-rotate-yaw-deg requires --ambi-order >= 1.")
        if config.ambi_normalization != "auto" or config.channel_order != "auto":
            raise typer.BadParameter(
                "--ambi-normalization/--channel-order require --ambi-order >= 1."
            )
        return

    try:
        normalize_ambisonic_metadata(
            order=order,
            normalization=config.ambi_normalization,
            channel_order=config.channel_order,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    expected_channels = ambisonic_channel_count(order)
    if config.input_layout != "auto":
        msg = "Ambisonics mode requires --input-layout auto."
        raise typer.BadParameter(msg)
    if config.ambi_decode_to == "none" and config.output_layout != "auto":
        msg = (
            "Ambisonics mode requires --output-layout auto unless --ambi-decode-to stereo is used."
        )
        raise typer.BadParameter(msg)

    try:
        in_info = sf.info(str(infile))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    input_channels = int(in_info.channels)

    if config.ambi_encode_from == "none":
        if input_channels != expected_channels:
            msg = (
                f"Input channels ({input_channels}) do not match --ambi-order {order} "
                f"({expected_channels} channels expected). "
                "Use matching Ambisonic input or set --ambi-encode-from mono|stereo."
            )
            raise typer.BadParameter(msg)
    else:
        if order != 1:
            msg = "--ambi-encode-from currently supports FOA only (--ambi-order 1)."
            raise typer.BadParameter(msg)
        required_channels = 1 if config.ambi_encode_from == "mono" else 2
        if input_channels != required_channels:
            msg = (
                f"--ambi-encode-from {config.ambi_encode_from} expects "
                f"{required_channels} input channel(s), got {input_channels}."
            )
            raise typer.BadParameter(msg)

    if config.ambi_decode_to == "stereo" and config.output_layout not in {"auto", "stereo"}:
        msg = "--ambi-decode-to stereo is only valid with --output-layout auto or stereo."
        raise typer.BadParameter(msg)


def _validate_automation_settings(config: RenderConfig, outfile: Path) -> None:
    """Validate automation file/options before render dispatch."""
    if config.automation_mode not in _AUTOMATION_MODE_CHOICES:
        raise typer.BadParameter(
            _choice_error(
                "--automation-mode",
                _AUTOMATION_MODE_CHOICES,
                str(config.automation_mode),
            )
        )

    if config.automation_block_ms <= 0.0:
        raise typer.BadParameter("--automation-block-ms must be > 0.")
    if config.automation_smoothing_ms < 0.0:
        raise typer.BadParameter("--automation-smoothing-ms must be >= 0.")
    if (
        config.automation_slew_limit_per_s is not None
        and config.automation_slew_limit_per_s < 0.0
    ):
        raise typer.BadParameter("--automation-slew-limit-per-s must be >= 0.")
    if not (0.0 <= float(config.automation_deadband) <= 1.0):
        raise typer.BadParameter("--automation-deadband must be in [0.0, 1.0].")
    if config.feature_vector_frame_ms <= 0.0:
        raise typer.BadParameter("--feature-vector-frame-ms must be > 0.")
    if config.feature_vector_hop_ms <= 0.0:
        raise typer.BadParameter("--feature-vector-hop-ms must be > 0.")
    if config.feature_guide_policy not in FEATURE_GUIDE_POLICY_CHOICES:
        raise typer.BadParameter(
            _choice_error(
                "--feature-guide-policy",
                set(FEATURE_GUIDE_POLICY_CHOICES),
                str(config.feature_guide_policy),
            )
        )
    if config.feature_guide is not None and not Path(config.feature_guide).exists():
        raise typer.BadParameter(f"Feature guide file not found: {config.feature_guide}")

    has_automation_source = (
        config.automation_file is not None
        or len(config.automation_points) > 0
        or len(config.feature_vector_lanes) > 0
    )
    has_automation_args = (
        config.automation_mode != "auto"
        or abs(float(config.automation_block_ms) - 20.0) > 1e-12
        or abs(float(config.automation_smoothing_ms) - 20.0) > 1e-12
        or (
            config.automation_slew_limit_per_s is not None
            and abs(float(config.automation_slew_limit_per_s)) > 1e-12
        )
        or abs(float(config.automation_deadband)) > 1e-12
        or len(config.automation_clamp) > 0
        or config.automation_trace_out is not None
        or abs(float(config.feature_vector_frame_ms) - 40.0) > 1e-12
        or abs(float(config.feature_vector_hop_ms) - 20.0) > 1e-12
        or config.feature_guide is not None
        or config.feature_guide_policy != "align"
        or config.feature_vector_trace_out is not None
    )
    if not has_automation_source and has_automation_args:
        msg = (
            "--automation-mode/--automation-block-ms/--automation-smoothing-ms/"
            "--automation-slew-limit-per-s/--automation-deadband/"
            "--automation-clamp/--automation-trace-out/--feature-vector-frame-ms/"
            "--feature-vector-hop-ms/--feature-guide/--feature-guide-policy/"
            "--feature-vector-trace-out require --automation-file, --automation-point, "
            "or --feature-vector-lane."
        )
        raise typer.BadParameter(msg)

    if not has_automation_source:
        return

    source_path: Path | None = None
    if config.automation_file is not None:
        source_path = Path(config.automation_file)
        if not source_path.exists():
            raise typer.BadParameter(f"Automation file not found: {config.automation_file}")
        if source_path.suffix.lower() not in {".json", ".csv"}:
            raise typer.BadParameter("--automation-file must be a .json or .csv file.")

    try:
        parse_automation_clamp_overrides(config.automation_clamp)
        parse_automation_point_specs(config.automation_points)
        parse_feature_vector_lane_specs(config.feature_vector_lanes)
        targets = collect_automation_targets(
            path=source_path,
            point_specs=config.automation_points,
            feature_lane_specs=config.feature_vector_lanes,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    conv_selected = config.engine == "conv" or (
        config.engine == "auto"
        and (
            config.ir is not None
            or config.ir_gen
            or config.self_convolve
            or len(config.ir_blend) > 0
        )
    )
    if conv_selected:
        engine_targets = sorted(target for target in targets if target in ENGINE_AUTOMATION_TARGETS)
        if len(engine_targets) > 0:
            joined = ", ".join(engine_targets)
            msg = (
                "Automation targets require algorithmic render path; "
                f"use --engine algo (targets: {joined})."
            )
            raise typer.BadParameter(msg)
        conv_targets = sorted(target for target in targets if target in CONV_AUTOMATION_TARGETS)
        if "ir-blend-alpha" in conv_targets and len(config.ir_blend) == 0:
            msg = "Automation target 'ir-blend-alpha' requires --ir-blend."
            raise typer.BadParameter(msg)
    else:
        conv_targets = sorted(target for target in targets if target in CONV_AUTOMATION_TARGETS)
        if len(conv_targets) > 0:
            joined = ", ".join(conv_targets)
            msg = (
                "Automation targets require convolution render path; "
                f"use --engine conv (targets: {joined})."
            )
            raise typer.BadParameter(msg)

    if config.automation_trace_out is not None:
        trace_path = Path(config.automation_trace_out)
        if trace_path.resolve() == outfile.resolve():
            raise typer.BadParameter("--automation-trace-out must differ from OUTFILE.")
        if trace_path.suffix.lower() != ".csv":
            raise typer.BadParameter("--automation-trace-out must use .csv extension.")
    if config.feature_vector_trace_out is not None:
        feature_trace = Path(config.feature_vector_trace_out)
        if feature_trace.resolve() == outfile.resolve():
            raise typer.BadParameter("--feature-vector-trace-out must differ from OUTFILE.")
        if feature_trace.suffix.lower() != ".csv":
            raise typer.BadParameter("--feature-vector-trace-out must use .csv extension.")


def _resolve_repro_bundle_path(
    *,
    outfile: Path,
    repro_bundle: bool,
    repro_bundle_out: Path | None,
) -> Path | None:
    """Resolve final repro bundle path from render CLI options."""
    if repro_bundle_out is not None:
        return repro_bundle_out.resolve()
    if repro_bundle:
        return Path(f"{outfile}.repro.json").resolve()
    return None


def _validate_repro_bundle_path(
    *,
    infile: Path,
    outfile: Path,
    analysis_out: str | None,
    repro_bundle_path: Path | None,
) -> None:
    """Validate optional repro bundle output path for render command."""
    if repro_bundle_path is None:
        return
    if repro_bundle_path.suffix.lower() != ".json":
        raise typer.BadParameter("--repro-bundle-out must use .json extension.")
    if repro_bundle_path.resolve() in {infile.resolve(), outfile.resolve()}:
        raise typer.BadParameter("--repro-bundle-out must differ from INFILE and OUTFILE.")
    if analysis_out is not None and repro_bundle_path.resolve() == Path(analysis_out).resolve():
        raise typer.BadParameter("--repro-bundle-out must differ from --analysis-out.")


def _validate_failure_report_path(
    *,
    infile: Path,
    outfile: Path,
    analysis_out: str | None,
    repro_bundle_path: Path | None,
    failure_report_out: Path | None,
) -> None:
    """Validate optional failure-report output path for render command."""
    if failure_report_out is None:
        return
    resolved = failure_report_out.resolve()
    if resolved.suffix.lower() != ".json":
        raise typer.BadParameter("--failure-report-out must use .json extension.")
    if resolved in {infile.resolve(), outfile.resolve()}:
        raise typer.BadParameter("--failure-report-out must differ from INFILE and OUTFILE.")
    if analysis_out is not None and resolved == Path(analysis_out).resolve():
        raise typer.BadParameter("--failure-report-out must differ from --analysis-out.")
    if repro_bundle_path is not None and resolved == repro_bundle_path.resolve():
        raise typer.BadParameter("--failure-report-out must differ from --repro-bundle-out.")


def _validate_lucky_call(
    config: RenderConfig,
    lucky: int | None,
    lucky_out_dir: Path | None,
    repro_bundle_path: Path | None = None,
    failure_report_out: Path | None = None,
) -> None:
    """Validate lucky-mode options for randomized batch rendering."""
    _validate_generic_lucky_call(lucky, lucky_out_dir)
    if lucky is None:
        return
    if repro_bundle_path is not None:
        msg = "Do not use --repro-bundle/--repro-bundle-out with --lucky."
        raise typer.BadParameter(msg)
    if failure_report_out is not None:
        msg = "Do not use --failure-report-out with --lucky."
        raise typer.BadParameter(msg)
    if config.analysis_out is not None:
        msg = "Do not use --analysis-out with --lucky (analysis files are per-output by default)."
        raise typer.BadParameter(msg)
    if config.frames_out is not None:
        msg = "Do not use --frames-out with --lucky."
        raise typer.BadParameter(msg)
    if config.automation_trace_out is not None:
        msg = "Do not use --automation-trace-out with --lucky."
        raise typer.BadParameter(msg)
    if config.feature_vector_trace_out is not None:
        msg = "Do not use --feature-vector-trace-out with --lucky."
        raise typer.BadParameter(msg)


def _validate_generic_lucky_call(lucky: int | None, lucky_out_dir: Path | None) -> None:
    """Validate generic lucky-mode options shared by multiple commands."""
    if lucky is None:
        return
    if lucky < 1:
        msg = "--lucky must be >= 1."
        raise typer.BadParameter(msg)
    if lucky_out_dir is not None and lucky_out_dir.exists() and not lucky_out_dir.is_dir():
        msg = f"--lucky-out-dir is not a directory: {lucky_out_dir}"
        raise typer.BadParameter(msg)


def _validate_render_call(infile: Path, outfile: Path, config: RenderConfig) -> None:
    """Validate render CLI arguments before pipeline execution."""
    _ensure_distinct_paths(infile, outfile, "INFILE", "OUTFILE")
    _validate_output_audio_path(outfile, config.output_subtype)

    if config.self_convolve:
        if config.ir is not None:
            msg = "Use either --ir or --self-convolve, not both."
            raise typer.BadParameter(msg)
        if config.ir_gen:
            msg = "Use either --ir-gen or --self-convolve, not both."
            raise typer.BadParameter(msg)
        if config.engine == "algo":
            msg = "--self-convolve is only valid with --engine conv or --engine auto."
            raise typer.BadParameter(msg)

    _validate_ir_blend_settings(config)

    if (
        config.engine == "conv"
        and config.ir is None
        and not config.ir_gen
        and not config.self_convolve
    ):
        msg = "Convolution render requires --ir PATH, --ir-gen, or --self-convolve."
        raise typer.BadParameter(msg)

    if config.ir is not None and not Path(config.ir).exists():
        msg = f"IR file not found: {config.ir}"
        raise typer.BadParameter(msg)

    _validate_ir_route_map_name(config.ir_route_map)
    _validate_conv_route_settings(
        conv_route_start=config.conv_route_start,
        conv_route_end=config.conv_route_end,
        conv_route_curve=config.conv_route_curve,
    )
    _validate_ambisonic_settings(infile, config)
    _validate_automation_settings(config, outfile)

    conv_enabled = config.engine == "conv" or (
        config.engine == "auto" and (config.ir is not None or config.ir_gen or config.self_convolve)
    )
    algo_enabled = not conv_enabled
    if config.unsafe_self_oscillate and not algo_enabled:
        msg = (
            "--unsafe-self-oscillate is only valid for algorithmic renders "
            "(--engine algo or auto without convolution inputs)."
        )
        raise typer.BadParameter(msg)
    if not config.unsafe_self_oscillate and config.shimmer_feedback > 0.98:
        msg = (
            "--shimmer-feedback above 0.98 requires --unsafe-self-oscillate. "
            "Default safe range is 0.0..0.98."
        )
        raise typer.BadParameter(msg)
    if not config.unsafe_self_oscillate and abs(float(config.unsafe_loop_gain) - 1.02) > 1e-9:
        msg = "--unsafe-loop-gain requires --unsafe-self-oscillate."
        raise typer.BadParameter(msg)
    if config.unsafe_self_oscillate and config.unsafe_loop_gain <= 1.0:
        msg = "--unsafe-loop-gain must be > 1.0 when --unsafe-self-oscillate is enabled."
        raise typer.BadParameter(msg)
    if (
        config.conv_route_start is not None or config.conv_route_end is not None
    ) and not conv_enabled:
        msg = "--conv-route-start/--conv-route-end are only valid for convolution renders."
        raise typer.BadParameter(msg)

    if config.ir_route_map != "auto" and not conv_enabled:
        msg = "--ir-route-map is only valid for convolution render workflows."
        raise typer.BadParameter(msg)

    if (
        config.ir is not None
        and (config.engine == "conv" or config.engine == "auto")
        and not config.self_convolve
    ):
        try:
            in_info = sf.info(str(infile))
            ir_info = sf.info(str(config.ir))
            in_channels = int(in_info.channels)
            ir_channels = int(ir_info.channels)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        effective_in_channels = in_channels
        if config.ambi_order > 0:
            effective_in_channels = ambisonic_channel_count(int(config.ambi_order))
            if (
                ir_channels not in {1, effective_in_channels}
                and (ir_channels % max(1, effective_in_channels)) != 0
            ):
                msg = (
                    f"IR channel layout ({ir_channels}) is incompatible with Ambisonics order "
                    f"{config.ambi_order} ({effective_in_channels} channels)."
                )
                raise typer.BadParameter(msg)

        if (
            config.ir_route_map == "auto"
            and config.output_layout == "auto"
            and effective_in_channels > 0
            and ir_channels > effective_in_channels
            and ir_channels % effective_in_channels == 0
        ):
            resolved_out_channels = ir_channels // effective_in_channels
            msg = (
                "Ambiguous matrix-packed IR layout detected. "
                f"Input channels={effective_in_channels}, IR channels={ir_channels}, "
                f"resolved output channels={resolved_out_channels}. "
                "Set --ir-route-map full and/or set --output-layout explicitly "
                "(for example: --output-layout 7.1.2 --ir-route-map full)."
            )
            raise typer.BadParameter(msg)

        output_layout_name = str(config.output_layout).strip().lower()
        output_layout_channels = _LAYOUT_CHANNELS.get(output_layout_name)
        if (
            config.ir_route_map == "auto"
            and output_layout_channels is not None
            and output_layout_channels >= 16
            and ir_channels in {1, effective_in_channels}
        ):
            msg = (
                "Auto route-map is ambiguous for large output layouts when IR channels are "
                "mono or equal to input channels. "
                f"Output layout={output_layout_name}, input_channels={effective_in_channels}, "
                f"ir_channels={ir_channels}. "
                "Set --ir-route-map explicitly (recommended: broadcast for mono/matched IR, "
                "full for matrix-packed IR)."
            )
            raise typer.BadParameter(msg)

    if config.wet == 0.0 and config.dry == 0.0:
        msg = "At least one of --wet or --dry must be non-zero."
        raise typer.BadParameter(msg)

    if config.allpass_stages == 0 and len(config.allpass_delays_ms) > 0:
        msg = "--allpass-delays-ms cannot be used when --allpass-stages is 0."
        raise typer.BadParameter(msg)
    if config.allpass_stages == 0 and len(config.allpass_gains) > 0:
        msg = "--allpass-gain list cannot be used when --allpass-stages is 0."
        raise typer.BadParameter(msg)
    if len(config.allpass_gains) > 0 and len(config.allpass_gains) != config.allpass_stages:
        msg = (
            "When using comma-separated --allpass-gain values, provide exactly "
            f"{config.allpass_stages} entries (got {len(config.allpass_gains)})."
        )
        raise typer.BadParameter(msg)
    if len(config.comb_delays_ms) > 64:
        msg = "--comb-delays-ms supports at most 64 entries."
        raise typer.BadParameter(msg)
    if len(config.allpass_delays_ms) > 128:
        msg = "--allpass-delays-ms supports at most 128 entries."
        raise typer.BadParameter(msg)
    if len(config.fdn_dfm_delays_ms) > 64:
        msg = "--fdn-dfm-delays-ms supports at most 64 entries."
        raise typer.BadParameter(msg)

    _validate_fdn_matrix_name(config.fdn_matrix)
    _validate_fdn_tv_settings(
        fdn_matrix=config.fdn_matrix,
        fdn_tv_rate_hz=config.fdn_tv_rate_hz,
        fdn_tv_depth=config.fdn_tv_depth,
    )
    _validate_fdn_sparse_settings(
        fdn_matrix=config.fdn_matrix,
        fdn_sparse=config.fdn_sparse,
        fdn_sparse_degree=config.fdn_sparse_degree,
    )
    _validate_fdn_graph_settings(
        fdn_matrix=config.fdn_matrix,
        fdn_graph_topology=config.fdn_graph_topology,
        fdn_graph_degree=config.fdn_graph_degree,
    )
    _validate_fdn_multiband_settings(
        fdn_rt60_low=config.fdn_rt60_low,
        fdn_rt60_mid=config.fdn_rt60_mid,
        fdn_rt60_high=config.fdn_rt60_high,
        fdn_xover_low_hz=config.fdn_xover_low_hz,
        fdn_xover_high_hz=config.fdn_xover_high_hz,
    )
    _validate_fdn_link_filter_settings(
        fdn_link_filter=config.fdn_link_filter,
        fdn_link_filter_hz=config.fdn_link_filter_hz,
        fdn_link_filter_mix=config.fdn_link_filter_mix,
    )
    _validate_fdn_spatial_coupling_settings(
        fdn_spatial_coupling_mode=config.fdn_spatial_coupling_mode,
        fdn_spatial_coupling_strength=config.fdn_spatial_coupling_strength,
    )
    _validate_fdn_nonlinearity_settings(
        fdn_nonlinearity=config.fdn_nonlinearity,
        fdn_nonlinearity_amount=config.fdn_nonlinearity_amount,
        fdn_nonlinearity_drive=config.fdn_nonlinearity_drive,
    )
    _validate_perceptual_macro_settings(
        fdn_rt60_tilt=config.fdn_rt60_tilt,
        room_size_macro=config.room_size_macro,
        clarity_macro=config.clarity_macro,
        warmth_macro=config.warmth_macro,
        envelopment_macro=config.envelopment_macro,
    )
    _validate_fdn_tonal_correction_settings(
        fdn_tonal_correction_strength=config.fdn_tonal_correction_strength,
    )
    resolved_fdn_lines = (
        len(config.comb_delays_ms) if len(config.comb_delays_ms) > 0 else int(config.fdn_lines)
    )
    _validate_fdn_cascade_settings(
        fdn_lines=resolved_fdn_lines,
        fdn_cascade=config.fdn_cascade,
        fdn_cascade_mix=config.fdn_cascade_mix,
        fdn_cascade_delay_scale=config.fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=config.fdn_cascade_rt60_ratio,
    )
    if len(config.fdn_dfm_delays_ms) not in {0, 1, resolved_fdn_lines}:
        msg = (
            "--fdn-dfm-delays-ms must include either 1 value or exactly "
            f"{resolved_fdn_lines} values."
        )
        raise typer.BadParameter(msg)

    if config.freeze:
        if config.start is None or config.end is None:
            msg = "--freeze requires both --start and --end."
            raise typer.BadParameter(msg)
        if config.end <= config.start:
            msg = "--end must be greater than --start when --freeze is enabled."
            raise typer.BadParameter(msg)
    elif config.start is not None or config.end is not None:
        msg = "--start/--end are only valid when --freeze is enabled."
        raise typer.BadParameter(msg)

    if config.output_peak_norm == "target" and config.output_peak_target_dbfs is None:
        msg = "--output-peak-norm target requires --output-peak-target-dbfs."
        raise typer.BadParameter(msg)
    if config.output_peak_norm != "target" and config.output_peak_target_dbfs is not None:
        msg = "--output-peak-target-dbfs is only valid with --output-peak-norm target."
        raise typer.BadParameter(msg)

    if config.ir_gen and config.ir is not None:
        msg = "Use either --ir or --ir-gen, not both."
        raise typer.BadParameter(msg)

    if config.mod_min >= config.mod_max:
        msg = "--mod-min must be less than --mod-max."
        raise typer.BadParameter(msg)

    if config.mod_target == "none" and len(config.mod_sources) > 0:
        msg = "--mod-source requires --mod-target."
        raise typer.BadParameter(msg)
    if config.mod_target != "none" and len(config.mod_sources) == 0:
        msg = "--mod-target requires at least one --mod-source."
        raise typer.BadParameter(msg)

    if config.mod_target in {"mix", "wet"}:
        if config.mod_min < 0.0 or config.mod_max > 1.0:
            msg = "For --mod-target mix/wet, use --mod-min/--mod-max in [0.0, 1.0]."
            raise typer.BadParameter(msg)

    try:
        parse_mod_sources(config.mod_sources)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    for route_spec in config.mod_routes:
        try:
            parse_mod_route_spec(route_spec)
        except ValueError as exc:
            raise typer.BadParameter(f"invalid --mod-route '{route_spec}': {exc}") from exc


def _validate_analyze_call(infile: Path, json_out: Path | None, frames_out: Path | None) -> None:
    """Validate analyze command output paths."""
    if json_out is not None and infile.resolve() == json_out.resolve():
        msg = "--json-out must be different from input file."
        raise typer.BadParameter(msg)
    if frames_out is not None and infile.resolve() == frames_out.resolve():
        msg = "--frames-out must be different from input file."
        raise typer.BadParameter(msg)


def _validate_ir_gen_call(
    out_ir: Path,
    out_format: IRFileFormat,
    rt60: float | None,
    rt60_low: float | None,
    rt60_high: float | None,
    modal_q_min: float,
    modal_q_max: float,
    modal_low_hz: float,
    modal_high_hz: float,
    resonator_q_min: float,
    resonator_q_max: float,
    resonator_low_hz: float,
    resonator_high_hz: float,
    fdn_lines: int,
    fdn_matrix: str,
    fdn_tv_rate_hz: float,
    fdn_tv_depth: float,
    fdn_sparse: bool,
    fdn_sparse_degree: int,
    fdn_cascade: bool,
    fdn_cascade_mix: float,
    fdn_cascade_delay_scale: float,
    fdn_cascade_rt60_ratio: float,
    fdn_rt60_low: float | None,
    fdn_rt60_mid: float | None,
    fdn_rt60_high: float | None,
    fdn_rt60_tilt: float,
    fdn_tonal_correction_strength: float,
    fdn_xover_low_hz: float,
    fdn_xover_high_hz: float,
    fdn_link_filter: str,
    fdn_link_filter_hz: float,
    fdn_link_filter_mix: float,
    fdn_graph_topology: str,
    fdn_graph_degree: int,
    fdn_spatial_coupling_mode: str,
    fdn_spatial_coupling_strength: float,
    fdn_nonlinearity: str,
    fdn_nonlinearity_amount: float,
    fdn_nonlinearity_drive: float,
    room_size_macro: float,
    clarity_macro: float,
    warmth_macro: float,
    envelopment_macro: float,
) -> None:
    """Validate IR generation options and output path constraints."""
    resolved = _resolve_ir_output_path(out_ir, out_format)
    _validate_output_audio_path(resolved, "auto")

    if rt60 is not None and (rt60_low is not None or rt60_high is not None):
        msg = "Use either --rt60 or --rt60-low/--rt60-high, not both."
        raise typer.BadParameter(msg)
    if (rt60_low is None) != (rt60_high is None):
        msg = "Both --rt60-low and --rt60-high must be provided together."
        raise typer.BadParameter(msg)
    if rt60_low is not None and rt60_high is not None and rt60_low > rt60_high:
        msg = "--rt60-low must be <= --rt60-high."
        raise typer.BadParameter(msg)
    if modal_q_min > modal_q_max:
        msg = "--modal-q-min must be <= --modal-q-max."
        raise typer.BadParameter(msg)
    if modal_low_hz >= modal_high_hz:
        msg = "--modal-low-hz must be < --modal-high-hz."
        raise typer.BadParameter(msg)
    if resonator_q_min > resonator_q_max:
        msg = "--resonator-q-min must be <= --resonator-q-max."
        raise typer.BadParameter(msg)
    if resonator_low_hz >= resonator_high_hz:
        msg = "--resonator-low-hz must be < --resonator-high-hz."
        raise typer.BadParameter(msg)

    _validate_fdn_matrix_name(fdn_matrix)
    _validate_fdn_tv_settings(
        fdn_matrix=fdn_matrix,
        fdn_tv_rate_hz=fdn_tv_rate_hz,
        fdn_tv_depth=fdn_tv_depth,
    )
    _validate_fdn_sparse_settings(
        fdn_matrix=fdn_matrix,
        fdn_sparse=fdn_sparse,
        fdn_sparse_degree=fdn_sparse_degree,
    )
    _validate_fdn_graph_settings(
        fdn_matrix=fdn_matrix,
        fdn_graph_topology=fdn_graph_topology,
        fdn_graph_degree=fdn_graph_degree,
    )
    _validate_fdn_cascade_settings(
        fdn_lines=fdn_lines,
        fdn_cascade=fdn_cascade,
        fdn_cascade_mix=fdn_cascade_mix,
        fdn_cascade_delay_scale=fdn_cascade_delay_scale,
        fdn_cascade_rt60_ratio=fdn_cascade_rt60_ratio,
    )
    _validate_fdn_multiband_settings(
        fdn_rt60_low=fdn_rt60_low,
        fdn_rt60_mid=fdn_rt60_mid,
        fdn_rt60_high=fdn_rt60_high,
        fdn_xover_low_hz=fdn_xover_low_hz,
        fdn_xover_high_hz=fdn_xover_high_hz,
    )
    _validate_fdn_link_filter_settings(
        fdn_link_filter=fdn_link_filter,
        fdn_link_filter_hz=fdn_link_filter_hz,
        fdn_link_filter_mix=fdn_link_filter_mix,
    )
    _validate_fdn_spatial_coupling_settings(
        fdn_spatial_coupling_mode=fdn_spatial_coupling_mode,
        fdn_spatial_coupling_strength=fdn_spatial_coupling_strength,
    )
    _validate_fdn_nonlinearity_settings(
        fdn_nonlinearity=fdn_nonlinearity,
        fdn_nonlinearity_amount=fdn_nonlinearity_amount,
        fdn_nonlinearity_drive=fdn_nonlinearity_drive,
    )
    _validate_perceptual_macro_settings(
        fdn_rt60_tilt=fdn_rt60_tilt,
        room_size_macro=room_size_macro,
        clarity_macro=clarity_macro,
        warmth_macro=warmth_macro,
        envelopment_macro=envelopment_macro,
    )
    _validate_fdn_tonal_correction_settings(
        fdn_tonal_correction_strength=fdn_tonal_correction_strength,
    )


def _validate_ir_morph_sweep_call(
    *,
    ir_a: Path,
    ir_b: Path,
    out_dir: Path,
    mode: str,
    early_alpha: float | None,
    late_alpha: float | None,
    mismatch_policy: IRMorphMismatchPolicy,
    cache_dir: str,
    out_prefix: str,
    alpha_points: list[float] | None,
    alpha_steps: int,
    checkpoint_file: Path | None,
    resume: bool,
) -> None:
    """Validate IR morph-sweep command inputs."""
    _ensure_distinct_paths(ir_a, ir_b, "IR_A", "IR_B")
    out_resolved = out_dir.resolve()
    if out_resolved in {ir_a.resolve(), ir_b.resolve()}:
        raise typer.BadParameter("OUT_DIR must be different from IR input file paths.")

    try:
        validate_ir_morph_mode_name(mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        validate_ir_morph_mismatch_policy_name(mismatch_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if early_alpha is not None and not (0.0 <= float(early_alpha) <= 1.0):
        raise typer.BadParameter("--early-alpha must be in [0.0, 1.0].")
    if late_alpha is not None and not (0.0 <= float(late_alpha) <= 1.0):
        raise typer.BadParameter("--late-alpha must be in [0.0, 1.0].")
    if str(cache_dir).strip() == "":
        raise typer.BadParameter("--cache-dir must not be empty.")
    if str(out_prefix).strip() == "":
        raise typer.BadParameter("--out-prefix must not be empty.")
    points = [] if alpha_points is None else alpha_points
    if len(points) == 0 and int(alpha_steps) < 2:
        raise typer.BadParameter("--alpha-steps must be >= 2 when --alpha is not provided.")
    if resume and checkpoint_file is None:
        raise typer.BadParameter("--resume requires --checkpoint-file.")


def _resolve_ir_morph_sweep_alphas(
    *,
    alpha_points: list[float] | None,
    alpha_start: float,
    alpha_end: float,
    alpha_steps: int,
) -> tuple[float, ...]:
    """Resolve alpha timeline values for morph-sweep execution."""
    if alpha_points is not None and len(alpha_points) > 0:
        return tuple(float(np.clip(float(value), 0.0, 1.0)) for value in alpha_points)
    if int(alpha_steps) <= 1:
        return (float(np.clip(alpha_start, 0.0, 1.0)),)
    values = np.linspace(float(alpha_start), float(alpha_end), int(alpha_steps), dtype=np.float64)
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in values.tolist())


def _alpha_token(alpha: float) -> str:
    """Return filename-safe alpha token."""
    token = f"{float(alpha):.3f}".replace("-", "m").replace(".", "p")
    return f"a{token}"


def _upsert_checkpoint_row(rows: list[Any], row: dict[str, Any]) -> None:
    """Insert or replace checkpoint result row by outfile."""
    key = str(row.get("outfile", ""))
    for idx, existing in enumerate(rows):
        if not isinstance(existing, dict):
            continue
        if str(existing.get("outfile", "")) != key:
            continue
        rows[idx] = row
        return
    rows.append(row)


def _summarize_numeric_column(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    """Return min/max/mean summary for a numeric QA column."""
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        values.append(value)
    if len(values) == 0:
        return None
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def _validate_ir_morph_call(
    *,
    ir_a: Path,
    ir_b: Path,
    out_ir: Path,
    mode: str,
    early_alpha: float | None,
    late_alpha: float | None,
    mismatch_policy: IRMorphMismatchPolicy,
    cache_dir: str,
) -> None:
    """Validate IR morph command inputs."""
    _ensure_distinct_paths(ir_a, ir_b, "IR_A", "IR_B")
    _ensure_distinct_paths(ir_a, out_ir, "IR_A", "OUT_IR")
    _ensure_distinct_paths(ir_b, out_ir, "IR_B", "OUT_IR")
    _validate_output_audio_path(out_ir, "auto")

    try:
        validate_ir_morph_mode_name(mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        validate_ir_morph_mismatch_policy_name(mismatch_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if early_alpha is not None and not (0.0 <= float(early_alpha) <= 1.0):
        msg = "--early-alpha must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)
    if late_alpha is not None and not (0.0 <= float(late_alpha) <= 1.0):
        msg = "--late-alpha must be in [0.0, 1.0]."
        raise typer.BadParameter(msg)

    if str(cache_dir).strip() == "":
        msg = "--cache-dir must not be empty."
        raise typer.BadParameter(msg)


def _validate_ir_process_call(in_ir: Path, out_ir: Path) -> None:
    """Validate IR process command path arguments."""
    _ensure_distinct_paths(in_ir, out_ir, "IN_IR", "OUT_IR")
    _validate_output_audio_path(out_ir, "auto")


def _validate_ir_analyze_call(ir_file: Path, json_out: Path | None) -> None:
    """Validate IR analyze optional output path."""
    if json_out is not None and ir_file.resolve() == json_out.resolve():
        msg = "--json-out must be different from input IR file."
        raise typer.BadParameter(msg)


def _validate_batch_job_paths(infile: Path, outfile: Path, idx: int) -> None:
    """Validate one batch job's input/output paths."""
    if infile.resolve() == outfile.resolve():
        msg = f"jobs[{idx - 1}] infile and outfile must be different."
        raise typer.BadParameter(msg)
    if not infile.exists():
        msg = f"jobs[{idx - 1}] infile not found: {infile}"
        raise typer.BadParameter(msg)
    _validate_output_audio_path(outfile, "auto")


def _ensure_distinct_paths(in_path: Path, out_path: Path, in_label: str, out_label: str) -> None:
    """Ensure input and output paths are not identical."""
    if in_path.resolve() == out_path.resolve():
        msg = f"{in_label} and {out_label} must be different paths."
        raise typer.BadParameter(msg)


def _validate_output_audio_path(path: Path, out_subtype_mode: str) -> None:
    """Validate output extension and requested SoundFile subtype support."""
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "":
        msg = f"Output path must include an audio file extension: {path} (try .wav or .flac)."
        raise typer.BadParameter(msg)

    format_map = {
        "wav": "WAV",
        "flac": "FLAC",
        "aif": "AIFF",
        "aiff": "AIFF",
        "ogg": "OGG",
        "caf": "CAF",
        "au": "AU",
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        supported = ", ".join(f".{ext}" for ext in sorted(format_map))
        suggestion = _did_you_mean(suffix, set(format_map.keys()))
        if suggestion is None:
            for ext in sorted(format_map.keys()):
                if suffix.startswith(ext) or ext.startswith(suffix):
                    suggestion = ext
                    break
        if suggestion is not None:
            msg = (
                f"Unsupported output audio extension: .{suffix}. "
                f"Did you mean '.{suggestion}'? Supported: {supported}."
            )
        else:
            msg = f"Unsupported output audio extension: .{suffix}. Supported: {supported}."
        raise typer.BadParameter(msg)

    subtype_map = {
        "auto": None,
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
    }
    subtype = subtype_map.get(out_subtype_mode)
    if out_subtype_mode not in subtype_map:
        msg = f"Unsupported --out-subtype value: {out_subtype_mode}"
        raise typer.BadParameter(msg)

    if subtype is None:
        if not sf.check_format(fmt):
            msg = f"SoundFile cannot write format '{fmt}' for output path {path}"
            raise typer.BadParameter(msg)
    else:
        if not sf.check_format(fmt, subtype):
            supported_subtypes: list[str] = []
            for mode, candidate in subtype_map.items():
                if candidate is None:
                    continue
                if sf.check_format(fmt, candidate):
                    supported_subtypes.append(mode)
            supported_text = ", ".join(sorted(supported_subtypes))
            msg = (
                f"Subtype '{subtype}' is not supported for format '{fmt}'. "
                f"Use --out-subtype auto or one of: {supported_text}."
            )
            raise typer.BadParameter(msg)


if __name__ == "__main__":
    app()
