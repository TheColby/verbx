"""System, onboarding, and diagnostic CLI commands."""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from verbx import __version__
from verbx.config import RenderConfig
from verbx.core.accel import (
    cuda_available,
    is_apple_silicon,
    resolve_device,
    resolve_device_for_engine,
)
from verbx.core.pipeline import run_render_pipeline

console = Console()
progress_console = Console(force_terminal=True, color_system="truecolor")


@contextmanager
def _processing_status(description: str, *, enabled: bool = True) -> Any:
    """Render a compact single-task status indicator."""
    progress = Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(
            bar_width=24,
            complete_style="bright_green",
            finished_style="green",
            pulse_style="bright_blue",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=progress_console,
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


def version() -> None:
    """Print CLI/package version."""
    console.print(f"verbx {__version__}")


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
    json_out: Annotated[Path | None, typer.Option(
        "--json-out",
        resolve_path=True,
        help="Optional path to write quickstart verification/smoke JSON.",
    )] = None,
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Run a tiny end-to-end render smoke test with synthetic input audio.",
    ),
    smoke_out_dir: Annotated[Path | None, typer.Option(
        "--smoke-out-dir",
        resolve_path=True,
        help="Optional output directory for smoke-test artifacts.",
    )] = None,
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
        report = collect_runtime_diagnostics()
        print_runtime_checks_table(report, title="verbx Quickstart Verify")
    if smoke_test:
        with _processing_status("Quickstart render smoke test"):
            smoke_report = run_render_smoke_test(out_dir=smoke_out_dir)
        print_render_smoke_test_table(smoke_report, title="verbx Quickstart Smoke Test")

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
        write_json_atomic(json_out.resolve(), cast(dict[str, Any], payload))

    if strict:
        verify_ok = True if report is None else bool(report.get("ready", False))
        smoke_ok = True if smoke_report is None else bool(smoke_report.get("ok", False))
        if not verify_ok or not smoke_ok:
            raise typer.Exit(code=2)


def doctor(
    json_out: Annotated[Path | None, typer.Option(
        "--json-out",
        resolve_path=True,
        help="Optional path to write machine-readable diagnostics JSON.",
    )] = None,
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
    smoke_out_dir: Annotated[Path | None, typer.Option(
        "--smoke-out-dir",
        resolve_path=True,
        help="Optional output directory for doctor smoke-test artifacts.",
    )] = None,
) -> None:
    """Print runtime diagnostics for launch-day troubleshooting."""
    report = collect_runtime_diagnostics()

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
    print_runtime_checks_table(report, title="verbx Doctor Checks")
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
            smoke_report = run_render_smoke_test(out_dir=smoke_out_dir)
        print_render_smoke_test_table(smoke_report, title="verbx Doctor Smoke Test")

    if json_out is not None:
        if smoke_report is None:
            write_json_atomic(json_out.resolve(), report)
        else:
            payload = dict(report)
            payload["render_smoke_test"] = smoke_report
            payload["ready"] = bool(payload.get("ready", False) and smoke_report.get("ok", False))
            write_json_atomic(json_out.resolve(), payload)
    if strict and (
        int(report.get("failed_checks", 0)) > 0
        or (smoke_report is not None and not bool(smoke_report.get("ok", False)))
    ):
        raise typer.Exit(code=2)


def dependency_versions() -> dict[str, str | None]:
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


def collect_runtime_diagnostics() -> dict[str, Any]:
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
    report["dependencies"] = dependency_versions()
    checks = runtime_checks(report)
    failed_checks = [item for item in checks if not bool(item.get("ok", False))]
    report["checks"] = checks
    report["checks_total"] = len(checks)
    report["failed_checks"] = len(failed_checks)
    report["issues"] = failed_checks
    report["ready"] = len(failed_checks) == 0
    report["status"] = "ok" if len(failed_checks) == 0 else "warn"
    report["recommendations"] = runtime_recommendations(report)
    return report


def runtime_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
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


def runtime_recommendations(report: dict[str, Any]) -> list[str]:
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


def run_render_smoke_test(*, out_dir: Path | None) -> dict[str, Any]:
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


def print_render_smoke_test_table(report: dict[str, Any], *, title: str) -> None:
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


def print_runtime_checks_table(report: dict[str, Any], *, title: str) -> None:
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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
