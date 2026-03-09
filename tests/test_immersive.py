from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from verbx.cli import app
from verbx.core.immersive import (
    ImmersiveQCGates,
    QueueJobClaim,
    QueueWorkerConfig,
    evaluate_immersive_qc,
    fold_down_to_stereo,
    generate_immersive_handoff_package,
    run_file_queue_worker,
    summarize_file_queue,
)

runner = CliRunner()


def test_fold_down_to_stereo_for_5p1_shape() -> None:
    audio = np.zeros((256, 6), dtype=np.float32)
    audio[:, 0] = 0.8
    audio[:, 1] = -0.5
    folded = fold_down_to_stereo(audio, layout="5.1")
    assert folded.shape == (256, 2)


def test_evaluate_immersive_qc_returns_gate_payload() -> None:
    audio = np.zeros((1024, 2), dtype=np.float32)
    audio[20:200, 0] = 0.5
    audio[20:200, 1] = -0.4
    gates = ImmersiveQCGates(
        target_lufs=-24.0,
        lufs_tolerance=40.0,
        max_true_peak_dbfs=0.0,
        max_fold_down_delta_db=20.0,
        min_channel_occupancy=0.0,
    )
    report = evaluate_immersive_qc(audio=audio, sr=48_000, label="stereo_test", gates=gates)
    assert "passes" in report
    assert "metrics" in report
    assert report["channels"] == 2


def test_generate_immersive_handoff_package_writes_outputs(tmp_path: Path) -> None:
    bed = np.zeros((4096, 6), dtype=np.float32)
    for ch in range(6):
        bed[120 : 120 + (100 + (ch * 10)), ch] = 0.25 + (0.05 * ch)
    bed_path = tmp_path / "bed.wav"
    sf.write(str(bed_path), bed, 48_000)

    obj = np.zeros((2048, 1), dtype=np.float32)
    obj[60:260, 0] = 0.35
    obj_path = tmp_path / "obj.wav"
    sf.write(str(obj_path), obj, 48_000)

    scene = {
        "scene_name": "scene_alpha",
        "bed": {
            "name": "bed_main",
            "path": str(bed_path),
            "layout": "5.1",
            "render_options": {"wet": 0.7, "rt60": 3.0},
        },
        "objects": [
            {
                "id": "obj_001",
                "name": "lead",
                "path": str(obj_path),
                "layout": "mono",
                "render_options": {"wet": 0.45, "rt60": 2.0},
            }
        ],
        "policy": {"mode": "balanced"},
    }
    out_dir = tmp_path / "handoff"
    summary = generate_immersive_handoff_package(scene=scene, out_dir=out_dir, strict=False)
    outputs = summary["outputs"]
    assert Path(outputs["deliverables_manifest"]).exists()
    assert Path(outputs["adm_sidecar"]).exists()
    assert Path(outputs["object_stem_manifest"]).exists()
    assert Path(outputs["qa_bundle"]).exists()


def test_run_file_queue_worker_processes_job(tmp_path: Path) -> None:
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    audio = np.zeros((1024, 1), dtype=np.float32)
    audio[0, 0] = 0.5
    sf.write(str(infile), audio, 16_000)

    queue_file = tmp_path / "queue.json"
    queue_payload = {
        "version": "0.7",
        "backend": "file",
        "jobs": [
            {
                "id": "job_1",
                "infile": str(infile),
                "outfile": str(outfile),
                "options": {},
                "max_retries": 0,
            }
        ],
    }
    queue_file.write_text(json.dumps(queue_payload), encoding="utf-8")

    def runner_fn(job: QueueJobClaim) -> None:
        in_path = Path(str(job["infile"]))
        out_path = Path(str(job["outfile"]))
        x, sr = sf.read(str(in_path), always_2d=True, dtype="float32")
        sf.write(str(out_path), x, sr)

    config = QueueWorkerConfig(
        worker_id="test_worker",
        heartbeat_dir=tmp_path / "heartbeats",
        poll_ms=50,
        max_jobs=1,
    )
    summary = run_file_queue_worker(queue_path=queue_file, runner=runner_fn, config=config)
    assert int(summary["success"]) == 1
    assert outfile.exists()

    status = summarize_file_queue(queue_file)
    assert int(status["success_jobs"]) == 1


def test_cli_immersive_qc_and_handoff_and_worker(tmp_path: Path) -> None:
    sr = 16_000
    bed = np.zeros((2048, 6), dtype=np.float32)
    for ch in range(6):
        bed[100 : 300 + (ch * 5), ch] = 0.2 + (0.03 * ch)
    bed_path = tmp_path / "bed.wav"
    sf.write(str(bed_path), bed, sr)

    obj = np.zeros((1536, 1), dtype=np.float32)
    obj[80:280, 0] = 0.4
    obj_path = tmp_path / "obj.wav"
    sf.write(str(obj_path), obj, sr)

    qc_json = tmp_path / "qc.json"
    qc_result = runner.invoke(
        app,
        [
            "immersive",
            "qc",
            str(bed_path),
            "--layout",
            "5.1",
            "--json-out",
            str(qc_json),
        ],
    )
    assert qc_result.exit_code == 0, qc_result.stdout
    assert qc_json.exists()

    scene = {
        "scene_name": "scene_cli",
        "bed": {
            "name": "bed_main",
            "path": str(bed_path),
            "layout": "5.1",
            "render_options": {"wet": 0.7, "rt60": 3.2},
        },
        "objects": [
            {
                "id": "obj_1",
                "name": "obj_a",
                "path": str(obj_path),
                "layout": "mono",
                "render_options": {"wet": 0.4, "rt60": 2.5},
            }
        ],
        "policy": {"mode": "balanced"},
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    out_dir = tmp_path / "handoff_cli"
    handoff_result = runner.invoke(
        app,
        ["immersive", "handoff", str(scene_path), str(out_dir), "--warn-only"],
    )
    assert handoff_result.exit_code == 0, handoff_result.stdout
    assert len(list(out_dir.glob("*.json"))) >= 3

    in_render = tmp_path / "render_in.wav"
    out_render = tmp_path / "render_out.wav"
    tone = np.zeros((1024, 1), dtype=np.float32)
    tone[40:160, 0] = 0.6
    sf.write(str(in_render), tone, sr)
    queue_payload = {
        "version": "0.7",
        "backend": "file",
        "jobs": [
            {
                "id": "render_job_1",
                "infile": str(in_render),
                "outfile": str(out_render),
                "max_retries": 0,
                "options": {"engine": "algo", "rt60": 1.0, "repeat": 1, "progress": False},
            }
        ],
    }
    queue_path = tmp_path / "queue_cli.json"
    queue_path.write_text(json.dumps(queue_payload), encoding="utf-8")

    worker_result = runner.invoke(
        app,
        [
            "immersive",
            "queue",
            "worker",
            str(queue_path),
            "--worker-id",
            "cli_worker",
            "--heartbeat-dir",
            str(tmp_path / "hb"),
            "--max-jobs",
            "1",
            "--poll-ms",
            "50",
        ],
    )
    assert worker_result.exit_code == 0, worker_result.stdout
    assert out_render.exists()
    assert (tmp_path / "hb" / "cli_worker.heartbeat.json").exists()


def test_immersive_handoff_strict_fails_on_any_qc_violation(tmp_path: Path) -> None:
    bed = np.zeros((2048, 2), dtype=np.float32)
    bed[32:640, 0] = 0.5
    bed[32:640, 1] = -0.45
    bed_path = tmp_path / "bed_qc_fail.wav"
    sf.write(str(bed_path), bed, 48_000)

    scene = {
        "scene_name": "scene_qc_fail",
        "bed": {
            "name": "bed_main",
            "path": str(bed_path),
            "layout": "stereo",
            "render_options": {"wet": 0.6, "rt60": 2.5},
        },
        "policy": {"mode": "balanced"},
        "qc_gates": {
            "target_lufs": -18.0,
            "lufs_tolerance": 30.0,
            "max_true_peak_dbfs": -80.0,
            "max_fold_down_delta_db": 20.0,
            "min_channel_occupancy": 0.0,
            "occupancy_threshold_dbfs": -90.0,
        },
    }

    with pytest.raises(ValueError, match="one or more QC gates failed"):
        generate_immersive_handoff_package(
            scene=scene,
            out_dir=tmp_path / "handoff_strict_qc",
            strict=True,
        )


def test_immersive_handoff_validates_sample_rate_consistency(tmp_path: Path) -> None:
    bed = np.zeros((2048, 2), dtype=np.float32)
    bed[48:300, 0] = 0.2
    bed[48:300, 1] = -0.2
    bed_path = tmp_path / "bed_sr.wav"
    sf.write(str(bed_path), bed, 48_000)

    obj = np.zeros((1024, 1), dtype=np.float32)
    obj[20:220, 0] = 0.25
    obj_path = tmp_path / "obj_sr.wav"
    sf.write(str(obj_path), obj, 44_100)

    scene = {
        "scene_name": "scene_sr_mismatch",
        "sample_rate": 48_000,
        "bed": {
            "name": "bed_main",
            "path": str(bed_path),
            "layout": "stereo",
            "render_options": {"wet": 0.6, "rt60": 2.5},
        },
        "objects": [
            {
                "id": "obj_1",
                "name": "obj_sr",
                "path": str(obj_path),
                "layout": "mono",
                "render_options": {"wet": 0.4, "rt60": 1.5},
            }
        ],
        "policy": {"mode": "balanced"},
    }

    warn_summary = generate_immersive_handoff_package(
        scene=scene,
        out_dir=tmp_path / "handoff_warn_sr",
        strict=False,
    )
    validation = warn_summary.get("validation", {})
    assert isinstance(validation, dict)
    errors = validation.get("errors", [])
    assert isinstance(errors, list)
    assert any("obj_sr: sample rate 44100" in str(item) for item in errors)

    with pytest.raises(ValueError, match="sample rate 44100"):
        generate_immersive_handoff_package(
            scene=scene,
            out_dir=tmp_path / "handoff_strict_sr",
            strict=True,
        )


def test_queue_summary_rejects_duplicate_job_ids(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float32)
    infile = tmp_path / "in_dup.wav"
    sf.write(str(infile), audio, 16_000)

    queue_file = tmp_path / "queue_duplicate_ids.json"
    queue_file.write_text(
        json.dumps(
            {
                "version": "0.7",
                "backend": "file",
                "jobs": [
                    {
                        "id": "job_dup",
                        "infile": str(infile),
                        "outfile": str(tmp_path / "out1.wav"),
                        "options": {},
                    },
                    {
                        "id": "job_dup",
                        "infile": str(infile),
                        "outfile": str(tmp_path / "out2.wav"),
                        "options": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate queue job id"):
        summarize_file_queue(queue_file)
