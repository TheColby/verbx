from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from verbx.cli import app

runner = CliRunner()


def _write_minimal_sofa(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.attrs["Conventions"] = "SOFA"
        handle.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        # (M, R, N): one measurement, two receivers, 64 samples
        ir = np.zeros((1, 2, 64), dtype=np.float64)
        ir[0, 0, 10] = 1.0
        ir[0, 1, 12] = 1.0
        handle.create_dataset("Data.IR", data=ir)
        handle.create_dataset("Data.SamplingRate", data=np.asarray([48_000.0], dtype=np.float64))


def test_ir_sofa_inspect_emits_json(tmp_path: Path) -> None:
    sofa = tmp_path / "mini.sofa"
    _write_minimal_sofa(sofa)
    out_json = tmp_path / "inspect.json"
    result = runner.invoke(
        app,
        ["ir", "sofa-inspect", str(sofa), "--json-out", str(out_json)],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["attributes"]["Conventions"] == "SOFA"
    assert payload["datasets"]["Data.IR"]["shape"] == [1, 2, 64]


def test_ir_sofa_convert_writes_ir_matrix(tmp_path: Path) -> None:
    sofa = tmp_path / "mini.sofa"
    _write_minimal_sofa(sofa)
    out_ir = tmp_path / "matrix.wav"
    out_json = tmp_path / "convert.json"
    result = runner.invoke(
        app,
        [
            "ir",
            "sofa-convert",
            str(sofa),
            str(out_ir),
            "--measurement-index",
            "0",
            "--json-out",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert out_ir.exists()
    assert int(payload["channels"]) == 2
    assert int(payload["frames"]) == 64
