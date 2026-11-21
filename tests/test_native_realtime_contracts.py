"""Signal and lifecycle contracts for the native realtime callback, without JUCE."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def realtime_contract_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which(os.environ.get("CC", "cc"))
    assert compiler is not None, "A C compiler is required for native realtime contracts"
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("native-realtime") / "contracts"
    flags = ["-O1", "-g", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    if os.environ.get("VERBX_NATIVE_SANITIZERS") == "1":
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    subprocess.run(
        [
            compiler,
            "-std=c11",
            *flags,
            "-I",
            str(root / "native/verbx_c/include"),
            str(root / "tests/native/realtime_contracts.c"),
            str(root / "native/verbx_c/src/plugin_realtime.c"),
            str(root / "native/verbx_c/src/plugin_params.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


@pytest.mark.parametrize("quality", range(4), ids=["host", "2x", "4x", "target192k"])
@pytest.mark.parametrize("model", range(3), ids=["algorithmic", "spring", "plate"])
def test_impulse_reset_and_block_partition_contract(
    realtime_contract_executable: Path,
    model: int,
    quality: int,
) -> None:
    result = subprocess.run(
        [str(realtime_contract_executable), str(model), str(quality)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_extreme_parameter_automation_remains_finite(
    realtime_contract_executable: Path,
) -> None:
    result = subprocess.run(
        [str(realtime_contract_executable), "automation"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
