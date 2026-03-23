from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_npm_launcher_runs_help() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed in test environment")
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    result = subprocess.run(
        ["node", "npm/verbx.js", "--help"],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in (result.stdout + result.stderr)


def test_npm_launcher_falls_back_when_python_env_is_invalid() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed in test environment")
    env = dict(os.environ)
    env["PYTHON"] = "definitely-not-a-real-python-executable"
    result = subprocess.run(
        ["node", "npm/verbx.js", "--help"],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in (result.stdout + result.stderr)
