from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_native_scaffold_builds_and_reports_version(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    assert cc is not None, "C compiler 'cc' is required for the native scaffold test."

    repo_root = Path(__file__).resolve().parents[1]
    exe = tmp_path / "verbx-c"
    command = [
        cc,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-I",
        str(repo_root / "native/verbx_c/include"),
        str(repo_root / "native/verbx_c/src/main.c"),
        str(repo_root / "native/verbx_c/src/cli.c"),
        "-o",
        str(exe),
    ]
    subprocess.run(command, check=True, cwd=repo_root)

    result = subprocess.run(
        [str(exe), "version"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "verbx-c 0.8.0-dev"
