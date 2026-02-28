from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from verbx.core import accel


def test_resolve_device_auto_prefers_cuda(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(accel, "is_apple_silicon", lambda: False)
    assert accel.resolve_device("auto") == "cuda"


def test_resolve_device_auto_prefers_mps_when_no_cuda(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    monkeypatch.setattr(accel, "is_apple_silicon", lambda: True)
    assert accel.resolve_device("auto") == "mps"


def test_resolve_device_explicit_cuda_falls_back(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    assert accel.resolve_device("cuda") == "cpu"
