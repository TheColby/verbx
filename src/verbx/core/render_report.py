"""Typed render-report models shared across CLI and pipeline layers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RenderReport(Mapping[str, Any]):
    """Structured render report with a dict-like compatibility surface.

    The CLI and tests still consume reports with ``dict`` semantics today.  This
    model gives the pipeline a typed home without forcing a scary big-bang
    migration across every caller in one go.
    """

    engine: str
    sample_rate: int
    input_samples: int
    output_samples: int
    channels: int
    config: dict[str, Any]
    effective: dict[str, Any]
    ir_runtime: dict[str, Any] | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    analysis_path: str | None = None
    frames_path: str | None = None
    automation_trace_path: str | None = None
    feature_vector_trace_path: str | None = None
    repro_bundle_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable payload for reports and artifacts."""
        payload: dict[str, Any] = {
            "engine": self.engine,
            "sample_rate": int(self.sample_rate),
            "input_samples": int(self.input_samples),
            "output_samples": int(self.output_samples),
            "channels": int(self.channels),
            "config": self.config,
            "effective": self.effective,
        }
        optional_fields = {
            "ir_runtime": self.ir_runtime,
            "input": self.input,
            "output": self.output,
            "analysis_path": self.analysis_path,
            "frames_path": self.frames_path,
            "automation_trace_path": self.automation_trace_path,
            "feature_vector_trace_path": self.feature_vector_trace_path,
            "repro_bundle_path": self.repro_bundle_path,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        """Expose the classic ``dict.get`` interface for legacy call sites."""
        return self.to_dict().get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow narrow legacy mutation during the CLI transition period."""
        field_map = {
            "engine": "engine",
            "sample_rate": "sample_rate",
            "input_samples": "input_samples",
            "output_samples": "output_samples",
            "channels": "channels",
            "config": "config",
            "effective": "effective",
            "ir_runtime": "ir_runtime",
            "input": "input",
            "output": "output",
            "analysis_path": "analysis_path",
            "frames_path": "frames_path",
            "automation_trace_path": "automation_trace_path",
            "feature_vector_trace_path": "feature_vector_trace_path",
            "repro_bundle_path": "repro_bundle_path",
        }
        attr = field_map.get(key)
        if attr is None:
            raise KeyError(key)
        setattr(self, attr, value)

    def items(self):
        """Return item pairs for compatibility with table/JSON helpers."""
        return self.to_dict().items()
