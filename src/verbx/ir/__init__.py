"""Impulse-response synthesis and caching package.

Exports the high-level generation config and cache-aware entrypoints used by
both CLI commands and render pipeline integration.
"""

from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir, write_ir_artifacts
from verbx.ir.scala import ScalaScale, parse_scala_file, resolve_scala_frequencies

__all__ = [
    "IRGenConfig",
    "ScalaScale",
    "generate_or_load_cached_ir",
    "parse_scala_file",
    "resolve_scala_frequencies",
    "write_ir_artifacts",
]
