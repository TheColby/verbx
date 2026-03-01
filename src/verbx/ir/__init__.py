"""Impulse-response synthesis and caching package.

Exports the high-level generation config and cache-aware entrypoints used by
both CLI commands and render pipeline integration.
"""

from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir, write_ir_artifacts

__all__ = ["IRGenConfig", "generate_or_load_cached_ir", "write_ir_artifacts"]
