"""Logging setup for verbx CLI."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(verbose: bool) -> None:
    """Configure root logging with Rich output.

    ``verbose=False`` keeps console noise low for long renders while preserving
    warnings and critical runtime diagnostics.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        force=True,
    )
