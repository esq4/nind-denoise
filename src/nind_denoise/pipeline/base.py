"""Base classes and execution helpers for pipeline operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from nind_denoise.config import Config  # type: ignore[reportMissingImports]
from nind_denoise.config.config import Config

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Per-stage job context with typed, required fields for stage execution.

    This contains the mutable state specific to each pipeline stage execution,
    with non-optional fields to ensure type safety.
    """

    # Input/output paths (required)
    input_path: Path
    output_path: Path

    # Stage-specific processing parameters
    sigma: int = 1
    iterations: int = 10
    quality: int = 90

    # Optional stage-specific paths
    intermediate_path: Optional[Path] = None


class StageError(Exception):
    pass


class Operation(ABC):
    """Abstract base for all operations (export, denoise, deblur)."""

    @abstractmethod
    def describe(self) -> str:  # pragma: no cover - description only
        ...

    @abstractmethod
    def execute_with_env(
        self, cfg: Config, job_ctx: JobContext
    ) -> None:  # pragma: no cover
        """Execute operation with Environment + JobContext pattern."""
        ...

    @abstractmethod
    def verify_with_env(
        self, cfg: Config, job_ctx: JobContext
    ) -> None:  # pragma: no cover
        """Verify operation outputs with Environment + JobContext pattern.
        Implementations should raise StageError on failure.
        """
        ...

    # Shared helpers usable by all operations

    def _prepare_output_file(self, path: Path) -> None:
        """Ensure the output directory exists and remove any pre-existing file.
        Centralized helper so run_pipeline doesn't need to handle per-op output housekeeping.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive; mkdir rarely fails here
            pass

        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - defensive
            pass


class ExportOperation(Operation, ABC):

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def write_xmp_file(
        self, src_xmp: Path, dst_xmp: Path, stage: int, *, verbose: bool = False
    ) -> None:
        """Helper to build and write a stage-specific XMP file.
        Delegates to nind_denoise.xmp.write_xmp_file to avoid duplicating logic.
        """
        from nind_denoise.pipeline.orchestrator import write_xmp_file as _write_xmp_file

        _write_xmp_file(src_xmp, dst_xmp, stage, verbose=verbose)

    def _run_cmd(self, args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
        from . import (
            run_cmd as _run_cmd_shared,
        )  # Defer import to avoid circular dependency during import time

        if hasattr(self, "_ctx") and self._ctx.verbose:  # type: ignore[attr-defined]
            logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
        _run_cmd_shared(args, cwd=cwd)


class DenoiseOperation(Operation, ABC):

    def _run_cmd(self, args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
        from . import (
            run_cmd as _run_cmd_shared,
        )  # Defer import to avoid circular dependency during import time

        if hasattr(self, "_ctx") and self._ctx.verbose:  # type: ignore[attr-defined]
            logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
        _run_cmd_shared(args, cwd=cwd)
