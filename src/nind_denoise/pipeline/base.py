"""Base classes and execution helpers for pipeline operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..config import Tools  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


@dataclass
class Context:
    inpath: Optional[Path] = None
    outpath: Optional[Path] = None
    output_dir: Optional[Path] = None
    output_filepath: Optional[Path] = None
    sigma: Optional[int] = None
    iteration: Optional[str] = None
    quality: Optional[str] = None
    cmd_gmic: Optional[str] = None
    verbose: bool = False


class StageError(Exception):
    pass


class Operation(ABC):
    """Abstract base for all operations (export, denoise, deblur)."""

    @abstractmethod
    def describe(self) -> str:  # pragma: no cover - description only
        ...

    @abstractmethod
    def execute(self, ctx: Context) -> None:  # pragma: no cover
        ...

    @abstractmethod
    def verify(self, ctx: Optional[Context] = None) -> None:  # pragma: no cover
        """Verify expected outputs for this operation.
        Implementations should raise StageError on failure.
        """
        ...

    # Shared helpers usable by all operations
    def _run_cmd(self, args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
        from . import (
            run_cmd as _run_cmd_shared,
        )  # Defer import to avoid circular dependency during import time

        if hasattr(self, "_ctx") and self._ctx.verbose:  # type: ignore[attr-defined]
            logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
        _run_cmd_shared(args, cwd=cwd)

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
    def __init__(self, tools: Tools):
        self.tools = tools

    def write_xmp_file(
        self, src_xmp: Path, dst_xmp: Path, stage: int, *, verbose: bool = False
    ) -> None:
        """Helper to build and write a stage-specific XMP file.
        Delegates to nind_denoise.xmp.write_xmp_file to avoid duplicating logic.
        """
        from ..xmp import write_xmp_file as _write_xmp_file

        _write_xmp_file(src_xmp, dst_xmp, stage, verbose=verbose)


class DenoiseOperation(Operation, ABC):
    pass


class DeblurOperation(Operation, ABC):
    def __init__(self, tools: Optional[Tools] = None):
        self.tools = tools
