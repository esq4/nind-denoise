"""Base classes and execution helpers for pipeline operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence
import logging

from ..config import Tools  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


@dataclass
class Context:
    input_path: Path | None = None
    output_dir: Path | None = None
    # Fields below mirror the legacy pipeline.Context to ease interop
    outpath: Path | None = None
    stage_two_output_filepath: Path | None = None
    sigma: int | None = None
    iteration: str | None = None
    quality: str | None = None
    cmd_gmic: str | None = None
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

    # Shared helpers usable by all operations
    def _run_cmd(self, args: Sequence[str | Path], cwd: Path | None = None) -> None:
        # Defer import to avoid circular dependency during import time
        from . import run_cmd as _run_cmd_shared

        if getattr(self, "_ctx", None):
            try:
                if self._ctx.verbose:  # type: ignore[attr-defined]
                    logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
            except Exception:  # pylint: disable=broad-exception-caught
                # Be tolerant if injected context doesn't have expected shape
                pass
        _run_cmd_shared(args, cwd=cwd)


class ExportOperation(Operation, ABC):
    def __init__(self, tools: Tools):
        self.tools = tools


class DenoiseOperation(Operation, ABC):
    pass


class DeblurOperation(Operation, ABC):
    def __init__(self, tools: Tools | None = None):
        self.tools = tools