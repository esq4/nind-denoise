"""Base classes and execution helpers for pipeline operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import AsyncIterator, Optional, Self, Sequence, Type

from nind_denoise.config.config import Config

logger = logging.getLogger(__name__)


class OperationsFactory:
    """
    Async-iterable factory that resolves stage names to concrete Operation classes.

    Parameters:
        stages: Ordered sequence of stage identifiers to resolve (e.g. ["darktable", "nind_pt", "gmic"]).
        op_name: Operation category to resolve for each stage. One of: "exporter", "denoiser", "deblur".

    Yields:
        Type[Operation]: For each stage in `stages`, the registered Operation class for the selected category.

    Usage:
        async for Op in OperationsFactory(stages, "denoiser"):
            # `Op` is a class (subclass of Operation); instantiate or inspect as needed
            ...

    Notes:
        - Resolution is performed lazily during iteration via the package registries.
        - A ValueError is raised if an unknown `op_name` is provided.
        - A KeyError may be raised if a given stage name is not registered for the chosen category.
    """

    def __init__(self, stages: Sequence[str], op_name: str):
        self.stages = stages
        self.op_name = op_name  # one of: "exporter", "denoiser", "deblur"

    def __aiter__(self) -> AsyncIterator[Type["Operation"]]:
        """
        Return an async iterator over Operation classes for the configured category.

        Iterates over `self.stages` in order and yields the class registered for each
        stage under the selected `op_name` category.

        Raises:
            ValueError: If `op_name` is not one of the supported categories.
            KeyError: If a stage name is not registered for the chosen category.
        """

        async def _gen() -> AsyncIterator[Type["Operation"]]:
            # Lazy import to avoid circular imports at module import time
            from nind_denoise.pipeline import get_exporter, get_denoiser, get_deblur

            resolvers = {
                "exporter": get_exporter,
                "denoiser": get_denoiser,
                "deblur": get_deblur,
            }
            resolver = resolvers.get(self.op_name)
            if resolver is None:
                raise ValueError(f"Unknown operation type: {self.op_name}")

            for stage in self.stages:
                yield resolver(stage)

        return _gen()


class Operation(ABC):
    """Abstract base for all operations (export, denoise, deblur)."""

    def __init__(self):
        self._ctx = self.cfg  # type: ignore[attr-defined]

    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def describe(self) -> str:  # pragma: no cover - description only
        ...

    @abstractmethod
    def execute(self, cfg: Config) -> None:  # pragma: no cover
        """Execute operation with Config."""
        ...

    @abstractmethod
    def verify(self, cfg: Config) -> None:  # pragma: no cover
        """Verify operation outputs. Implementations should raise StageError on failure."""
        ...

    def __enter__(self) -> Self:
        # Attach the config as lightweight context for downstream helpers
        self._ctx = self.cfg  # type: ignore[attr-defined]
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        # Cleanup any ephemeral context and propagate exceptions
        if hasattr(self, "_ctx"):
            try:
                delattr(self, "_ctx")
            except Exception:
                pass
        return False

    async def __aenter__(self) -> Self:
        self._ctx = self.cfg  # type: ignore[attr-defined]
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if hasattr(self, "_ctx"):
            try:
                delattr(self, "_ctx")
            except Exception:
                pass
        return False


class ExportOperation(Operation, ABC):

    @abstractmethod
    def get_result(self):
        pass

    def execute(self, cfg: Config) -> None:
        """
                Execute a command with the given arguments and working directory.

                :param cfg: The configuration object containing command-line arguments.
                :type cfg: Config

                :return: This method does not return any value.
                :rtype: None
        # TODO"""
        if hasattr(self, "_ctx"):
            if getattr(self._ctx, "verbose", False):  # type: ignore[attr-defined]
                logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
            self._ctx.run_cmd(args, cwd=cwd)

    def _prepare_output_file(self) -> None:
        """Ensure the output directory exists and remove any pre-existing file.
        Centralized helper so run_pipeline doesn't need to handle per-op output housekeeping.
        """
        try:
            self._ctxpath.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive; mkdir rarely fails here
            pass

        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - defensive
            pass

    def write_xmp_file(
        self, src_xmp: Path, dst_xmp: Path, stage: int, *, verbose: bool = False
    ) -> None:
        """Helper to build and write a stage-specific XMP file.
        Delegates to nind_denoise.xmp.write_xmp_file to avoid duplicating logic.
        """
        from nind_denoise.pipeline.orchestrator import write_xmp_file as _write_xmp_file

        _write_xmp_file(src_xmp, dst_xmp, stage, verbose=verbose)


class DenoiseOperation(Operation, ABC):

    @abstra
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

    def write_xmp_file(
        self, src_xmp: Path, dst_xmp: Path, stage: int, *, verbose: bool = False
    ) -> None:
        """Helper to build and write a stage-specific XMP file.
        Delegates to nind_denoise.xmp.write_xmp_file to avoid duplicating logic.
        """
        from nind_denoise.pipeline.orchestrator import write_xmp_file as _write_xmp_file

        _write_xmp_file(src_xmp, dst_xmp, stage, verbose=verbose)


class DenoiseStage(DenoiseOperation):
    def __init__(self, input_tif: Path, output_tif: Path, options: DenoiseOptions):
        self.input_tif = input_tif
        self.output_tif = output_tif
        self.options = options

    def describe(self) -> str:  # pragma: no cover - description only
        return "Denoise (NIND PT)"

    def execute(self, ctx: Context) -> None:
        # Store context for helpers/logging
        self._ctx = ctx  # type: ignore[attr-defined]

        if not self.input_tif.exists():
            raise StageError(f"Input TIFF missing for denoise: {self.input_tif}")
        from denoise import NIND

        denoiser = NIND(self.options)
        denoiser.run(self.input_tif, self.output_tif)
        self.verify()

    def verify(self, ctx: Context | None = None) -> None:
        if not self.output_tif.exists():
            raise StageError(f"Denoise stage expected output missing: {self.output_tif}")


class StageError(Exception):
    pass
