"""Base classes and execution helpers for pipeline operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Optional, Sequence, Type

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
        pass

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


class ExportOperation(Operation, ABC):

    @abstractmethod
    def get_result(self):
        pass

    def execute(self, args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
        """
        Execute a command with the given arguments and working directory.

        :param args: The sequence of arguments to be passed to the command.
                     Each argument can be a string or a path object.
        :type args: Sequence[str | Path]

        :param cwd: The working directory for the command. If not provided,
                   the current working directory is used.
        :type cwd: Optional[Path]

        :return: This method does not return any value.
        :rtype: None

        .. note::

           If the verbose mode is enabled in the context, it logs the command
           being executed.

        .. warning::

           Ensure that the provided arguments and working directory are valid;
           otherwise, the command execution may fail.

        .. seealso::

           The `self._ctx.run_cmd` method for executing commands.
        """
        if hasattr(self, "_ctx"):
            if getattr(self._ctx, "verbose", False):  # type: ignore[attr-defined]
                logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
            self._ctx.run_cmd(args, cwd=cwd)


class DenoiseOperation(Operation, ABC):

    def _run_cmd(self, args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
        from nind_denoise import config as _config

        if hasattr(self, "_ctx") and getattr(self._ctx, "verbose", False):  # type: ignore[attr-defined]
            logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
        _config.run_cmd(args, cwd=cwd)


class StageError(Exception):
    pass
