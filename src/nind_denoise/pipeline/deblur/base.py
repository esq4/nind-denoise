from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Optional, Sequence

from nind_denoise.config import Tools
from nind_denoise.pipeline.base import Operation, logger


class Deblur(Operation, ABC):
    def __init__(self, tools: Optional[Tools] = None):
        self.tools = tools

    def _run_cmd(self, args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
        from . import (
            run_cmd as _run_cmd_shared,
        )  # Defer import to avoid circular dependency during import time

        if hasattr(self, "_ctx") and self._ctx.verbose:  # type: ignore[attr-defined]
            logger.info("%s: %s", self.describe(), " ".join(map(str, args)))
        _run_cmd_shared(args, cwd=cwd)
