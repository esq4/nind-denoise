from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import pathlib
import subprocess
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class Context:
    outpath: pathlib.Path
    stage_two_output_filepath: pathlib.Path
    sigma: int
    iteration: str
    quality: str
    cmd_gmic: str
    output_dir: pathlib.Path
    verbose: bool = False


def run_cmd(args: Iterable[pathlib.Path | str], cwd: pathlib.Path | None = None) -> None:
    """Run a subprocess command with proper logging and Path handling."""
    cmd = [str(a) for a in args]
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=cwd, check=True)


class Stage(ABC):
    @abstractmethod
    def execute(self, ctx: Context) -> None:  # pragma: no cover - interface only
        ...


class DeblurStage(Stage, ABC):
    pass


class NoOpDeblur(DeblurStage):
    def execute(self, ctx: Context) -> None:
        if ctx.verbose:
            logger.info("RL-deblur disabled; skipping.")


class RLDeblur(DeblurStage):
    def execute(self, ctx: Context) -> None:
        outpath = ctx.outpath
        stage_two_output_filepath = ctx.stage_two_output_filepath
        sigma = ctx.sigma
        iterations = ctx.iteration
        quality = ctx.quality
        cmd_gmic = ctx.cmd_gmic
        output_dir = ctx.output_dir

        # Build and run gmic command; no need to rename files as we pass args list
        args = [
            cmd_gmic,
            stage_two_output_filepath,
            '-deblur_richardsonlucy', f"{sigma},{iterations},1",
            '-/', '256', 'cut', '0,255', 'round',
            '-o', f"{outpath.name},{quality}",
        ]
        run_cmd(args, cwd=output_dir)
        if ctx.verbose:
            logger.info("Applied RL-deblur to: %s", outpath)
