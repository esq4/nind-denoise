from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import pathlib
import subprocess


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


class Stage(ABC):
    @abstractmethod
    def execute(self, ctx: Context) -> None:  # pragma: no cover - interface only
        ...


class DeblurStage(Stage, ABC):
    pass


class NoOpDeblur(DeblurStage):
    def execute(self, ctx: Context) -> None:
        # nothing to do
        if ctx.verbose:
            print('RL-deblur disabled; skipping.')


class RLDeblur(DeblurStage):
    def execute(self, ctx: Context) -> None:
        outpath = ctx.outpath
        stage_two_output_filepath = ctx.stage_two_output_filepath
        sigma = ctx.sigma
        iteration = ctx.iteration
        quality = ctx.quality
        cmd_gmic = ctx.cmd_gmic
        output_dir = ctx.output_dir
        # cope with spaces in the filename for gmic
        if ' ' in outpath.name:
            restore_original_outpath = outpath.name
            outpath = outpath.rename(outpath.with_name(outpath.name.replace(' ', '_')))
        else:
            restore_original_outpath = None
        subprocess.run([cmd_gmic, stage_two_output_filepath,
                        '-deblur_richardsonlucy', str(sigma) + ',' + str(iteration) + ',' + '1',
                        '-/', '256', 'cut', '0,255', 'round',
                        '-o', outpath.name + ',' + str(quality)],
                       cwd=output_dir, check=True)
        if ctx.verbose:
            print('Applied RL-deblur to:', outpath)
        if restore_original_outpath is not None:
            outpath.replace(outpath.with_name(restore_original_outpath))  # restore original name with spaces
