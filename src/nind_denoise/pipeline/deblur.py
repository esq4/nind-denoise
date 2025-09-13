from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import DeblurOperation, Context, StageError


@dataclass
class RLParams:
    sigma: int
    iterations: int
    quality: int


class DeblurStageNoOp(DeblurOperation):
    def describe(self) -> str:
        return "Deblur (disabled)"

    def execute(self, ctx: Context) -> None:
        # Intentionally does nothing
        return


class DeblurStageRL(DeblurOperation):
    def __init__(self, s2_tif: Path, final_out: Path, p: RLParams):
        super().__init__(tools=None)
        self.s2_tif = s2_tif
        self.final_out = final_out
        self.p = p

    def describe(self) -> str:
        return "Deblur (RL via gmic)"

    def execute(self, ctx: Context) -> None:
        if ctx.output_dir is None:
            raise StageError("output_dir not provided in Context")
        if ctx.cmd_gmic is None:
            raise StageError("cmd_gmic not provided in Context")
        # Mirror the legacy command structure expected by tests
        args = [
            str(ctx.cmd_gmic),
            str(self.s2_tif),
            "-deblur_richardsonlucy",
            f"{self.p.sigma},{self.p.iterations},1",
            "-/",
            "256",
            "cut",
            "0,255",
            "round",
            "-o",
            f"{self.final_out.name},{self.p.quality}",
        ]
        self._run_cmd(args, cwd=ctx.output_dir)
        # Logging handled by base helper via ctx.verbose