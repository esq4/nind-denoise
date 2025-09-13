from __future__ import annotations

from pathlib import Path

from .base import ExportOperation, Context, StageError


class ExportStage1(ExportOperation):
    def __init__(self, tools, input_raw: Path, stage1_xmp: Path, out_tif32: Path):
        super().__init__(tools)
        self.input_raw = input_raw
        self.stage1_xmp = stage1_xmp
        self.out_tif32 = out_tif32

    def describe(self) -> str:
        return "Export Stage 1 (32-bit TIFF)"

    def execute(self, ctx: Context) -> None:
        self._run_cmd(
            [
                self.tools.darktable,
                self.input_raw,
                self.stage1_xmp.name,
                self.out_tif32.name,
                "--apply-custom-presets",
                "false",
                "--core",
                "--conf",
                "plugins/imageio/format/tiff/bpp=32",
            ],
            cwd=self.out_tif32.parent,
        )
        if not self.out_tif32.exists():
            raise StageError(f"s1 export missing: {self.out_tif32}")


class ExportStage2(ExportOperation):
    def __init__(self, tools, s1_denoised_tif: Path, stage2_xmp: Path, out_tif16: Path):
        super().__init__(tools)
        self.s1_denoised_tif = s1_denoised_tif
        self.stage2_xmp = stage2_xmp
        self.out_tif16 = out_tif16

    def describe(self) -> str:
        return "Export Stage 2 (16-bit TIFF)"

    def execute(self, ctx: Context) -> None:
        self._run_cmd(
            [
                self.tools.darktable,
                self.s1_denoised_tif,
                self.stage2_xmp.name,
                self.out_tif16.name,
                "--icc-intent",
                "PERCEPTUAL",
                "--icc-type",
                "SRGB",
                "--apply-custom-presets",
                "false",
                "--core",
                "--conf",
                "plugins/imageio/format/tiff/bpp=16",
            ],
            cwd=self.out_tif16.parent,
        )