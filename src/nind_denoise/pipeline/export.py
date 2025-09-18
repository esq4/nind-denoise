from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import trio

from .base import ExportOperation, StageError
from ..config.config import Config


# TODO: you were working on resolving the coexistence of this and ExportOperation
class DarktableExport(ExportOperation):
    def __init__(self, cfg: Config, img: Path | Dict[Path, Path]):
        if isinstance(img, dict):
            self.img = img["image"]
            self._xmp = img["xmp"]
        else:
            self.img = img
            self._xmp = img.with_suffix(img.suffix + ".xmp")
        self.cfg = cfg

    @property
    def xmp(self):
        return self._xmp

    async def get_result(self):
        async with trio.Path.open(self.cfg.output_path, "rb") as f:
            return io.Bytes(f.read())

    def describe(self) -> str:
        return f"Export Stage {self.stage_number} ({32 if self.stage_number == 1 else 16}-bit TIFF)"

    def execute(self) -> None:
        """Execute export stage using only Config (which includes per-job fields)."""
        out_tif = Path(cfg.output_path)  # type: ignore[arg-type]
        stage_xmp = out_tif.with_suffix(f".s{self.stage_number}.xmp")

        # Build the stage-X XMP adjacent to the input TIFF before export
        if not src_xmp.exists():
            raise StageError(f"Missing input XMP: {src_xmp}")

        # Prepare output file path (create dir, remove any pre-existing file)
        self._prepare_output_file(out_tif)

        # Ensure stage-X XMP is written where the export runs (cwd)
        self.write_xmp_file(
            src_xmp,
            stage_xmp,
            stage=self.stage_number,
            verbose=bool(getattr(cfg, "verbose", False)),
        )

        # Resolve darktable path from configuration
        dt_tool = getattr(cfg.tools, "_darktable", None)
        darktable_path = str(dt_tool.path) if dt_tool is not None else "darktable-cli"

        cmd_args = [
            darktable_path,
            img,
            stage_xmp.name,
            out_tif.name,
            "--apply-custom-presets",
            "false",
            "--core",
            "--conf",
            f"plugins/imageio/format/tiff/bpp={32 if self.stage_number == 1 else 16}",
        ]

        # Add stage-specific arguments
        if self.stage_number == 2:
            cmd_args.extend(["--icc-intent", "PERCEPTUAL", "--icc-type", "SRGB"])

        self.execute(cmd_args, cwd=out_tif.parent)
        self.verify(cfg)

    def verify(self, cfg: Config) -> None:
        """Verify export stage outputs exist (allow tiff/tif normalization)."""
        out_tif = Path(cfg.output_path)  # type: ignore[arg-type]

        if not out_tif.exists():
            # Darktable may shorten '.tiff' to '.tif' on some platforms; normalize to requested
            if out_tif.suffix.lower() == ".tiff":
                alt_path = out_tif.with_suffix(".tif")
                if alt_path.exists():
                    alt_path.replace(out_tif)

        if not out_tif.exists():
            raise StageError(f"Stage {self.stage_number} export missing: {out_tif}")
