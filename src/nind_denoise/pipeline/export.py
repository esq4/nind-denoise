from __future__ import annotations

from pathlib import Path

from .base import Context, ExportOperation, StageError


class ExportStage(ExportOperation):
    def __init__(
        self,
        tools,
        input_tif: Path,
        src_xmp: Path,
        stage_xmp: Path,
        out_tif: Path,
        stage_number: int,
    ):
        super().__init__(tools)
        self.input_tif = input_tif
        self.src_xmp = src_xmp
        self.stage_xmp = stage_xmp
        self.out_tif = out_tif
        self.stage_number = stage_number

    def describe(self) -> str:
        return f"Export Stage {self.stage_number} ({32 if self.stage_number == 1 else 16}-bit TIFF)"

    def execute(self, ctx: Context) -> None:
        # Store context for logging helpers and delegate to run
        self._ctx = ctx  # type: ignore[attr-defined]
        self.run(ctx)

    def run(self, ctx: Context) -> None:
        # Build the stage-X XMP adjacent to the input TIFF before export
        if not self.src_xmp.exists():
            raise StageError(f"Missing input XMP: {self.src_xmp}")

        # Prepare output file path (create dir, remove any pre-existing file)
        self._prepare_output_file(self.out_tif)

        # Ensure stage-X XMP is written where the export runs (cwd)
        self.write_xmp_file(
            self.src_xmp, self.stage_xmp, stage=self.stage_number, verbose=ctx.verbose
        )

        cmd_args = [
            self.tools.darktable,
            self.input_tif,
            self.stage_xmp.name,
            self.out_tif.name,
            "--apply-custom-presets",
            "false",
            "--core",
            "--conf",
            f"plugins/imageio/format/tiff/bpp={32 if self.stage_number == 1 else 16}",
        ]

        # Add stage-specific arguments
        if self.stage_number == 2:
            cmd_args.extend(["--icc-intent", "PERCEPTUAL", "--icc-type", "SRGB"])

        self._run_cmd(cmd_args, cwd=self.out_tif.parent)
        self.verify()

    def verify(self, ctx: Context | None = None) -> None:
        if not self.out_tif.exists():
            # Darktable may shorten '.tiff' to '.tif' on some platforms; normalize to requested
            if self.out_tif.suffix.lower() == ".tiff":
                alt_path = self.out_tif.with_suffix(".tif")
                if alt_path.exists():
                    alt_path.replace(self.out_tif)
        if not self.out_tif.exists():
            raise StageError(
                f"Stage {self.stage_number} export missing: {self.out_tif}"
            )
