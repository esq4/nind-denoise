from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import Context, DenoiseOperation, Environment, JobContext, StageError


@dataclass
class DenoiseOptions:
    model_path: Path
    overlap: int = 6
    batch_size: int = 1


class DenoiseStage(DenoiseOperation):
    def __init__(self, s1_tif: Path, s1_denoised_tif: Path, opts: DenoiseOptions):
        self.s1_tif = s1_tif
        self.s1_denoised_tif = s1_denoised_tif
        self.opts = opts

    def describe(self) -> str:
        return "Denoise"

    def execute(self, ctx: Context) -> None:
        # Prepare output file (ensure directory exists and unlink stale file)
        self._prepare_output_file(self.s1_denoised_tif)
        # Import locally to avoid import-time side effects
        from nind_denoise import denoise_image as _dim
        from types import SimpleNamespace as NS

        args = NS(
            cs=None,
            ucs=None,
            overlap=self.opts.overlap,
            input=str(self.s1_tif),
            output=str(self.s1_denoised_tif),
            batch_size=self.opts.batch_size,
            debug=False,
            exif_method="piexif",
            g_network="UtNet",
            model_path=str(self.opts.model_path),
            model_parameters=None,
            max_subpixels=None,
            whole_image=False,
            pad=None,
            models_dpath=None,
        )
        _dim.run_from_args(args)
        if not self.s1_denoised_tif.exists():
            raise StageError(f"denoise did not create: {self.s1_denoised_tif}")

    def verify(self, ctx: Context | None = None) -> None:
        """Verify that the denoised output file was created successfully."""
        if not self.s1_denoised_tif.exists():
            raise StageError(
                f"Denoise stage expected output missing: {self.s1_denoised_tif}"
            )

    def execute_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Execute denoise stage with type-safe Environment + JobContext."""
        # Input comes from JobContext
        s1_tif = job_ctx.input_path
        s1_denoised_tif = job_ctx.output_path

        # Model path comes from Environment config
        model_config = env.config.get("models", {}).get("nind_generator_650.pt", {})
        model_path = Path(model_config.get("path", "models/nind/generator_650.pt"))

        # Create DenoiseOptions with model path
        opts = DenoiseOptions(
            model_path=model_path,
            overlap=6,  # Could be configurable via JobContext in future
            batch_size=1,  # Could be configurable via JobContext in future
        )

        # Prepare output file (ensure directory exists and unlink stale file)
        self._prepare_output_file(s1_denoised_tif)

        # Import locally to avoid import-time side effects
        from nind_denoise import denoise_image as _dim
        from types import SimpleNamespace as NS

        args = NS(
            cs=None,
            ucs=None,
            overlap=opts.overlap,
            input=str(s1_tif),
            output=str(s1_denoised_tif),
            batch_size=opts.batch_size,
            debug=False,
            exif_method="piexif",
            g_network="UtNet",
            model_path=str(opts.model_path),
            model_parameters=None,
            max_subpixels=None,
            whole_image=False,
            pad=None,
            models_dpath=None,
        )
        _dim.run_from_args(args)

        if not s1_denoised_tif.exists():
            raise StageError(f"denoise did not create: {s1_denoised_tif}")

    def verify_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Verify denoise stage outputs with type-safe Environment + JobContext."""
        s1_denoised_tif = job_ctx.output_path

        if not s1_denoised_tif.exists():
            raise StageError(
                f"Denoise stage expected output missing: {s1_denoised_tif}"
            )
