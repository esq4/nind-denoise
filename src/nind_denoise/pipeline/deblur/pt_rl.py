"""PyTorch Richardson-Lucy deblur operation for the nind-denoise pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..base import Context, DeblurOperation, Environment, JobContext, StageError

logger = logging.getLogger(__name__)


class RLDeblurPT(DeblurOperation):
    """PyTorch implementation of the RL deblur stage.

    This stage mirrors the behavior of :class:`RLDeblur` but uses a pure Python
    PyTorch backend instead of invoking the external `gmic` CLI.
    """

    def describe(self) -> str:
        return "Deblur (RL via PyTorch)"

    def execute(self, ctx: Context) -> None:
        if ctx.output_dir is None:
            raise StageError("output_dir not provided in Context")
        if ctx.outpath is None or ctx.output_filepath is None:
            raise StageError("outpath and output_filepath must be set")

        outpath = Path(ctx.outpath)
        s2 = Path(ctx.output_filepath)
        sigma = float(ctx.sigma) if ctx.sigma is not None else 1.0
        iterations = int(ctx.iteration) if ctx.iteration is not None else 10
        quality = int(ctx.quality) if ctx.quality is not None else 90

        if not s2.exists():
            raise StageError(f"Stage-2 input not found: {s2}")

        # Load stage-2 TIFF (or other) as RGB, obtain HxWxC uint8 array
        with Image.open(s2) as im:
            im = im.convert("RGB")
            img_np = np.array(im, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np)  # HxWxC, uint8

        # Import locally to avoid import-time side effects
        from nind_denoise.rl_pt import richardson_lucy_gaussian

        try:
            # Run RL on torch
            deblur = richardson_lucy_gaussian(
                img_tensor, sigma=sigma, iterations=iterations
            )

            # Convert back to PIL and save with quality
            if deblur.dtype == torch.uint8:
                out_np = deblur.cpu().numpy()
            else:
                out_np = (
                    (deblur.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
                )
            out_img = Image.fromarray(out_np, mode="RGB")
            outpath.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(outpath, quality=quality)

            if ctx.verbose:
                logger.info("Applied RL-deblur (PyTorch) to: %s", outpath)

        except Exception as exc:
            # Be resilient to PyTorch issues in integration envs
            logger.warning(
                "PyTorch RL-deblur failed to execute (%s); keeping exported image as-is",
                exc,
            )
            # Copy the stage-2 output to final output as fallback
            if s2 != outpath:
                import shutil

                shutil.copy2(s2, outpath)
            return

        # Post-conditions
        self.verify(ctx)

    def verify(self, ctx: Context | None = None) -> None:
        if ctx is None or ctx.outpath is None:
            # Nothing to verify without context
            return
        if not ctx.outpath.exists():
            raise StageError(
                f"PyTorch RL deblur stage expected output missing: {ctx.outpath}"
            )

    def execute_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Execute PyTorch RL deblur with type-safe Environment + JobContext."""
        input_path = job_ctx.intermediate_path or job_ctx.input_path
        output_path = job_ctx.output_path
        sigma = float(job_ctx.sigma)
        iterations = job_ctx.iterations
        quality = job_ctx.quality

        if not input_path.exists():
            raise StageError(f"Stage input not found: {input_path}")

        try:
            # Load input as RGB, obtain HxWxC uint8 array
            with Image.open(input_path) as im:
                im = im.convert("RGB")
                img_np = np.array(im, dtype=np.uint8)
            img_tensor = torch.from_numpy(img_np)  # HxWxC, uint8

            # Import locally to avoid import-time side effects
            from nind_denoise.rl_pt import richardson_lucy_gaussian

            # Run RL on torch
            deblur = richardson_lucy_gaussian(
                img_tensor, sigma=sigma, iterations=iterations
            )

            # Convert back to PIL and save with quality
            if deblur.dtype == torch.uint8:
                out_np = deblur.cpu().numpy()
            else:
                out_np = (
                    (deblur.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
                )
            out_img = Image.fromarray(out_np, mode="RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(output_path, quality=quality)

            if env.verbose:
                logger.info("Applied RL-deblur (PyTorch) to: %s", output_path)

        except Exception as exc:
            # Be resilient to PyTorch issues in integration envs
            logger.warning(
                "PyTorch RL-deblur failed to execute (%s); keeping exported image as-is",
                exc,
            )
            # Copy the input to final output as fallback
            if input_path != output_path:
                import shutil

                shutil.copy2(input_path, output_path)
            return

        self.verify_with_env(env, job_ctx)

    def verify_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Verify PyTorch RL deblur outputs with type-safe Environment + JobContext."""
        if not job_ctx.output_path.exists():
            raise StageError(
                f"PyTorch RL deblur stage expected output missing: {job_ctx.output_path}"
            )
