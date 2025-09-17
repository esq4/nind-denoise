"""Deblur operation(s) for the brummer2019-denoise pipeline."""

from __future__ import annotations

import logging
from abc import ABC
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..base import JobContext, Operation, StageError
from ...config import Tools
from ...config.config import Config

logger = logging.getLogger(__name__)


class Deblur(Operation, ABC):
    def __init__(self, tools: Optional[Tools] = None):
        self.tools = tools


class RLDeblur(Deblur):
    """RL-deblur stage implemented via gmic."""

    def describe(self) -> str:  # pragma: no cover - description only
        return "Deblur (RL via gmic)"

    def execute_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
        """Execute RL deblur with type-safe Environment + JobContext."""
        if not cfg.gmic:
            raise StageError("GMIC tool not available in Environment")

        input_path = job_ctx.intermediate_path or job_ctx.input_path
        output_path = job_ctx.output_path

        # Use a temporary target name to avoid issues with spaces; rename at the end
        tmp_out = output_path.with_name(output_path.stem + "__tmp.jpg")
        args = [
            str(cfg.gmic),
            str(input_path.name),
            "-fx_sharpen_reinhard",
            str(job_ctx.sigma),
            str(job_ctx.iterations),
            "-o_jpg",
            f"{tmp_out.name},{job_ctx.quality}",
        ]

        try:
            self._run_cmd(args, cwd=output_path.parent)
        except Exception as exc:
            logger.warning(
                "RL-deblur failed to execute (%s); keeping exported image as-is", exc
            )
            return

        # Rename back to requested output_path (if gmic produced it)
        tmp_path = output_dir / tmp_out.name
        if tmp_path.exists():
            tmp_path.replace(output_path)
        else:
            logger.warning(
                "RL-deblur did not create expected output %s; keeping exported image as-is",
                tmp_path,
            )
            return

        self.verify_with_env(cfg, job_ctx)

    def verify_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
        """Verify RL deblur outputs with type-safe Environment + JobContext."""
        if not job_ctx.output_path.exists():
            raise StageError(
                f"Deblur stage expected output missing: {job_ctx.output_path}"
            )


class RLDeblurPT(Deblur):
    """PyTorch implementation of the RL deblur stage.

    This stage mirrors the behavior of :class:`RLDeblur` but uses a pure Python
    PyTorch backend instead of invoking the external `gmic` CLI.
    """

    def describe(self) -> str:
        return "Deblur (RL via PyTorch)"

    def execute_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
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
            from nind_denoise.libs.RichardsonLucyPytorch import richardson_lucy_gaussian

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

            if cfg.verbose:
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

        self.verify_with_env(cfg, job_ctx)

    def verify_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
        """Verify PyTorch RL deblur outputs with type-safe Environment + JobContext."""
        if not job_ctx.output_path.exists():
            raise StageError(
                f"PyTorch RL deblur stage expected output missing: {job_ctx.output_path}"
            )
