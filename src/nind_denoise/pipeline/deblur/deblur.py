"""Deblur operation(s) for the brummer2019-denoise pipeline."""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..base import Operation, StageError
from ...config.config import Config

logger = logging.getLogger(__name__)


class Deblur(Operation, ABC):

    # Context manager support (sync)

    # Async context manager support

    def _prepare_output_file(self, path: Path) -> None:
        """Ensure the output directory exists and remove any pre-existing file.
        Centralized helper so run_pipeline doesn't need to handle per-op output housekeeping.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive; mkdir rarely fails here
            pass

        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - defensive
            pass

    def write_xmp_file(
        self, src_xmp: Path, dst_xmp: Path, stage: int, *, verbose: bool = False
    ) -> None:
        """Helper to build and write a stage-specific XMP file.
        Delegates to nind_denoise.xmp.write_xmp_file to avoid duplicating logic.
        """
        from nind_denoise.pipeline.orchestrator import write_xmp_file as _write_xmp_file

        _write_xmp_file(src_xmp, dst_xmp, stage, verbose=verbose)


class RLDeblur(Deblur):
    """RL-deblur stage implemented via gmic."""

    def describe(self) -> str:  # pragma: no cover - description only
        return "Deblur (RL via gmic)"

    def execute(self, cfg: Config) -> None:
        """Execute RL deblur using only Config (with per-job fields)."""
        # Resolve gmic path
        gmic_tool = getattr(cfg.tools, "_gmic", None)
        if gmic_tool is None:
            raise StageError("GMIC tool not available in Config.tools")

        input_path = Path(cfg.intermediate_path or cfg.input_path)  # type: ignore[arg-type]
        output_path = Path(cfg.output_path)  # type: ignore[arg-type]

        # Use a temporary target name to avoid issues with spaces; rename at the end
        tmp_out = output_path.with_name(output_path.stem + "__tmp.jpg")
        args = [
            str(gmic_tool.path),
            str(input_path.name),
            "-fx_sharpen_reinhard",
            str(getattr(cfg, "sigma", 1)),
            str(getattr(cfg, "iterations", 10)),
            "-o_jpg",
            f"{tmp_out.name},{getattr(cfg, 'quality', 90)}",
        ]

        try:
            self._run_cmd(args, cwd=output_path.parent)
        except Exception as exc:
            logger.warning("RL-deblur failed to execute (%s); keeping exported image as-is", exc)
            return

        # Rename back to requested output_path (if gmic produced it)
        tmp_path = output_path.parent / tmp_out.name
        if tmp_path.exists():
            tmp_path.replace(output_path)
        else:
            logger.warning(
                "RL-deblur did not create expected output %s; keeping exported image as-is",
                tmp_path,
            )
            return

        self.verify(cfg)

    def verify(self, cfg: Config) -> None:
        """Verify RL deblur outputs exist."""
        output_path = Path(cfg.output_path)  # type: ignore[arg-type]
        if not output_path.exists():
            raise StageError(f"Deblur stage expected output missing: {output_path}")


class RLDeblurPT(Deblur):
    """PyTorch implementation of the RL deblur stage.

    This stage mirrors the behavior of :class:`RLDeblur` but uses a pure Python
    PyTorch backend instead of invoking the external `gmic` CLI.
    """

    def describe(self) -> str:
        return "Deblur (RL via PyTorch)"

    def execute(self, cfg: Config) -> None:
        """Execute PyTorch RL deblur using only Config (with per-job fields)."""
        input_path = Path(cfg.intermediate_path or cfg.input_path)  # type: ignore[arg-type]
        output_path = Path(cfg.output_path)  # type: ignore[arg-type]
        sigma = float(getattr(cfg, "sigma", 1))
        iterations = int(getattr(cfg, "iterations", 10))
        quality = int(getattr(cfg, "quality", 90))

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
            deblur = richardson_lucy_gaussian(img_tensor, sigma=sigma, iterations=iterations)

            # Convert back to PIL and save with quality
            if deblur.dtype == torch.uint8:
                out_np = deblur.cpu().numpy()
            else:
                out_np = (deblur.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            out_img = Image.fromarray(out_np, mode="RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(output_path, quality=quality)

            if getattr(cfg, "verbose", False):
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

        self.verify(cfg)

    def verify(self, cfg: Config) -> None:
        """Verify PyTorch RL deblur outputs exist."""
        output_path = Path(cfg.output_path)  # type: ignore[arg-type]
        if not output_path.exists():
            raise StageError(f"PyTorch RL deblur stage expected output missing: {output_path}")
