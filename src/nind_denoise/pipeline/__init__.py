from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Options
from .. import pipeline as _compat

from .base import Context
from .export import ExportStage1, ExportStage2
from .denoise import DenoiseStage, DenoiseOptions
from .deblur import DeblurStageRL, DeblurStageNoOp, RLParams

__all__ = [
    "Context",
    "ExportStage1",
    "ExportStage2",
    "DenoiseStage",
    "DenoiseOptions",
    "DeblurStageRL",
    "DeblurStageNoOp",
    "RLParams",
    "run_pipeline",
]


def run_pipeline(opts: Options, input_path: Path) -> Path:
    """Typed run_pipeline orchestrator.

    Delegates to the legacy dict-based adapter in nind_denoise.pipeline for
    compatibility, and returns the computed final output path.
    """
    # Compute expected output path using legacy helpers for consistency
    ext = _compat.get_output_extension({"--extension": getattr(opts, "extension", "jpg")})
    output_dir, outpath = _compat.resolve_output_paths(
        input_path,
        str(opts.output_path) if getattr(opts, "output_path", None) else None,
        ext,
    )
    _compat.run_pipeline_opts(opts, input_path)
    return outpath
