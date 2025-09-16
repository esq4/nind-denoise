"""Pipeline orchestration and public API for nind_denoise.pipeline package."""

from __future__ import annotations

from typing import Dict, Type

# Stage types and defaults
from .base import Context, DeblurOperation, DenoiseOperation, ExportOperation
from .deblur import RLDeblur, RLDeblurPT
from .denoise import DenoiseOptions, DenoiseStage
from .export import ExportStage
from .orchestrator import (
    get_output_extension,
    get_stage_filepaths,
    resolve_output_paths,
    run_pipeline,
    validate_input_file,
)
# Public utilities
from ..config import run_cmd, subprocess, valid_extensions
from ..exif import clone_exif

# Simple registries (extensible)
_EXPORTERS: Dict[str, Type[ExportOperation]] = {"darktable": ExportStage}
_DENOISERS: Dict[str, Type[DenoiseOperation]] = {"nind_pt": DenoiseStage}
_DEBLUR: Dict[str, Type[DeblurOperation]] = {"gmic": RLDeblur, "pt_rl": RLDeblurPT}


def register_exporter(name: str, cls: Type[ExportOperation]) -> None:
    _EXPORTERS[name] = cls


def get_exporter(name: str = "darktable") -> Type[ExportOperation]:
    return _EXPORTERS[name]


def register_denoiser(name: str, cls: Type[DenoiseOperation]) -> None:
    _DENOISERS[name] = cls


def get_denoiser(name: str = "nind_pt") -> Type[DenoiseOperation]:
    return _DENOISERS[name]


def register_deblur(name: str, cls: Type[DeblurOperation]) -> None:
    _DEBLUR[name] = cls


def get_deblur(name: str = "gmic") -> Type[DeblurOperation]:
    return _DEBLUR[name]


__all__ = [
    "Context",
    "run_pipeline",
    "get_stage_filepaths",
    "get_output_extension",
    "resolve_output_paths",
    "run_cmd",
    "clone_exif",
    "subprocess",
    "ExportStage",
    "RLDeblur",
    "RLDeblurPT",
    "DenoiseStage",
    "DenoiseOptions",
    "register_exporter",
    "get_exporter",
    "register_denoiser",
    "get_denoiser",
    "register_deblur",
    "get_deblur",
    "valid_extensions",
]
