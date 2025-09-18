"""Pipeline orchestration and public API for nind_denoise.pipeline package."""

from __future__ import annotations

from typing import Dict, Type

# Stage types and defaults
from .base import DenoiseOperation, ExportOperation
from .deblur import Deblur, RLDeblur, RLDeblurPT
from .denoise import DenoiseOptions, NIND
from .export import DarktableExport
from .orchestrator import (
    get_output_extension,
    get_stage_filepaths,
    resolve_output_paths,
    run_pipeline,
    validate_input_file,
)

# Simple registries (extensible)
_EXPORTERS: Dict[str, Type[ExportOperation]] = {"darktable": DarktableExport}
_DENOISERS: Dict[str, Type[DenoiseOperation]] = {"nind_pt": NIND}
_DEBLUR: Dict[str, Type[Deblur]] = {"gmic": RLDeblur, "pt_rl": RLDeblurPT}


def register_exporter(name: str, cls: Type[ExportOperation]) -> None:
    _EXPORTERS[name] = cls


def get_exporter(name: str = "darktable") -> Type[ExportOperation]:
    return _EXPORTERS[name]


def register_denoiser(name: str, cls: Type[DenoiseOperation]) -> None:
    _DENOISERS[name] = cls


def get_denoiser(name: str = "nind_pt") -> Type[DenoiseOperation]:
    return _DENOISERS[name]


def register_deblur(name: str, cls: Type[Deblur]) -> None:
    _DEBLUR[name] = cls


def get_deblur(name: str = "gmic") -> Type[Deblur]:
    return _DEBLUR[name]


__all__ = [
    "run_pipeline",
    "get_stage_filepaths",
    "get_output_extension",
    "resolve_output_paths",
    "DarktableExport",
    "RLDeblur",
    "RLDeblurPT",
    "NIND",
    "DenoiseOptions",
    "register_exporter",
    "get_exporter",
    "register_denoiser",
    "get_denoiser",
    "register_deblur",
    "get_deblur",
]
