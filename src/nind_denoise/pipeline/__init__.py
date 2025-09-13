"""Pipeline orchestration and public API for nind_denoise.pipeline package."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

from .base import Context
from .deblur import RLDeblur
from .denoise import DenoiseOptions, DenoiseStage
from .export import ExportStage
from ..config import read_config, resolve_tools, run_cmd, subprocess, valid_extensions
from ..exif import clone_exif

logger = logging.getLogger(__name__)

# Public re-exports
__all__ = [
    "Context",
    "RLDeblur",
    "valid_extensions",
    "run_pipeline",
    "get_stage_filepaths",
    "get_output_extension",
    "resolve_output_paths",
    "run_cmd",
    "clone_exif",
    "subprocess",
]

# Defaults
DEFAULT_JPEG_QUALITY = 90
DEFAULT_RL_SIGMA = 1
DEFAULT_RL_ITERATIONS = 10


def get_output_extension(args: dict) -> str:
    """Normalize output extension string from args dict."""
    ext = str(args.get("--extension", "")).strip() or "jpg"
    return ext if ext.startswith(".") else "." + ext


def resolve_output_paths(
    input_path: pathlib.Path, output_path_opt: Optional[str], out_ext: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """Compute output directory and final output path for the given input and extension."""
    output_dir = pathlib.Path(output_path_opt) if output_path_opt else input_path.parent
    outpath = (output_dir / input_path.name).with_suffix(out_ext)
    return output_dir, outpath


def get_stage_filepaths(outpath: pathlib.Path, stage: int):
    """Return stage-specific intermediate file paths."""
    if stage == 1:
        s1 = outpath.with_stem(outpath.stem + "_s1").with_suffix(".tif")
        s1d = outpath.with_stem(outpath.stem + "_s1_denoised").with_suffix(".tif")
        return s1, s1d
    s2 = outpath.with_stem(outpath.stem + "_s2").with_suffix(".tif")
    return s2


def validate_input_file(input_path: pathlib.Path) -> None:
    """Validate the input file and its XMP."""
    input_xmp = input_path.with_suffix(input_path.suffix + ".xmp")
    good_file = input_path.exists() and input_path.is_file()
    good_xmp = input_xmp.exists() and input_xmp.is_file()

    if not (good_file or good_xmp):
        logger.error("The input raw-image or its XMP were not found, or are not valid.")
        raise FileNotFoundError(str(input_path))


def download_model_if_needed(model_path: pathlib.Path) -> None:
    """Download the model file if it does not exist."""
    from torch import hub

    if not model_path.exists():
        logger.info("Downloading denoiser model to %s", model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        hub.download_url_to_file(
            "https://f005.backblazeb2.com/file/modelzoo/nind/generator_650.pt",
            str(model_path),
        )


def resolve_unique_output_path(outpath: pathlib.Path) -> None:
    """Ensure output path is unique."""
    i = 1
    while outpath.exists():
        outpath = outpath.with_stem(outpath.stem + "_" + str(i))
        i += 1
        if i >= 99:
            logger.error(
                "Too many files with the same name already exist in %s", outpath.parent
            )
            raise FileExistsError(str(outpath.parent))


def run_pipeline(_args: dict, _input_path: pathlib.Path) -> None:
    """Top-level orchestrator for the denoise pipeline.
    This version lives in the nind_denoise.pipeline package and uses only
    canonical module imports.
    """
    verbose = _args.get("--verbose", False)
    if verbose:
        logger.setLevel(logging.DEBUG)

    validate_input_file(_input_path)

    output_extension = get_output_extension(_args)
    output_dir, outpath = resolve_output_paths(
        _input_path, _args.get("--output-path"), output_extension
    )

    sigma = int(_args.get("--sigma", DEFAULT_RL_SIGMA))
    quality = int(_args.get("--quality", DEFAULT_JPEG_QUALITY))
    iterations = int(_args.get("--iterations", DEFAULT_RL_ITERATIONS))

    stage_one_output_filepath, stage_one_denoised_filepath = get_stage_filepaths(
        outpath, 1
    )
    stage_two_output_filepath = get_stage_filepaths(outpath, 2)

    config = read_config(verbose=verbose)

    # Resolve external tools (darktable-cli, gmic)
    dt_path = pathlib.Path(_args.get("--dt")) if _args.get("--dt") else None
    gmic_path = pathlib.Path(_args.get("--gmic")) if _args.get("--gmic") else None
    tools = resolve_tools(dt_path, gmic_path)

    rldeblur_enabled = bool(tools.gmic) and not _args.get("--no_deblur")
    if not rldeblur_enabled:
        logger.warning(
            "gmic (%s) does not exist or --no_deblur is set, disabled RL-deblur",
            tools.gmic if tools.gmic else _args.get("--gmic"),
        )

    resolve_unique_output_path(outpath)

    # Stage 1 export (32-bit TIFF)
    s1_xmp = stage_one_output_filepath.with_suffix(".s1.xmp")
    ExportStage1(tools, _input_path, s1_xmp, stage_one_output_filepath).execute(
        Context(verbose=verbose)
    )

    model_config = config["models"]["nind_generator_650.pt"]
    model_path = pathlib.Path(model_config["path"])
    download_model_if_needed(model_path)

    # Denoiser stage
    DenoiseStage(
        stage_one_output_filepath,
        stage_one_denoised_filepath,
        DenoiseOptions(model_path=model_path, overlap=6, batch_size=1),
    ).execute(Context(verbose=verbose))

    clone_exif(_input_path, stage_one_denoised_filepath)

    # Stage 2 export (16-bit TIFF)
    xmp2_dst = stage_one_denoised_filepath.with_suffix(".s2.xmp")
    ExportStage(
        tools,
        stage_one_denoised_filepath,
        _input_path.with_suffix(".xmp"),
        xmp2_dst,
        stage_two_output_filepath,
    ).execute(Context(verbose=verbose))

    # Deblur stage
    ctx = Context(
        outpath=outpath,
        output_filepath=stage_two_output_filepath,
        sigma=int(sigma),
        iteration=str(iterations),
        quality=str(quality),
        cmd_gmic=str(tools.gmic),
        output_dir=output_dir,
        verbose=verbose,
    )
    if rldeblur_enabled:
        RLDeblur().execute(ctx)

    clone_exif(stage_one_output_filepath, outpath, verbose=verbose)

    # Cleanup intermediate files
    if not _args.get("--debug"):
        for intermediate_file in [
            stage_one_output_filepath,
            stage_two_output_filepath,
            s1_xmp,
            xmp2_dst,
        ]:
            if intermediate_file != outpath:
                pathlib.Path(intermediate_file).unlink(missing_ok=True)
