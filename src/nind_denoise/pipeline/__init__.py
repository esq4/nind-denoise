"""Pipeline orchestration and public API for nind_denoise.pipeline package."""

from __future__ import annotations

import logging
import pathlib
import shutil
from dataclasses import dataclass
from typing import Iterable

from ..config import valid_extensions
from ..config import read_config
from ..xmp import parse_darktable_history_stack
from ..config import resolve_tools, run_cmd, subprocess
from ..exif import clone_exif

from .base import Context
from .deblur import RLDeblur

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


def get_output_extension(args) -> str:
    """Normalize output extension string from args dict."""
    ext = str(args.get("--extension", "")).strip() or "jpg"
    return ext if ext.startswith(".") else "." + ext


def resolve_output_paths(
    input_path: pathlib.Path, output_path_opt: str | None, out_ext: str
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


def run_pipeline(_args: dict, _input_path: pathlib.Path) -> None:
    """Top-level orchestrator for the denoise pipeline.

    This version lives in the nind_denoise.pipeline package and uses only
    canonical module imports.
    """
    verbose = _args.get("--verbose", False)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Processing %s", _input_path)

    output_extension = get_output_extension(_args)
    output_dir, outpath = resolve_output_paths(
        _input_path, _args.get("--output-path"), output_extension
    )

    input_xmp = _input_path.with_suffix(_input_path.suffix + ".xmp")
    sigma = int(_args.get("--sigma") or DEFAULT_RL_SIGMA)
    quality = int(_args.get("--quality") or DEFAULT_JPEG_QUALITY)
    iterations = int(_args.get("--iterations") or DEFAULT_RL_ITERATIONS)

    stage_one_output_filepath, stage_one_denoised_filepath = get_stage_filepaths(
        outpath, 1
    )
    stage_two_output_filepath = get_stage_filepaths(outpath, 2)

    config = read_config(verbose=verbose)

    # Resolve external tools (darktable-cli, gmic)
    dt_arg = _args.get("--dt")
    gmic_arg = _args.get("--gmic")
    dt_path = pathlib.Path(dt_arg) if dt_arg else None
    gmic_path = pathlib.Path(gmic_arg) if gmic_arg else None
    tools = resolve_tools(dt_path, gmic_path)

    cmd_darktable = tools.darktable
    cmd_gmic = tools.gmic

    rldeblur = bool(cmd_gmic) and not _args.get("--no_deblur")
    if not rldeblur:
        logger.warning(
            "gmic (%s) does not exist or --no_deblur is set, disabled RL-deblur",
            cmd_gmic if cmd_gmic else gmic_arg,
        )
        stage_two_output_filepath = outpath

    # input validation
    good_file = _input_path.exists() and _input_path.is_file()
    good_xmp = input_xmp.exists() and input_xmp.is_file()
    if not (good_file or good_xmp):
        logger.error("The input raw-image or its XMP were not found, or are not valid.")
        raise FileNotFoundError(str(_input_path))

    i = 1
    while outpath.exists():
        outpath = outpath.with_stem(outpath.stem + "_" + str(i))
        i += 1
        if i >= 99:
            logger.error(
                "Too many files with the same name already exist in %s", outpath.parent
            )
            raise FileExistsError(str(outpath.parent))

    parse_darktable_history_stack(input_xmp, config=config, verbose=verbose)

    # Stage 1 export (32-bit TIFF)
    stage_one_output_filepath.unlink(missing_ok=True)
    run_cmd(
        [
            cmd_darktable,
            _input_path,
            input_xmp.with_suffix(".s1.xmp"),
            stage_one_output_filepath.name,
            "--apply-custom-presets",
            "false",
            "--core",
            "--conf",
            "plugins/imageio/format/tiff/bpp=32",
        ],
        cwd=outpath.parent,
    )

    if not stage_one_output_filepath.exists():
        logger.error("First-stage export not found: %s", stage_one_output_filepath)
        raise ChildProcessError(str(stage_one_output_filepath))

    # Denoiser (invoke natively)
    stage_one_denoised_filepath.unlink(missing_ok=True)
    model_config = config["models"]["nind_generator_650.pt"]
    model_path = pathlib.Path(model_config["path"])
    if not model_path.exists():
        logger.info("Downloading denoiser model to %s", model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        from torch import hub

        hub.download_url_to_file(
            "https://f005.backblazeb2.com/file/modelzoo/nind/generator_650.pt",
            str(model_path),
        )
    # Invoke denoiser natively (no subprocess)
    from nind_denoise import denoise_image as _dim
    from types import SimpleNamespace as _NS

    _di_args = _NS(
        cs=None,
        ucs=None,
        overlap=6,
        input=str(stage_one_output_filepath),
        output=str(stage_one_denoised_filepath),
        batch_size=1,
        debug=False,
        exif_method="piexif",
        g_network="UtNet",
        model_path=str(model_path),
        model_parameters=None,
        max_subpixels=None,
        whole_image=False,
        pad=None,
        models_dpath=None,
    )
    _dim.run_from_args(_di_args)

    if not stage_one_denoised_filepath.exists():
        logger.error(
            "Denoiser did not output a file where expected: %s",
            stage_one_denoised_filepath,
        )
        raise RuntimeError(str(stage_one_denoised_filepath))

    clone_exif(_input_path, stage_one_denoised_filepath)

    # Stage 2 export (16-bit TIFF)
    if rldeblur and stage_two_output_filepath.is_file():
        stage_two_output_filepath.unlink()

    xmp2_src = input_xmp.with_suffix(".s2.xmp")
    xmp2_dst = stage_one_denoised_filepath.with_suffix(".s2.xmp")
    if not xmp2_src.exists():
        logger.error("Second-stage XMP sidecar missing: %s", xmp2_src)
        raise FileNotFoundError(str(xmp2_src))
    shutil.copy2(xmp2_src, xmp2_dst)

    run_cmd(
        [
            cmd_darktable,
            stage_one_denoised_filepath,
            xmp2_dst.name,
            stage_two_output_filepath.name,
            "--icc-intent",
            "PERCEPTUAL",
            "--icc-type",
            "SRGB",
            "--apply-custom-presets",
            "false",
            "--core",
            "--conf",
            "plugins/imageio/format/tiff/bpp=16",
        ],
        cwd=outpath.parent,
    )

    # Deblur stage
    ctx = Context(
        outpath=outpath,
        stage_two_output_filepath=stage_two_output_filepath,
        sigma=int(sigma),
        iteration=str(iterations),
        quality=str(quality),
        cmd_gmic=str(cmd_gmic),
        output_dir=output_dir,
        verbose=verbose,
    )
    if rldeblur:
        RLDeblur().execute(ctx)

    clone_exif(stage_one_output_filepath, outpath, verbose=verbose)

    if not _args.get("--debug"):
        for intermediate_file in [
            stage_one_output_filepath,
            stage_two_output_filepath,
            input_xmp.with_suffix(".s1.xmp"),
            input_xmp.with_suffix(".s2.xmp"),
            stage_one_denoised_filepath.with_suffix(".s2.xmp"),
        ]:
            if intermediate_file == outpath:
                continue
            pathlib.Path(intermediate_file).unlink(missing_ok=True)
