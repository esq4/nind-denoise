"""Pipeline orchestration and public API for nind_denoise.pipeline package."""

from __future__ import annotations

import copy
import logging
import pathlib
from pathlib import Path
from typing import Optional

import exiv2
from bs4 import BeautifulSoup

from .base import JobContext
from .deblur import RLDeblur
from .denoise import DenoiseOptions, DenoiseStage
from .export import ExportStage
from ..config.config import Config

logger = logging.getLogger(__name__)

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
    """Validate the input file and its XMP (both must exist)."""
    input_xmp = input_path.with_suffix(input_path.suffix + ".xmp")
    good_file = input_path.exists() and input_path.is_file()
    good_xmp = input_xmp.exists() and input_xmp.is_file()

    if not (good_file and good_xmp):
        logger.error(
            "The input raw-image and its XMP were not found, or are not valid."
        )
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


def resolve_unique_output_path(outpath: pathlib.Path) -> pathlib.Path:
    """Ensure output path is unique by appending an index suffix if needed."""
    i = 1
    while outpath.exists():
        outpath = outpath.with_stem(outpath.stem + "_" + str(i))
        i += 1
        if i >= 99:
            logger.error(
                "Too many files with the same name already exist in %s", outpath.parent
            )
            raise FileExistsError(str(outpath.parent))
    return outpath


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

    cfg = Config()

    # Resolve external tools (darktable-cli, gmic)
    dt_path = pathlib.Path(_args.get("--dt")) if _args.get("--dt") else None
    gmic_path = pathlib.Path(_args.get("--gmic")) if _args.get("--gmic") else None

    rldeblur_enabled = cfg.hasattr(name="gmic") and not _args.get("--no_deblur")
    if not rldeblur_enabled:
        logger.warning(
            "gmic (%s) does not exist or --no_deblur is set, disabled RL-deblur",
            cfg.gmic if cfg.gmic else _args.get("--gmic"),
        )

    outpath = resolve_unique_output_path(outpath)

    # Create immutable Environment for new context pattern
    environment = Config(config=cfg, verbose=verbose)

    # Stage 1 export (32-bit TIFF) - using new Environment + JobContext pattern
    input_xmp = _input_path.with_suffix(_input_path.suffix + ".xmp")
    s1_xmp = stage_one_output_filepath.with_suffix(".s1.xmp")
    s1_job_ctx = JobContext(
        input_path=_input_path,
        output_path=stage_one_output_filepath,
        sigma=sigma,
        iterations=iterations,
        quality=quality,
    )
    ExportStage(
        tools,
        _input_path,
        input_xmp,
        s1_xmp,
        stage_one_output_filepath,
        1,
    ).execute_with_env(environment, s1_job_ctx)

    model_config = cfg["models"]["nind_generator_650.pt"]
    model_path = pathlib.Path(model_config["path"])
    download_model_if_needed(model_path)

    # Denoiser stage
    denoise_job_ctx = JobContext(
        input_path=stage_one_output_filepath,
        output_path=stage_one_denoised_filepath,
        sigma=sigma,
        iterations=iterations,
        quality=quality,
    )
    DenoiseStage(
        stage_one_output_filepath,
        stage_one_denoised_filepath,
        DenoiseOptions(model_path=model_path, overlap=6, batch_size=1),
    ).execute_with_env(environment, denoise_job_ctx)

    clone_exif(_input_path, stage_one_denoised_filepath)

    # Stage 2 export (16-bit TIFF)
    xmp2_dst = stage_one_denoised_filepath.with_suffix(".s2.xmp")
    s2_job_ctx = JobContext(
        input_path=stage_one_denoised_filepath,
        output_path=stage_two_output_filepath,
        sigma=sigma,
        iterations=iterations,
        quality=quality,
    )
    ExportStage(
        tools,
        stage_one_denoised_filepath,
        input_xmp,
        xmp2_dst,
        stage_two_output_filepath,
        2,
    ).execute_with_env(environment, s2_job_ctx)

    # Deblur stage
    if rldeblur_enabled:
        deblur_job_ctx = JobContext(
            input_path=stage_two_output_filepath,
            output_path=outpath,
            sigma=sigma,
            iterations=iterations,
            quality=quality,
            intermediate_path=stage_two_output_filepath,
        )
        RLDeblur().execute_with_env(environment, deblur_job_ctx)

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


def build_xmp(xmp_text: str, stage: int, *, verbose: bool = False) -> str:
    """Return a stage-specific XMP string from an input XMP string.

    - stage=1: keep only first_stage ops; disable flip.
    - stage=2: keep only second_stage ops (and any overrides); update iop order.
    """
    cfg = (Config(),)
    operations = cfg.operations

    sidecar = BeautifulSoup(xmp_text, "xml")
    history = sidecar.find("darktable:history")
    if history is None:
        raise ValueError()

    history_org = copy.copy(history)

    def _stage1() -> None:
        history_ops = history.find_all("rdf:li")
        history_ops.sort(key=lambda tag: int(tag["darktable:num"]))
        for op in reversed(history_ops):
            if op["darktable:operation"] not in operations["first_stage"]:
                op.extract()
                if verbose:
                    logger.debug("--removed: %s", op["darktable:operation"])
            else:
                if op["darktable:operation"] == "flip":
                    op["darktable:enabled"] = "0"
                    if verbose:
                        logger.debug("default: %s", op["darktable:operation"])

    def _stage2() -> None:
        history.replace_with(history_org)
        history_ops2 = history_org.find_all("rdf:li")
        for op in reversed(history_ops2):
            if (
                op["darktable:operation"] not in operations["second_stage"]
                and op["darktable:operation"] in operations["first_stage"]
            ):
                op.extract()
                if verbose:
                    logger.debug("--removed: %s", op["darktable:operation"])
            elif op["darktable:operation"] in operations.get("overrides", {}):
                for key, val in operations["overrides"][
                    op["darktable:operation"]
                ].items():
                    op[key] = val
            if verbose:
                logger.debug(
                    "default: %s %s",
                    op["darktable:operation"],
                    op.get("darktable:enabled"),
                )
        description = sidecar.find("rdf:Description")
        if description:
            description["darktable:iop_order_version"] = "5"
            if description.has_attr("darktable:iop_order_list"):
                description["darktable:iop_order_list"] = (
                    description["darktable:iop_order_list"]
                    .replace("colorin,0,", "")
                    .replace("demosaic,0", "demosaic,0,colorin,0")
                )

    if stage == 1:
        _stage1()
    else:
        _stage2()

    return sidecar.prettify()


def write_xmp_file(
    src_xmp_path: Path, dst_xmp_path: Path, stage: int, *, verbose: bool = False
) -> None:
    """Read XMP from src, transform for stage, and write to dst."""
    if not src_xmp_path.exists():
        raise FileNotFoundError(str(src_xmp_path))
    xmp_text = src_xmp_path.read_text(encoding="utf-8")
    out_text = build_xmp(xmp_text, stage, verbose=verbose)
    dst_xmp_path.unlink(missing_ok=True)
    dst_xmp_path.write_text(out_text, encoding="utf-8")


def clone_exif(src_file: Path, dst_file: Path, verbose: bool = False) -> None:
    """Clone EXIF metadata from src_file to dst_file using exiv2.

    Raises the underlying exception if exiv2 fails; emits a helpful log when
    verbose is enabled.
    """
    try:
        src_image = exiv2.ImageFactory.open(str(src_file))
        src_image.readMetadata()
        dst_image = exiv2.ImageFactory.open(str(dst_file))
        dst_image.setExifData(src_image.exifData())
        dst_image.writeMetadata()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if verbose:
            logger.error("Error while copying EXIF data: %s", exc)
        raise
    if verbose:
        logger.info("Copied EXIF from %s to %s", src_file, dst_file)
