"""Denoise pipeline orchestration and stages.

This module contains all subprocess-calling logic formerly in denoise.py.
The top-level entry point is run_pipeline(args: dict, input_path: Path).
"""

from __future__ import annotations

import copy
import io
import logging
import pathlib
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import exiv2
import yaml
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import torch
from nind_denoise.rl_pt import richardson_lucy_gaussian

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_JPEG_QUALITY = 90
DEFAULT_RL_SIGMA = 1
DEFAULT_RL_ITERATIONS = 10


def _load_cli_config(path: str | None = None) -> dict:
    cfg_path = pathlib.Path(path or "src/config/cli.yaml")
    if cfg_path.is_file():
        try:
            with io.open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


_cli_cfg = _load_cli_config()
_default_exts = [
    "3FR",
    "ARW",
    "SR2",
    "SRF",
    "CR2",
    "CR3",
    "CRW",
    "DNG",
    "ERF",
    "FFF",
    "MRW",
    "NEF",
    "NRW",
    "ORF",
    "PEF",
    "RAF",
    "RW2",
]
_exts = _cli_cfg.get("valid_extensions") or _default_exts
valid_extensions = [
    ("." + e.lower()) if not e.startswith(".") else e.lower() for e in _exts
]


@dataclass
class Context:
    outpath: pathlib.Path
    stage_two_output_filepath: pathlib.Path
    sigma: int
    iteration: str
    quality: str
    cmd_gmic: str
    output_dir: pathlib.Path
    verbose: bool = False


def run_cmd(args: Iterable[pathlib.Path | str], cwd: pathlib.Path | None = None) -> None:
    cmd = [str(a) for a in args]
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=None if cwd is None else str(cwd), check=True)


class Stage(ABC):
    @abstractmethod
    def execute(self, ctx: Context) -> None:  # pragma: no cover
        ...


class DeblurStage(Stage, ABC):
    pass


class NoOpDeblur(DeblurStage):
    def execute(self, ctx: Context) -> None:
        if ctx.verbose:
            logger.info("RL-deblur disabled; skipping.")


class RLDeblur(DeblurStage):
    def execute(self, ctx: Context) -> None:
        outpath = ctx.outpath
        stage_two_output_filepath = ctx.stage_two_output_filepath
        sigma = ctx.sigma
        iterations = ctx.iteration
        quality = ctx.quality
        cmd_gmic = ctx.cmd_gmic
        output_dir = ctx.output_dir

        args = [
            str(cmd_gmic),
            str(stage_two_output_filepath),
            "-deblur_richardsonlucy",
            f"{sigma},{iterations},1",
            "-/",
            "256",
            "cut",
            "0,255",
            "round",
            "-o",
            f"{outpath.name},{quality}",
        ]
        run_cmd(args, cwd=output_dir)
        if ctx.verbose:
            logger.info("Applied RL-deblur to: %s", outpath)


class RLDeblurPT(DeblurStage):
    """PyTorch implementation of the RL deblur stage.

    This stage mirrors the behavior of :class:`RLDeblur` but uses a pure Python
    PyTorch backend instead of invoking the external `gmic` CLI.
    """

    def execute(self, ctx: Context) -> None:
        outpath = pathlib.Path(ctx.outpath)
        s2 = pathlib.Path(ctx.stage_two_output_filepath)
        sigma = float(ctx.sigma)
        iterations = int(ctx.iteration)
        quality = int(ctx.quality)

        if not s2.exists():
            raise FileNotFoundError(f"Stage-2 input not found: {s2}")

        # Load stage-2 TIFF (or other) as RGB, obtain HxWxC uint8 array
        with Image.open(s2) as im:
            im = im.convert("RGB")
            img_np = np.array(im, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np)  # HxWxC, uint8

        # Run RL on torch
        deblur = richardson_lucy_gaussian(img_tensor, sigma=sigma, iterations=iterations)

        # Convert back to PIL and save with quality
        if deblur.dtype == torch.uint8:
            out_np = deblur.cpu().numpy()
        else:
            out_np = (deblur.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        out_img = Image.fromarray(out_np, mode="RGB")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(outpath, quality=quality)

        if ctx.verbose:
            logger.info("Applied RL-deblur (PyTorch) to: %s", outpath)


# Utility and helpers moved from denoise.py

def read_config(config_path: str = "./src/config/operations.yaml", _nightmode: bool = False, verbose: bool = False) -> dict:
    with io.open(config_path, "r", encoding="utf-8") as instream:
        var = yaml.safe_load(instream)
    if _nightmode:
        if verbose:
            logger.info("Updating ops for nightmode ...")
        nightmode_ops = ["exposure", "toneequal"]
        var["operations"]["first_stage"].extend(nightmode_ops)
        for op in nightmode_ops:
            var["operations"]["second_stage"].remove(op)
    return var


def parse_darktable_history_stack(_input_xmp: pathlib.Path, config: dict, verbose: bool = False) -> None:
    operations = config["operations"]
    with _input_xmp.open(encoding="utf-8") as f:
        sidecar_xml = f.read()
    sidecar = BeautifulSoup(sidecar_xml, "xml")
    history = sidecar.find("darktable:history")
    history_org = copy.copy(history)
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
    s1 = _input_xmp.with_suffix(".s1.xmp")
    s1.unlink(missing_ok=True)
    s1.write_text(sidecar.prettify(), encoding="utf-8")

    history.replace_with(history_org)
    history_ops = history_org.find_all("rdf:li")
    for op in reversed(history_ops):
        if (
            op["darktable:operation"] not in operations["second_stage"]
            and op["darktable:operation"] in operations["first_stage"]
        ):
            op.extract()
            if verbose:
                logger.debug("--removed: %s", op["darktable:operation"])
        elif op["darktable:operation"] in operations.get("overrides", {}):
            for key, val in operations["overrides"][op["darktable:operation"]].items():
                op[key] = val
        if verbose:
            logger.debug(
                "default: %s %s", op["darktable:operation"], op.get("darktable:enabled")
            )
    description = sidecar.find("rdf:Description")
    description["darktable:iop_order_version"] = "5"
    if description.has_attr("darktable:iop_order_list"):
        description["darktable:iop_order_list"] = (
            description["darktable:iop_order_list"]
            .replace("colorin,0,", "")
            .replace("demosaic,0", "demosaic,0,colorin,0")
        )
    s2 = _input_xmp.with_suffix(".s2.xmp")
    s2.unlink(missing_ok=True)
    s2.write_text(sidecar.prettify(), encoding="utf-8")


def clone_exif(src_file: pathlib.Path, dst_file: pathlib.Path, verbose: bool = False) -> None:
    try:
        src_image = exiv2.ImageFactory.open(str(src_file))
        src_image.readMetadata()
        dst_image = exiv2.ImageFactory.open(str(dst_file))
        dst_image.setExifData(src_image.exifData())
        dst_image.writeMetadata()
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.error("Error while copying EXIF data: %s", e)
        raise
    if verbose:
        logger.info("Copied EXIF from %s to %s", src_file, dst_file)


def get_output_extension(args) -> str:
    return ("." + args["--extension"]) if args["--extension"][0] != "." else args["--extension"]


def resolve_output_paths(input_path: pathlib.Path, output_path_opt: str | None, out_ext: str) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir = pathlib.Path(output_path_opt) if output_path_opt else input_path.parent
    outpath = (output_dir / input_path.name).with_suffix(out_ext)
    return output_dir, outpath


def get_stage_filepaths(outpath: pathlib.Path, stage: int):
    if stage == 1:
        s1 = outpath.with_stem(outpath.stem + "_s1").with_suffix(".tif")
        s1d = outpath.with_stem(outpath.stem + "_s1_denoised").with_suffix(".tif")
        return s1, s1d
    s2 = outpath.with_stem(outpath.stem + "_s2").with_suffix(".tif")
    return s2


def get_command_paths(_args: dict):
    cmd_darktable = _args.get("--dt") or _cli_cfg.get("darktable_cli", "darktable-cli")
    cmd_gmic = _args.get("--gmic") or _cli_cfg.get("gmic", "gmic")
    return cmd_darktable, cmd_gmic


def run_pipeline(_args: dict, _input_path: pathlib.Path) -> None:
    verbose = _args.get("--verbose", False)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Processing %s", _input_path)

    output_extension = get_output_extension(_args)
    output_dir, outpath = resolve_output_paths(_input_path, _args.get("--output-path"), output_extension)

    input_xmp = _input_path.with_suffix(_input_path.suffix + ".xmp")
    sigma = int(_args.get("--sigma") or DEFAULT_RL_SIGMA)
    quality = int(_args.get("--quality") or DEFAULT_JPEG_QUALITY)
    iterations = int(_args.get("--iterations") or DEFAULT_RL_ITERATIONS)

    stage_one_output_filepath, stage_one_denoised_filepath = get_stage_filepaths(outpath, 1)
    stage_two_output_filepath = get_stage_filepaths(outpath, 2)

    config = read_config(verbose=verbose)
    cmd_darktable, cmd_gmic = get_command_paths(_args)

    cmd_darktable = pathlib.Path(str(cmd_darktable))
    cmd_gmic = pathlib.Path(str(cmd_gmic))

    rldeblur = True
    if not cmd_gmic.exists() or _args.get("--no_deblur"):
        logger.warning("gmic (%s) does not exist or --no_deblur is set, disabled RL-deblur", cmd_gmic)
        rldeblur = False
        stage_two_output_filepath = outpath

    if not cmd_darktable.exists():
        logger.error("darktable-cli (%s) does not exist or is not accessible.", cmd_darktable)
        raise RuntimeError(f"darktable-cli not found at {cmd_darktable}")

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
            logger.error("Too many files with the same name already exist in %s", outpath.parent)
            raise FileExistsError(str(outpath.parent))

    parse_darktable_history_stack(input_xmp, config=config, verbose=verbose)

    # Stage 1 export (32-bit TIFF)
    stage_one_output_filepath.unlink(missing_ok=True)
    run_cmd([
        cmd_darktable,
        _input_path,
        input_xmp.with_suffix(".s1.xmp"),
        stage_one_output_filepath.name,
        "--apply-custom-presets",
        "false",
        "--core",
        "--conf",
        "plugins/imageio/format/tiff/bpp=32",
    ], cwd=outpath.parent)

    if not stage_one_output_filepath.exists():
        logger.error("First-stage export not found: %s", stage_one_output_filepath)
        raise ChildProcessError(str(stage_one_output_filepath))

    # Denoiser
    stage_one_denoised_filepath.unlink(missing_ok=True)
    model_config = config["models"]["nind_generator_650.pt"]
    model_path = pathlib.Path(model_config["path"])
    if not model_path.exists():
        logger.info("Downloading denoiser model to %s", model_path)
        from torch import hub
        hub.download_url_to_file(
            "https://f005.backblazeb2.com/file/modelzoo/nind/generator_650.pt",
            str(model_path),
        )
    # Invoke denoiser natively (no subprocess)

    _dim = import_denoise_image()
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
    # autodetects cs/ucs/network as needed and runs
    _dim.run_from_args(_di_args)

    if not stage_one_denoised_filepath.exists():
        logger.error("Denoiser did not output a file where expected: %s", stage_one_denoised_filepath)
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

    run_cmd([
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
    ], cwd=outpath.parent)

    # Deblur stage
    deblur_stage = RLDeblur() if rldeblur else NoOpDeblur()
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
    deblur_stage.execute(ctx)

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


# Backwards-compatibility alias

denoise_file = run_pipeline
