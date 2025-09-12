#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Huy Hoang

Denoise the raw image denoted by <filename> and save the results.

Usage:
    denoise.py [-o <outpath> | --output-path=<outpath>] [-e <e> | --extension=<e>]
                [-d <darktable> | --dt=<darktable>] [-g <gmic> | --gmic=<gmic>] [ -q <q> | --quality=<q>]
                [--nightmode ] [ --no_deblur ] [ --debug ] [ --sigma=<sigma> ] [ --iterations=<iter> ]
                [-v | --verbose] <raw_image>
    denoise.py (help | -h | --help)
    denoise.py --version

Options:

  -o <outpath> --output-path=<outpath>  Where to save the result (defaults to current directory).
  -e <e> --extension=<e>                Output file extension. Supported formats are ....? [default: jpg].
  --dt=<darktable>                      Path to darktable-cli. Use this only if not automatically found.
  -g <gmic> --gmic=<gmic>               Path to gmic. Use this only if not automatically found.
  -q <q> --quality=<q>                  JPEG compression quality. Lower produces a smaller file at the cost of more artifacts. [default: 90].
  --nightmode                           Use for very dark images. Normalizes brightness (exposure, tonequal) before denoise [default: False].
  --no_deblur                           Do not perform RL-deblur [default: false].
  --debug                               Keep intermedia files.
  --sigma=<sigma>                       sigma to use for RL-deblur. Acceptable values are ....? [default: 1].
  --iterations=<iter>                   Number of iterations to perform during RL-deblur. Suggest keeping this to ...? [default: 10].

  -v --verbose
  --version                             Show version.
  -h --help                             Show this screen.
"""
import copy
import io
import logging
import os
import pathlib
import sys
import subprocess
import shutil

import exiv2
import yaml
from bs4 import BeautifulSoup
import typer

logger = logging.getLogger(__name__)

# load CLI config (valid extensions, tool defaults) from YAML


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

# define RAW extensions from config with safe fallback
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


class ToolNotFoundError(RuntimeError):
    pass


def normalize_exts(exts: str | list[str] | set[str]) -> set[str]:
    if isinstance(exts, str):
        exts = [exts]
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}


def check_good_input(path: pathlib.Path, extensions=None) -> bool:
    """
    Check whether the given path is a valid file with one of the specified extensions.

    :param path: The path to check.
    :param extensions: A list/set of acceptable file extensions or a single string.
    :return: True if valid, else False.
    """
    exts = normalize_exts(extensions) if extensions is not None else set()

    if not path.exists() or not path.is_file():
        logger.warning("Path is not a file or does not exist: %s", path)
        return False
    if exts and path.suffix.lower() not in exts and path.suffix.lower() != ".xmp":
        logger.info("Not a supported RAW file: %s; skipping.", path)
        return False
    return True


def clone_exif(src_file: pathlib.Path, dst_file: pathlib.Path, verbose=False) -> None:
    try:
        src_image = exiv2.ImageFactory.open(str(src_file))
        src_image.readMetadata()
        dst_image = exiv2.ImageFactory.open(str(dst_file))
        dst_image.setExifData(src_image.exifData())
        dst_image.writeMetadata()
    except Exception as e:
        if verbose:
            logger.error("Error while copying EXIF data: %s", e)
        raise
    if verbose:
        logger.info("Copied EXIF from %s to %s", src_file, dst_file)


def read_config(
    config_path="./src/config/operations.yaml", _nightmode=False, verbose=False
) -> dict:
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


def parse_darktable_history_stack(
    _input_xmp: pathlib.Path, config: dict, verbose=False
):
    operations = config["operations"]
    with _input_xmp.open() as f:
        sidecar_xml = f.read()
    sidecar = BeautifulSoup(sidecar_xml, "xml")
    # read the history stack
    history = sidecar.find("darktable:history")
    history_org = copy.copy(history)
    history_ops = history.find_all("rdf:li")
    # sort history ops
    history_ops.sort(key=lambda tag: int(tag["darktable:num"]))
    # remove ops not listed in operations["first_stage"]
    for op in reversed(history_ops):
        if op["darktable:operation"] not in operations["first_stage"]:
            op.extract()  # remove the op completely
            if verbose:
                logger.debug("--removed: %s", op["darktable:operation"])
        else:
            # for "flip": don't remove, only disable
            if op["darktable:operation"] == "flip":
                op["darktable:enabled"] = "0"
                if verbose:
                    logger.debug("default: %s", op["darktable:operation"])
    # write first stage sidecar
    s1 = _input_xmp.with_suffix(".s1.xmp")
    s1.unlink(missing_ok=True)
    s1.write_text(sidecar.prettify(), encoding="utf-8")

    # restore the history stack to original
    history.replace_with(history_org)
    history_ops = history_org.find_all("rdf:li")
    # filter for second stage and apply overrides
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
    # set iop_order_version to 5 (for JPEG)
    description = sidecar.find("rdf:Description")
    description["darktable:iop_order_version"] = "5"
    # bring colorin right next to demosaic (early in the stack)
    if description.has_attr("darktable:iop_order_list"):
        description["darktable:iop_order_list"] = (
            description["darktable:iop_order_list"]
            .replace("colorin,0,", "")
            .replace("demosaic,0", "demosaic,0,colorin,0")
        )
    # write second stage sidecar
    s2 = _input_xmp.with_suffix(".s2.xmp")
    s2.unlink(missing_ok=True)
    s2.write_text(sidecar.prettify(), encoding="utf-8")


def get_output_path(args, input_path):
    """Return output directory Path from args or input's parent (legacy helper)."""
    return (
        pathlib.Path(args["--output-path"])
        if args["--output-path"]
        else input_path.parent
    )


def get_output_extension(args):
    return (
        "." + args["--extension"]
        if args["--extension"][0] != "."
        else args["--extension"]
    )


def resolve_output_paths(
    input_path: pathlib.Path, output_path_opt: str | None, out_ext: str
) -> tuple[pathlib.Path, pathlib.Path]:
    if output_path_opt:
        p = pathlib.Path(output_path_opt)
        if p.suffix:
            return p.parent, p
        return p, p / input_path.with_suffix(out_ext).name
    out_dir = input_path.parent
    return out_dir, (out_dir / input_path.name).with_suffix(out_ext)


def get_stage_filepaths(outpath, stage):
    if stage == 1:
        return pathlib.Path(
            outpath.parent, outpath.stem + "_s1" + ".tif"
        ), pathlib.Path(outpath.parent, outpath.stem + "_s1_denoised" + ".tif")
    elif stage == 2:
        return pathlib.Path(outpath.parent, outpath.stem + "_s2" + ".tif")


def get_command_paths(args):
    """
    Get the paths to external command-line tools as pathlib.Path objects.
    """
    if args.get("--dt"):
        dt = args["--dt"]
    else:
        if os.name == "nt":
            dt = (
                _cli_cfg.get("tools", {}).get("windows", {}).get("darktable")
                or "C:/Program Files/darktable/bin/darktable-cli.exe"
            )
        else:
            dt = (
                _cli_cfg.get("tools", {}).get("posix", {}).get("darktable")
                or "/usr/bin/darktable-cli"
            )

    if args.get("--gmic"):
        gmic = args["--gmic"]
    else:
        if os.name == "nt":
            gmic = (
                _cli_cfg.get("tools", {}).get("windows", {}).get("gmic")
                or "~\\gmic-3.6.1-cli-win64\\gmic.exe"
            )
            gmic = os.path.expanduser(gmic)
        else:
            gmic = (
                _cli_cfg.get("tools", {}).get("posix", {}).get("gmic")
                or "/usr/bin/gmic"
            )
    return pathlib.Path(dt), pathlib.Path(gmic)


def denoise_file(_args: dict, _input_path: pathlib.Path):
    """Denoise a file using Darktable and gmic."""
    verbose = _args.get("--verbose", False)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Processing %s", _input_path)

    output_extension = get_output_extension(_args)
    output_dir, outpath = resolve_output_paths(
        _input_path, _args.get("--output-path"), output_extension
    )

    input_xmp = _input_path.with_suffix(_input_path.suffix + ".xmp")
    sigma = int(_args.get("--sigma") or 1)
    quality = int(_args.get("--quality") or 90)
    iterations = int(_args.get("--iterations") or 10)

    stage_one_output_filepath, stage_one_denoised_filepath = get_stage_filepaths(
        outpath, 1
    )
    stage_two_output_filepath = get_stage_filepaths(outpath, 2)

    config = read_config(verbose=verbose)
    cmd_darktable, cmd_gmic = get_command_paths(_args)
    logger.debug("darktable-cli: %s", cmd_darktable)

    # coerce potential strings from monkeypatched tests
    cmd_darktable = pathlib.Path(cmd_darktable)
    cmd_gmic = pathlib.Path(cmd_gmic)

    rldeblur = True
    if not cmd_gmic.exists() or _args.get("--no_deblur"):
        logger.warning(
            "gmic (%s) does not exist or --no_deblur is set, disabled RL-deblur",
            cmd_gmic,
        )
        rldeblur = False
        stage_two_output_filepath = outpath

    if not cmd_darktable.exists():
        logger.error(
            "darktable-cli (%s) does not exist or is not accessible.", cmd_darktable
        )
        raise ToolNotFoundError(f"darktable-cli not found at {cmd_darktable}")

    good_file = check_good_input(_input_path, valid_extensions) or check_good_input(
        input_xmp, ".xmp"
    )
    if not good_file:
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

    stage_one_output_filepath.unlink(missing_ok=True)

    # darktable stage 1 export (32-bit tiff)
    subprocess.run(
        [
            str(cmd_darktable),
            str(_input_path),
            str(input_xmp.with_suffix(".s1.xmp")),
            stage_one_output_filepath.name,
            "--apply-custom-presets",
            "false",
            "--core",
            "--conf",
            "plugins/imageio/format/tiff/bpp=32",
        ],
        cwd=str(outpath.parent),
        check=True,
    )

    if not stage_one_output_filepath.exists():
        logger.error("First-stage export not found: %s", stage_one_output_filepath)
        raise ChildProcessError(str(stage_one_output_filepath))

    # ========== call nind-denoise ==========
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

    subprocess.run(
        [
            sys.executable,
            str(pathlib.Path("src/nind_denoise/denoise_image.py").resolve()),
            "--network",
            "UtNet",
            "--model_path",
            str(model_path),
            "--input",
            str(stage_one_output_filepath),
            "--output",
            str(stage_one_denoised_filepath),
        ],
        check=True,
    )

    if not stage_one_denoised_filepath.exists():
        logger.error(
            "Denoiser did not output a file where expected: %s",
            stage_one_denoised_filepath,
        )
        raise RuntimeError(str(stage_one_denoised_filepath))

    clone_exif(_input_path, stage_one_denoised_filepath)

    # ========== invoke darktable-cli with second stage operations ==========
    if rldeblur and stage_two_output_filepath.is_file():
        stage_two_output_filepath.unlink()

    # Darktable on some systems fails to open XMP files outside the working directory.
    # Copy the generated .s2.xmp next to the stage-1 denoised TIFF and pass a relative name.
    xmp2_src = input_xmp.with_suffix(".s2.xmp")
    xmp2_dst = stage_one_denoised_filepath.with_suffix(".s2.xmp")
    if not xmp2_src.exists():
        logger.error("Second-stage XMP sidecar missing: %s", xmp2_src)
        raise FileNotFoundError(str(xmp2_src))
    # Ensure destination directory exists and copy
    try:
        shutil.copy2(xmp2_src, xmp2_dst)
    except Exception as _e:
        logger.error("Failed to prepare second-stage XMP in working directory: %s", _e)
        raise

    subprocess.run(
        [
            str(cmd_darktable),
            str(stage_one_denoised_filepath),
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
        cwd=str(outpath.parent),
        check=True,
    )

    # call RL-deblur via deblur stage
    try:
        from nind_denoise.pipeline import Context, NoOpDeblur, RLDeblur
    except ModuleNotFoundError:
        import importlib.machinery as _ilm
        import importlib.util as _ilu

        _pth = pathlib.Path(__file__).resolve().parent / "nind_denoise" / "pipeline.py"
        _ldr = _ilm.SourceFileLoader("pipeline_local", str(_pth))
        _spec = _ilu.spec_from_loader(_ldr.name, _ldr)
        _mod = _ilu.module_from_spec(_spec)
        _ldr.exec_module(_mod)
        Context, NoOpDeblur, RLDeblur = _mod.Context, _mod.NoOpDeblur, _mod.RLDeblur
    deblur_stage = RLDeblur() if rldeblur else NoOpDeblur()
    ctx = Context(
        outpath=outpath,
        stage_two_output_filepath=stage_two_output_filepath,
        sigma=sigma,
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
            intermediate_file.unlink(missing_ok=True)


def cli(
    raw_image: pathlib.Path = typer.Argument(..., help="Path to a RAW image file or directory."),
    output_path: pathlib.Path | None = typer.Option(None, "--output-path", "-o", help="Where to save the result (defaults to current directory)."),
    extension: str = typer.Option("jpg", "--extension", "-e", help="Output file extension."),
    dt: pathlib.Path | None = typer.Option(None, "--dt", "-d", help="Path to darktable-cli. Use this only if not automatically found."),
    gmic: pathlib.Path | None = typer.Option(None, "--gmic", "-g", help="Path to gmic. Use this only if not automatically found."),
    quality: int = typer.Option(90, "--quality", "-q", help="JPEG compression quality."),
    nightmode: bool = typer.Option(False, "--nightmode", help="Use for very dark images."),
    no_deblur: bool = typer.Option(False, "--no_deblur", help="Do not perform RL-deblur."),
    debug: bool = typer.Option(False, "--debug", help="Keep intermediate files."),
    sigma: int = typer.Option(1, "--sigma", help="sigma to use for RL-deblur."),
    iterations: int = typer.Option(10, "--iterations", help="Number of iterations for RL-deblur."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
):
    args = {
        "--output-path": str(output_path) if output_path else None,
        "--extension": extension,
        "--dt": str(dt) if dt else None,
        "--gmic": str(gmic) if gmic else None,
        "--quality": quality,
        "--nightmode": nightmode,
        "--no_deblur": no_deblur,
        "--debug": debug,
        "--sigma": sigma,
        "--iterations": iterations,
        "--verbose": verbose,
    }
    if raw_image.is_dir():
        for file in raw_image.iterdir():
            if file.suffix.lower() in valid_extensions:
                logger.info("----------------------- %s -------------------------", file.name)
                denoise_file(args, _input_path=file)
    else:
        denoise_file(args, _input_path=raw_image)


if __name__ == "__main__":
    typer.run(cli)
