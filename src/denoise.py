#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import pathlib

import typer

logger = logging.getLogger(__name__)

# valid_extensions is sourced from the pipeline to keep a single definition
try:
    from nind_denoise.pipeline import valid_extensions  # type: ignore
except ModuleNotFoundError:
    # Fallback dynamic import for src-layout checkouts
    import importlib.machinery as _ilm
    import importlib.util as _ilu

    _pth = pathlib.Path(__file__).resolve().parent / "nind_denoise" / "pipeline.py"
    _ldr = _ilm.SourceFileLoader("pipeline_local_validext", str(_pth))
    _spec = _ilu.spec_from_loader(_ldr.name, _ldr)
    _mod = _ilu.module_from_spec(_spec)
    import sys as _sys
    _sys.modules[_ldr.name] = _mod
    _ldr.exec_module(_mod)
    valid_extensions = _mod.valid_extensions  # type: ignore


def cli(
    raw_image: pathlib.Path = typer.Argument(
        ..., help="Path to a RAW image file or directory."
    ),
    output_path: pathlib.Path | None = typer.Option(
        None,
        "--output-path",
        "-o",
        help="Where to save the result (defaults to current directory).",
    ),
    extension: str = typer.Option(
        "jpg", "--extension", "-e", help="Output file extension."
    ),
    dt: pathlib.Path | None = typer.Option(
        None,
        "--dt",
        "-d",
        help="Path to darktable-cli. Use this only if not automatically found.",
    ),
    gmic: pathlib.Path | None = typer.Option(
        None,
        "--gmic",
        "-g",
        help="Path to gmic. Use this only if not automatically found.",
    ),
    quality: int = typer.Option(
        90, "--quality", "-q", help="JPEG compression quality."
    ),
    nightmode: bool = typer.Option(
        False, "--nightmode", help="Use for very dark images."
    ),
    no_deblur: bool = typer.Option(
        False, "--no_deblur", help="Do not perform RL-deblur."
    ),
    debug: bool = typer.Option(False, "--debug", help="Keep intermediate files."),
    sigma: int = typer.Option(1, "--sigma", help="sigma to use for RL-deblur."),
    iterations: int = typer.Option(
        10, "--iterations", help="Number of iterations for RL-deblur."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
):
    """Command-line interface entry point for nind-denoise.

    This Typer command accepts a RAW image or directory and orchestrates the
    denoise pipeline with options for output, tools, and deblurring.
    """
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

    from nind_denoise.pipeline import run_pipeline

    if raw_image.is_dir():
        for file in raw_image.iterdir():
            if file.suffix.lower() in valid_extensions:
                logger.info(
                    "----------------------- %s -------------------------", file.name
                )
                run_pipeline(args, file)
    else:
        run_pipeline(args, raw_image)


if __name__ == "__main__":
    typer.run(cli)
