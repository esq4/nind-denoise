#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import pathlib
import typer

logger = logging.getLogger(__name__)

def _process_inputs(raw_image: pathlib.Path, args: dict) -> None:
    """Dispatch processing over a single file or a directory tree."""
    from nind_denoise.pipeline import run_pipeline 
    from nind_denoise.config import valid_extensions

    if raw_image.is_dir():
        for file in raw_image.iterdir():
            if file.suffix.lower() in valid_extensions:
                logger.info(
                    "----------------------- %s -------------------------", file.name
                )
                run_pipeline(args, file)
    else:
        run_pipeline(args, raw_image)


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

    # Import and process lazily to avoid heavy deps on --help
    _process_inputs(raw_image, args)


if __name__ == "__main__":
    typer.run(cli)
