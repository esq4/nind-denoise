#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Command-line interface for NIND-denoise

This module provides a Typer-based CLI for restoring and enhancing "natural images" (i.e., real photos). In this
initial version, a neural network denoising model is paired with more traditional image processing methods.
network is s combined with traditional image processing techniques.

The CLI supports both single file and batch directory processing, with
configurable output paths, quality settings, and deblurring parameters.

Example:
    Process a single RAW file with default settings:
        $ python denoise.py image.cr2

    Batch process a directory with custom output and nightmode:
        $ python denoise.py /path/to/raw/images -o /output/dir --nightmode

    Process with custom deblurring parameters:
        $ python denoise.py image.raw --sigma 2 --iterations 15

References:
    Benoit Brummer, Christophe De Vleeschouwer, "Natural Image Noise Dataset", Proceedings of the IEEE/CVF
        Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2019)

    W. Richardson, "Bayesian-Based Iterative Method of Image Restoration*," J. Opt. Soc. Am.  62, 55-59 (1972).

"""
import pathlib

import typer


def _build_args_dict(
    output_path: pathlib.Path | None,
    extension: str,
    dt: pathlib.Path | None,
    gmic: pathlib.Path | None,
    quality: int,
    nightmode: bool,
    no_deblur: bool,
    debug: bool,
    sigma: int,
    iterations: int,
    verbose: bool,
) -> dict:
    """Construct the args dict consumed by the pipeline.

    Split out to keep the CLI thin and improve readability/testability.
    """
    return {
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


def _process_inputs(raw_image: pathlib.Path, args: dict) -> None:
    """Dispatch processing over a single file or a directory tree.

    This function handles both single file and batch directory processing by
    determining the input type and iterating over valid RAW files when a
    directory is provided. For directories, only files with extensions
    matching the valid_extensions configuration are processed.

    Args:
        raw_image (pathlib.Path): Path to either a single RAW image file or
            a directory containing RAW images to be processed.
        args (dict): Dictionary containing all CLI arguments and options
            passed to the run_pipeline function. Keys match CLI parameter
            names (e.g., "--output-path", "--quality", "--nightmode").

    Note:
        The function uses lazy imports to avoid loading heavy dependencies
        unless actual processing is required. Directory traversal is
        non-recursive and only processes files in the immediate directory.
    """
    from nind_denoise.pipeline import run_pipeline

    if raw_image.is_dir():
        for file in raw_image.iterdir():
            run_pipeline(args, file)
    else:
        run_pipeline(args, raw_image)


def cli(
    raw_image: pathlib.Path = typer.Argument(..., help="Path to a RAW image file or directory."),
    output_path: pathlib.Path | None = typer.Option(
        None,
        "--output-path",
        "-o",
        help="Where to save the result (defaults to current directory).",
    ),
    extension: str = typer.Option("jpg", "--extension", "-e", help="Output file extension."),
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
    quality: int = typer.Option(90, "--quality", "-q", help="JPEG compression quality."),
    nightmode: bool = typer.Option(False, "--nightmode", help="Use for very dark images."),
    no_deblur: bool = typer.Option(False, "--no_deblur", help="Do not perform RL-deblur."),
    debug: bool = typer.Option(False, "--debug", help="Keep intermediate files."),
    sigma: int = typer.Option(1, "--sigma", help="sigma to use for RL-deblur."),
    iterations: int = typer.Option(10, "--iterations", help="Number of iterations for RL-deblur."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
):
    """Command-line interface entry point for NIND neural image denoising.

    This Typer-based CLI orchestrates a comprehensive RAW image denoising
    pipeline.
    The processing workflow includes:
    1. RAW file conversion to 32-bit TIFF using darktable-cli
    2. Neural network-based denoising using pre-trained models
    3. Conversion to 16-bit TIFF for deblurring stage
    4. Optional Richardson-Lucy deconvolution for sharpening
    5. Final output generation in specified format and quality

    Args:
        raw_image (pathlib.Path): Path to a RAW image file or directory
            containing RAW files. Supported formats depend on darktable-cli
            capabilities and include common RAW extensions like .cr2, .nef,
            .arw, .dng, etc.
        output_path (pathlib.Path | None, optional): Destination directory
            for processed images. If None, outputs are saved to the current
            working directory. Defaults to None.
        extension (str, optional): Output file format extension. Supported
            formats include "jpg", "png", "tiff". Quality settings apply
            only to JPEG output. Defaults to "jpg".
        dt (pathlib.Path | None, optional): Custom path to darktable-cli
            executable. Use only if the tool is not in PATH or requires
            a specific version. Defaults to None (auto-detection).
        gmic (pathlib.Path | None, optional): Custom path to G'MIC executable.
            G'MIC is used for advanced image processing operations. Use only
            if not automatically detected. Defaults to None (auto-detection).
        quality (int, optional): JPEG compression quality (1-100). Higher
            values produce better quality but larger files. Only applies to
            JPEG output format. Defaults to 90.
        nightmode (bool, optional): Enable specialized processing for very
            dark images. Adjusts denoising parameters and possibly model
            selection for low-light scenarios. Defaults to False.
        no_deblur (bool, optional): Skip Richardson-Lucy deblurring stage.
            Deblurring can improve sharpness but increases processing time
            and may introduce artifacts in some cases. Defaults to False.
        debug (bool, optional): Preserve intermediate files for debugging
            and analysis. Useful for troubleshooting pipeline issues but
            consumes additional disk space. Defaults to False.
        sigma (int, optional): Standard deviation parameter for Richardson-Lucy
            deblurring kernel. Higher values create stronger deblurring effect
            but may amplify noise. Valid range typically 1-5. Defaults to 1.
        iterations (int, optional): Number of Richardson-Lucy iterations.
            More iterations provide stronger deblurring but increase processing
            time and potential artifacts. Typical range 5-20. Defaults to 10.
        verbose (bool, optional): Enable detailed logging output for
            monitoring pipeline progress and debugging issues. Defaults to False.

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
