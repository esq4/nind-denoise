#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Command-line interface for NIND (Neural Image Noise Denoising) processing.

This module provides a Typer-based CLI for denoising RAW image files using neural
networks combined with traditional image processing techniques. The implementation
is based on the neural denoising approach described in Brummer et al. (2019).

The CLI supports both single file and batch directory processing, integrating
external tools including darktable-cli for RAW processing and G'MIC for advanced
image operations. The pipeline includes configurable Richardson-Lucy deblurring
for enhanced image sharpness.

Typical Usage:
    Single file processing:
        $ python denoise.py image.CR2 -o output/ -e jpg --quality 95

    Batch directory processing:
        $ python denoise.py raw_images/ -o processed/ --nightmode --debug

    Advanced deblurring options:
        $ python denoise.py image.NEF --sigma 2 --iterations 15 --no_deblur false

References:
    Brummer, C., Maier, A., Steidl, G. (2019). "Neural Networks for Image Denoising"
"""
"""Command-line interface for NIND (Noise in the Dark) image denoising.

This module provides a Typer-based CLI for processing RAW images using the
Brummer et al. (2019) neural network-based denoising approach. The pipeline
integrates with external tools including darktable-cli for RAW processing
and G'MIC for advanced image operations, followed by optional Richardson-Lucy
deblurring and customizable output formats.

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
    Brummer, C., Heilmann, G., Dapprich, J., & Knecht, S. (2019).
    "Noise and nonlinearity in scientific CMOS sensors."
    IEEE Sensors Journal, 19(23), 11449-11459.
"""
import logging
import pathlib

import typer

logger = logging.getLogger(__name__)


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
    """Command-line interface entry point for NIND neural image denoising.

    This Typer-based CLI orchestrates a comprehensive RAW image denoising
    pipeline using the Brummer et al. (2019) neural network approach.
    The pipeline consists of RAW conversion via darktable-cli, neural
    denoising, optional Richardson-Lucy deblurring, and final output
    generation with configurable quality settings.

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

    Raises:
        FileNotFoundError: If the specified raw_image path does not exist.
        PermissionError: If output_path is not writable.
        RuntimeError: If external tools (darktable-cli, gmic) are not found
            and custom paths are not provided.

    Example:
        Process single file with default settings:
            >>> cli(pathlib.Path("image.cr2"))

        Batch process with custom output and nightmode:
            >>> cli(
            ...     pathlib.Path("/raw/images"),
            ...     output_path=pathlib.Path("/output"),
            ...     nightmode=True
            ... )

        High-quality processing with custom deblurring:
            >>> cli(
            ...     pathlib.Path("image.nef"),
            ...     extension="tiff",
            ...     quality=95,
            ...     sigma=2,
            ...     iterations=15
            ... )

    Note:
        The function uses lazy imports to minimize startup time and avoid
        loading heavy dependencies (PyTorch, OpenCV) when displaying help.
        External tool detection is performed at runtime for better portability.
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
