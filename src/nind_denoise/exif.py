"""EXIF utilities for nind-denoise.

Thin adapter over the exiv2 Python bindings to copy EXIF metadata between
images. Kept minimal and import-safe for tests.
"""
from __future__ import annotations

from pathlib import Path
import logging

import exiv2  # type: ignore

logger = logging.getLogger(__name__)


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
