"""nind_denoise package public API surface.

This package exposes a stable CLI and re-exports a few convenience symbols for
advanced users and tests. The stable user-facing API remains the CLI.
"""

__version__ = "0.2.0"

# Re-export the denoise_image submodule for convenient import
# Allows: from nind_denoise import denoise_image
from . import denoise_image
from .pipeline import run_pipeline
from .config import valid_extensions

__all__ = [
    "__version__",
    "denoise_image",
    "run_pipeline",
    "valid_extensions",
]
