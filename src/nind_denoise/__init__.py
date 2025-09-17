"""nind_denoise package public API surface.

This package exposes a stable CLI and re-exports a few convenience symbols for
advanced users and tests. The stable user-facing API remains the CLI.
"""

__version__ = "0.3.1"

# Keep the public surface minimal; avoid importing heavy modules at import time.
from . import config, libs, pipeline, train

__all__ = ["__version__", pipeline, train, libs, config]
