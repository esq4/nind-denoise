"""nind_denoise package public API surface.

This package exposes a stable CLI and re-exports a few convenience symbols for
advanced users and tests. The stable user-facing API remains the CLI.
"""

from __future__ import annotations

__version__ = "0.3.1"

# Keep the public surface minimal; avoid importing heavy modules at import time.
from . import config, libs, pipeline, train

__all__ = ["__version__", pipeline, train, libs, config]


class NindError(Exception):
    """Base class for brummer2019-denoise exceptions."""


class ConfigError(NindError):
    pass


class ExternalToolNotFound(NindError):
    """Raised when an expected external tool (darktable-cli, gmic) is missing."""


class SubprocessError(NindError):
    pass


class FileLayoutError(NindError):
    pass


class ConfigurationError(Exception):
    """Raised when configuration loading fails."""

    pass
