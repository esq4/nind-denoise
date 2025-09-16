"""nind_denoise package public API surface.

This package exposes a stable CLI and re-exports a few convenience symbols for
advanced users and tests. The stable user-facing API remains the CLI.
"""

__version__ = "0.3.1"

from .config import valid_extensions  # noqa: F401

# Keep the public surface minimal; avoid importing heavy modules at import time.
from .pipeline import run_pipeline  # noqa: F401

__all__ = [
    "__version__",
    "run_pipeline",
    "valid_extensions",
]


# Lazy attribute loading to avoid importing heavy modules on package import
def __getattr__(name: str):  # pragma: no cover - simple lazy import
    if name == "denoise_image":
        import importlib

        return importlib.import_module("nind_denoise.denoise_image")
    raise AttributeError(name)
