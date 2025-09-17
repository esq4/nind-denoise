from __future__ import annotations


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
