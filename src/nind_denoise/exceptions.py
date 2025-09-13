class NindError(Exception):
    """Base class for nind-denoise exceptions."""


class ConfigError(NindError):
    pass


class ExternalToolNotFound(NindError):
    """Raised when an expected external tool (darktable-cli, gmic) is missing."""


class SubprocessError(NindError):
    pass


class FileLayoutError(NindError):
    pass
