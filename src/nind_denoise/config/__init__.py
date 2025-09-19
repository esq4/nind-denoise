"""Configuration management for nind-denoise.

This submodule provides configuration management functionality for the nind-denoise
package, including YAML configuration file loading, external tool management, and
runtime configuration options.

The main components are:
    Config: Primary configuration class that loads and manages application settings
    Tool: Wrapper for external command-line tools with validation and execution
    Tools: Platform-aware collection of external tools
    logger: Module logger for configuration-related messages

Example:
    >>> from nind_denoise.config import Config
    >>> config = Config("config.yaml",model="my_model",nightmode=False)
    >>> model_path = config.model_path
"""

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from .config import Config, Tool, Tools, logger
from .. import SubprocessError


def run_cmd(args: Sequence[str | Path], cwd: Optional[Path] = None) -> None:
    """Run a subprocess command with robust default behavior.

    - Ensures all args are converted to str
    - Converts cwd Path to str
    - Raises SubprocessError on failure
    """
    str_args = [str(a) for a in args]
    str_cwd = str(cwd) if cwd is not None else None
    try:
        subprocess.run(str_args, cwd=str_cwd, check=True, text=True, capture_output=False)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - error path
        raise SubprocessError(str(exc)) from exc


__all__ = ["Config", "Tool", "Tools", "logger", "run_cmd", "subprocess"]
