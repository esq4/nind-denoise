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

from .config import Config, Tool, Tools, logger

__all__ = ["Config", "Tool", "Tools", "logger"]
