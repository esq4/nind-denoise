"""Refactored configuration loading and options for brummer2019-denoise.

This module provides:
- Configuration data structures
- YAML config loading with fallbacks
- External tool resolution
- File extension validation
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional

import yaml

from nind_denoise.exceptions import ConfigurationError, ExternalToolNotFound

logger = logging.getLogger(__name__)


class Tool:
    """External command-line tool wrapper with validation and execution capabilities.

    This class provides a validated wrapper for external command-line tools,
    ensuring the tool executable exists before creating the instance and
    providing methods for argument management and command execution.

    Attributes:
        path (Path): Path to the external tool executable.
    """

    def __init__(self, path: Path, args):
        """Initialize a Tool instance with path validation.

        Args:
            path (Path): Path to the external tool executable.
            args (list[str]): Initial command-line arguments for the tool.

        Raises:
            ExternalToolNotFound: If the tool executable does not exist at the given path.
        """
        if not path.exists(follow_symlinks=True):
            raise ExternalToolNotFound(path)
        self.path = path
        self._args = args

    @property
    def args(self) -> list[str]:
        """Get the current command-line arguments for the tool.

        Returns:
            list[str]: Copy of the current command-line arguments.
        """
        return self._args

    @args.setter
    def args(self, new_args: list[str]):
        """Set new command-line arguments for the tool.

        Args:
            new_args (list[str]): New command-line arguments to set for the tool.

        Note:
            This method creates a copy of the provided arguments to prevent
            external modifications to the internal argument list.
        """
        self._args = new_args.copy()
        return self

    def append_arg(self, new_arg: str):
        """Append a single argument to the current argument list.

        Args:
            new_arg (str): The command-line argument to append to the existing list.
        """
        self._args.append(new_arg)

    def run_command(self) -> None:
        """Execute a subprocess command with proper error handling.

        Raises:
            ConfigurationError: If command execution fails.
        """
        cmd = [str(arg) for arg in self.args]
        cwd_str = str(os.curdir)

        logger.debug("Executing command: %s (cwd=%s)", " ".join(cmd), cwd_str)

        try:
            subprocess.run(cmd, cwd=cwd_str, check=True)
        except subprocess.CalledProcessError as exc:
            error_msg = (
                f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}"
            )
            raise ConfigurationError(error_msg) from exc


class Tools:
    """Platform-aware collection of external tools.

    This class manages a collection of external command-line tools, automatically
    selecting the appropriate tool configuration based on the current platform
    (Windows vs POSIX systems). Tools are instantiated only for the current platform.

    The class dynamically creates attributes for each tool based on the platform
    configuration, with tool names prefixed by an underscore.
    """

    def __init__(self, tools_cfg: Dict[str, Dict[str, str]]):
        """Initialize the Tools collection with platform-specific configuration.

        Args:
            tools_cfg (Dict[str, Dict[str, str]]): Dictionary mapping platform keys
                to tool configurations. Each tool configuration should contain
                'path' and 'args' keys.

        Note:
            Only tools matching the current platform will be instantiated as Tool objects.
        """
        self._platform_key = self._get_platform_key()

        for pk, platform_tools in tools_cfg.items():
            if pk == self._platform_key:
                for tool_name, tool_config in platform_tools.items():
                    setattr(
                        self,
                        "_" + str(tool_name),
                        Tool(Path(tool_config["path"]), tool_config["args"]),
                    )

    @staticmethod
    def _get_platform_key() -> str:
        """Get the platform key for configuration lookup.

        This method determines the current operating system and returns the
        appropriate key for platform-specific tool configuration lookup.

        Returns:
            str: Either "windows" for Windows systems or "posix" for Unix-like systems
                (Linux, macOS, etc.).
        """
        system = platform.system().lower()
        return "windows" if "windows" in system else "posix"


class Config:
    """Primary configuration manager for nind-denoise applications.

    This class loads and manages configuration data from YAML files, providing
    access to model configurations, external tools, and processing operations.
    It supports runtime configuration options such as model selection and
    nightmode operation adjustments.

    The configuration system is designed to handle:
    - Model path resolution and default model selection
    - Platform-specific external tool management
    - Dynamic operation configuration with nightmode support
    - YAML-based configuration file loading with validation

    Attributes:
        _config (dict): Raw configuration data loaded from YAML file.
        _model_name (str): Name of the currently selected model.
        _tools (Tools): Platform-specific collection of external tools.
        _operations (dict): Cached operations configuration.
    """

    def __init__(
        self,
        path: Path | str = "",
        nightmode: Optional[bool] = False,
        model: Optional[str | Path] = "",
        verbose: Optional[bool] = False,
    ):
        """Initialize the Config instance with YAML configuration loading.

        Args:
            path (Path | str): Path to the YAML configuration file. If None,
                defaults to 'config.yaml' in the same directory as this module.
            model (Optional[str]): Name of the specific model to use. If None,
                the default model from the configuration will be selected.
            nightmode (Optional[bool]): Whether to enable nightmode operation
                adjustments. This parameter is stored but currently unused in
                the initialization process.
            verbose (Optional[bool]): Whether to enable verbose mode.

        Note:
            The configuration file is expected to contain 'models', 'tools',
            'operations', and 'nightmode_ops' sections. The model selection
            logic will find the first model marked as default if no specific
            model is provided.
        """
        self.verbose = verbose
        self.nightmode = nightmode
        config_path = Path(__file__).parent / "config.yaml" if path == "" else path
<<<<<<< HEAD
        model = model.name if type(model) == "Path" else model
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        if model != "":
            self._model_name = model
        else:
=======
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        if model != "":
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
            for k, v in self._config["models"].items():
                if v.get("default", False):
                    self._model_name = k
                    break
<<<<<<< HEAD
                else:
                    pass
=======
        else:
            self._model_name = model
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
        self._tools = Tools(self._config["tools"])
        self._operations = None

    @property
    def models(self) -> Dict[str, Dict[str, str]]:
        """Get the available model configurations.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping model names to their
                configuration dictionaries. Each model configuration typically
                contains keys like 'path', 'default', and other model-specific
                settings.

        Note:
            This property provides direct access to the models section of the
            loaded configuration. In future versions, this may be replaced by
            a dedicated Model class for better encapsulation.
        """
        return self._config["models"]

    @property
    def model_path(self) -> Path:
        """Get the file system path to the currently selected model.

        Returns:
            Path: Path object pointing to the currently selected model file
                or directory, as specified in the model's configuration.

<<<<<<< HEAD
        Raises:
            ConfigurationError: If no valid model is selected or model not found.

=======
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
        Note:
            This property uses the currently selected model name (stored in
            _model_name) to look up the path from the models configuration.
        """
<<<<<<< HEAD
        if not self._model_name or self._model_name not in self.models:
            raise ConfigurationError(f"No valid model selected: '{self._model_name}'")
=======
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
        return Path(self.models[self._model_name]["path"])

    @property
    def tools(self) -> Tools:
        """Get the platform-specific collection of external tools.

        Returns:
            Tools: Platform-aware collection of external command-line tools
                that have been validated and are ready for use. Only tools
                compatible with the current platform are included.

        Note:
            The tools are initialized during Config construction based on
            the 'tools' section of the configuration file and the current
            platform detection.
        """
        return self._tools

    @property
    def operations(self) -> Dict:
        """Get the operations configuration with nightmode adjustments applied.

        This property returns the operations configuration, potentially modified
        based on nightmode settings. It handles moving nightmode operations from
        the second stage to the first stage and caches the result for performance.

        Returns:
            Dict: Operations configuration dictionary with nightmode adjustments
                applied. Contains 'operations' key with 'first_stage' and
                'second_stage' lists of operation names.

        Note:
            The result is cached after the first access. Nightmode operations
            are moved from second_stage to first_stage if they exist, and
            warnings are logged for any nightmode operations not found in
            the second stage.
        """
        # return cached version if exists
        if self._operations is not None:
            return self._operations
        else:
            ops = self._config["operations"].copy()
            nightmode_ops = self._config["nightmode_ops"]

            operations = ops.setdefault("operations", {})
            first_stage = operations.setdefault("first_stage", [])
            second_stage = operations.setdefault("second_stage", [])
            # Move nightmode ops  to first stage
            for op in nightmode_ops:
                if op not in second_stage:
<<<<<<< HEAD
                    logger.warning("%s not found in second stage ops. Ignoring...", op)
=======
                    logger.warning(
                        str(op), " not found in second stage ops. Ignoring..."
                    )
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
                if op not in first_stage:
                    first_stage.append(op)

            # Remove nightmode ops from second stage
            operations["second_stage"] = [
                op for op in second_stage if op not in nightmode_ops
            ]

            return ops
