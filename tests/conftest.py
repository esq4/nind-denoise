import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml

# Ensure the repository's src directory is importable for package imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def create_test_config():
    """Factory fixture to create temporary config files for testing."""

    def _create_config(
        tmp_path: Path,
        models: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
        operations: Optional[Dict[str, Any]] = None,
        nightmode_ops: Optional[list] = None,
    ) -> Path:
        """Create a temporary config file with specified content."""
        if models is None:
            models = {"test_model": {"path": "/fake/model/path", "default": True}}

        if tools is None:
            # Create fake tool files for validation
            fake_gmic = tmp_path / "fake_gmic.exe"
            fake_dt = tmp_path / "fake_dt.exe"
            fake_gmic.write_bytes(b"")
            fake_dt.write_bytes(b"")

            tools = {
                "windows": {
                    "gmic": {"path": str(fake_gmic), "args": []},
                    "darktable": {"path": str(fake_dt), "args": []},
                },
                "posix": {
                    "gmic": {"path": str(fake_gmic), "args": []},
                    "darktable": {"path": str(fake_dt), "args": []},
                },
            }

        if operations is None:
            operations = {
                "operations": {
                    "first_stage": ["demosaic", "flip"],
                    "second_stage": ["colorout", "sharpen"],
                }
            }

        if nightmode_ops is None:
            nightmode_ops = ["sharpen"]

        config_data = {
            "models": models,
            "tools": tools,
            "operations": operations,
            "nightmode_ops": nightmode_ops,
        }

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data))
        return config_file

    return _create_config
