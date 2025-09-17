<<<<<<< Updated upstream
import types
import pytest

class _FakeExifImage:
    def __init__(self):
        self._data = {'Exif.Image.Make': 'UnitTest'}
    def readMetadata(self):
        return None
    def exifData(self):
        return self._data
    def setExifData(self, data):
        self._data = data
    def writeMetadata(self):
        return None

class _FakeImageFactory:
    def __init__(self):
        self._last = _FakeExifImage()
    def open(self, path):
        return _FakeExifImage()

@pytest.fixture
def fake_exiv2_module():
    fake = types.SimpleNamespace(ImageFactory=_FakeImageFactory())
    return fake

@pytest.fixture
def sample_xmp(tmp_path):
    # minimal XMP with a few operations
    content = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:darktable='http://darktable.sf.net/' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n"
        "  <rdf:RDF>\n"
        "    <rdf:Description darktable:iop_order_version='3' darktable:iop_order_list='demosaic,0,colorin,0,exposure,0,toneequal,0,flip,0'>\n"
        "      <darktable:history>\n"
        "        <rdf:li darktable:num='1' darktable:operation='demosaic' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='2' darktable:operation='colorin' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='3' darktable:operation='exposure' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='4' darktable:operation='toneequal' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='5' darktable:operation='flip' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='6' darktable:operation='unlistedop' darktable:enabled='1'/>\n"
        "      </darktable:history>\n"
        "    </rdf:Description>\n"
        "  </rdf:RDF>\n"
        "</x:xmpmeta>\n"
    )
    xmp = tmp_path / 'img.RAF.xmp'
    xmp.write_text(content, encoding='utf-8')
    return xmp
=======
import sys
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Dict, Optional

import pytest
import yaml
=======
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882

# Ensure the repository's src directory is importable for package imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
<<<<<<< HEAD


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
=======
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
>>>>>>> Stashed changes
