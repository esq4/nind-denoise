import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path
import importlib
import builtins

import pytest


def _install_exiv2_stub(success=True):
    """Install a stubbed exiv2 module into sys.modules.

    If success is True, methods will behave and capture calls.
    If False, readMetadata will raise an exception.
    """
    calls = {}

    class _StubImage:
        def __init__(self, fpath):
            self.fpath = Path(fpath)
            self._exif = {"DummyKey": "DummyVal"}

        def readMetadata(self):
            if not success:
                raise ValueError("boom")
            calls.setdefault("read", []).append(self.fpath)

        def exifData(self):
            return self._exif

        def setExifData(self, data):  # noqa: N802 (external API style)
            calls.setdefault("set", []).append((self.fpath, data))

        def writeMetadata(self):
            calls.setdefault("write", []).append(self.fpath)

    def _open(fpath):
        return _StubImage(fpath)

    mod = ModuleType("exiv2")
    mod.ImageFactory = SimpleNamespace(open=_open)
    sys.modules["exiv2"] = mod
    return calls


def test_clone_exif_success(tmp_path, monkeypatch):
    calls = _install_exiv2_stub(success=True)
    exif = importlib.import_module("nind_denoise.exif")
    # Monkeypatch the exiv2 binding used by the module directly to ensure our stub is used
    exif.exiv2 = sys.modules["exiv2"]

    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    # Actual content doesn't matter for the exiv2 stub
    src.write_bytes(b"src")
    dst.write_bytes(b"dst")

    exif.clone_exif(src, dst)
    assert calls.get("read") and calls.get("write")


def test_clone_exif_error_propagates(tmp_path):
    _install_exiv2_stub(success=False)
    exif = importlib.import_module("nind_denoise.exif")
    # Monkeypatch the exiv2 binding used by the module directly to ensure our stub is used
    exif.exiv2 = sys.modules["exiv2"]

    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    src.write_bytes(b"src")
    dst.write_bytes(b"dst")

    with pytest.raises(Exception):
        exif.clone_exif(src, dst, verbose=True)
