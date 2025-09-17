import sys
import types


def _import_pipeline_with_stub():
    # Stub exiv2 so importing nind_denoise.pipeline doesn’t require the external binding
    sys.modules.setdefault("exiv2", types.SimpleNamespace())
    import importlib

    return importlib.import_module("nind_denoise.pipeline")


def test_validate_input_file_requires_xmp(tmp_path):
    pipeline = _import_pipeline_with_stub()

    raw = tmp_path / "IMG_0001.ARW"
    raw.write_bytes(b"")

    # Missing sidecar XMP should raise
    import pytest

    with pytest.raises(FileNotFoundError):
        pipeline.validate_input_file(raw)

    # Adding XMP satisfies validation
    (tmp_path / "IMG_0001.ARW.xmp").write_text("<xmp/>", encoding="utf-8")
    pipeline.validate_input_file(raw)  # should not raise
