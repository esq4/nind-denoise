import sys
from types import SimpleNamespace
from pathlib import Path


def _import_pipeline_with_stub():
    # Stub exiv2 so importing nind_denoise.pipeline doesn’t require the external binding
    sys.modules.setdefault("exiv2", SimpleNamespace())
    import importlib

    pipeline = importlib.import_module("nind_denoise.pipeline")
    return pipeline


def test_get_output_extension_normalization():
    pipeline = _import_pipeline_with_stub()

    assert pipeline.get_output_extension({}) == ".jpg"
    assert pipeline.get_output_extension({"--extension": "jpg"}) == ".jpg"
    assert pipeline.get_output_extension({"--extension": ".tif"}) == ".tif"


def test_resolve_output_paths_default_dir(tmp_path):
    pipeline = _import_pipeline_with_stub()

    input_path = tmp_path / "IMG_0001.ARW"
    # Existence is not required; function uses only .parent and .name
    out_dir, outpath = pipeline.resolve_output_paths(input_path, None, ".jpg")

    assert out_dir == tmp_path
    assert outpath == (tmp_path / "IMG_0001.jpg")


def test_get_stage_filepaths(tmp_path):
    pipeline = _import_pipeline_with_stub()

    outpath = tmp_path / "final.jpg"
    s1, s1d = pipeline.get_stage_filepaths(outpath, 1)
    assert s1.suffix == ".tif" and s1d.suffix == ".tif"
    assert s1.stem.endswith("_s1")
    assert s1d.stem.endswith("_s1_denoised")

    s2 = pipeline.get_stage_filepaths(outpath, 2)
    assert s2.suffix == ".tif" and s2.stem.endswith("_s2")
