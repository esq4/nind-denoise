import types
from pathlib import Path

import pytest


def _tools_stub(tmp_path: Path):
    return types.SimpleNamespace(darktable=tmp_path / "darktable-cli", gmic=None)


def test_export_verify_renames_tiff_to_requested(tmp_path, monkeypatch):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import JobContext
    from nind_denoise.config.config import Config

    tools = _tools_stub(tmp_path)
    out_tiff = tmp_path / "stage1.tiff"
    alt_tif = tmp_path / "stage1.tif"
    alt_tif.write_bytes(b"")  # simulate darktable produced .tif instead of .tiff

    stg = ExportStage(
        tools,
        input_tif=tmp_path / "in.ARW",
        src_xmp=tmp_path / "in.ARW.xmp",
        stage_xmp=tmp_path / "stage1.s1.xmp",
        out_tif=out_tiff,
        stage_number=1,
    )

    # Directly call verify to test rename behavior without running any command
<<<<<<< HEAD
    # Create fake tool files for validation
    fake_gmic = tmp_path / "fake_gmic.exe"
    fake_dt = tmp_path / "fake_dt.exe"
    fake_gmic.write_bytes(b"")
    fake_dt.write_bytes(b"")

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        f"""
models:
  test_model:
    path: /fake/model/path
    default: true
tools:
  windows:
    gmic:
      path: {fake_gmic}
      args: []
    darktable:
      path: {fake_dt}
      args: []
  posix:
    gmic:
      path: {fake_gmic}
      args: []
    darktable:
      path: {fake_dt}
      args: []
operations:
  operations:
    first_stage: ["demosaic", "flip"]
    second_stage: ["colorout", "sharpen"]
nightmode_ops: ["sharpen"]
"""
    )
    cfg = Config(path=config_file)
    job_ctx = JobContext(input_path=tmp_path / "in.ARW", output_path=out_tiff)
=======
    cfg = Config(tools=tools, config={})
    job_ctx = JobContext(
        input_path=tmp_path / "in.ARW", output_path=out_tiff, output_dir=tmp_path
    )
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
    stg.verify_with_env(cfg, job_ctx)
    assert out_tiff.exists()
    assert not alt_tif.exists()


def test_export_missing_xmp_raises(tmp_path):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import JobContext, StageError
    from nind_denoise.config.config import Config

    tools = _tools_stub(tmp_path)
    input_img = tmp_path / "IMG_0001.ARW"
    input_img.write_bytes(b"")
    src_xmp = tmp_path / "IMG_0001.ARW.xmp"  # do not create
    stage_xmp = tmp_path / "stage1.s1.xmp"
    out_tif = tmp_path / "stage1.tif"

    stg = ExportStage(tools, input_img, src_xmp, stage_xmp, out_tif, 1)

<<<<<<< HEAD
    # Create fake tool files for validation
    fake_gmic = tmp_path / "fake_gmic.exe"
    fake_dt = tmp_path / "fake_dt.exe"
    fake_gmic.write_bytes(b"")
    fake_dt.write_bytes(b"")

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        f"""
models:
  test_model:
    path: /fake/model/path
    default: true
tools:
  windows:
    gmic:
      path: {fake_gmic}
      args: []
    darktable:
      path: {fake_dt}
      args: []
  posix:
    gmic:
      path: {fake_gmic}
      args: []
    darktable:
      path: {fake_dt}
      args: []
operations:
  operations:
    first_stage: ["demosaic", "flip"]
    second_stage: ["colorout", "sharpen"]
nightmode_ops: ["sharpen"]
"""
    )
    cfg = Config(path=config_file)
    job_ctx = JobContext(input_path=input_img, output_path=out_tif)
=======
    cfg = Config(tools=tools, config={})
    job_ctx = JobContext(input_path=input_img, output_path=out_tif, output_dir=tmp_path)
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882

    with pytest.raises(StageError):
        stg.execute_with_env(cfg, job_ctx)
