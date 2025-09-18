import types
from pathlib import Path

SAMPLE_XMP = """
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:darktable="http://darktable.sf.net/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" darktable:iop_order_version="4" darktable:iop_order_list="demosaic,0,colorin,0">
      <darktable:history>
        <rdf:Seq>
          <rdf:li darktable:operation="demosaic" darktable:num="0" darktable:enabled="1" />
          <rdf:li darktable:operation="flip" darktable:num="1" darktable:enabled="1" />
          <rdf:li darktable:operation="colorout" darktable:num="2" darktable:enabled="1" />
          <rdf:li darktable:operation="sharpen" darktable:num="3" darktable:enabled="1" />
        </rdf:Seq>
      </darktable:history>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def _tools_stub(tmp_path: Path):
    # Create a minimal Tools-like stub with darktable path
    return types.SimpleNamespace(darktable=tmp_path / "darktable-cli", gmic=None)


def test_export_stage_stage1_builds_cmd(monkeypatch, tmp_path):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import JobContext
    from nind_denoise.config.config import Config

    # Arrange inputs
    input_img = tmp_path / "IMG_0001.ARW"
    input_img.write_bytes(b"")
    src_xmp = tmp_path / "IMG_0001.ARW.xmp"
    src_xmp.write_text(SAMPLE_XMP, encoding="utf-8")
    stage_xmp = tmp_path / "stage1.s1.xmp"
    out_tif = tmp_path / "stage1.tif"

    tools = _tools_stub(tmp_path)

    captured = {}

    def fake_run_cmd(self, args, cwd=None):  # noqa: D401, ANN001
        captured["args"] = [str(a) for a in args]
        captured["cwd"] = str(cwd) if cwd is not None else None
        # Do not actually run anything

    # Avoid verify failing since we don't produce real outputs
    monkeypatch.setattr(ExportStage, "_run_cmd", fake_run_cmd, raising=True)
    monkeypatch.setattr(
        ExportStage,
        "verify_with_env",
        lambda self, cfg, job_ctx: None,
        raising=True,
    )

    stg = ExportStage(tools, input_img, src_xmp, stage_xmp, out_tif, 1)

    # Create fake tool files for validation
    fake_gmic = tmp_path / "fake_gmic.exe"
    fake_dt = tmp_path / "fake_dt.exe"
    fake_gmic.write_bytes(b"")
    fake_dt.write_bytes(b"")

    # Create Config and JobContext for new pattern
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
    cfg = Config(path=config_file, verbose=True)
    job_ctx = JobContext(input_path=input_img, output_path=out_tif)
    stg.execute_with_env(cfg, job_ctx)

    # Assert command args
    args = captured["args"]
    assert args[0].endswith("darktable-cli")
    assert args[1].endswith("IMG_0001.ARW")
    assert args[2].endswith("stage1.s1.xmp")
    assert args[3].endswith("stage1.tif")
    # TIFF bpp should be 32 for stage 1
    assert any("bpp=32" in a for a in args)
    assert captured["cwd"] == str(tmp_path)


def test_export_stage_stage2_builds_cmd(monkeypatch, tmp_path):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import JobContext
    from nind_denoise.config.config import Config

    # Arrange inputs
    input_img = tmp_path / "denoised.tif"
    input_img.write_bytes(b"")
    src_xmp = tmp_path / "IMG_0001.ARW.xmp"
    src_xmp.write_text(SAMPLE_XMP, encoding="utf-8")
    stage_xmp = tmp_path / "stage2.s2.xmp"
    out_tif = tmp_path / "stage2.tif"

    tools = _tools_stub(tmp_path)

    captured = {}

    def fake_run_cmd(self, args, cwd=None):  # noqa: D401, ANN001
        captured["args"] = [str(a) for a in args]
        captured["cwd"] = str(cwd) if cwd is not None else None

    monkeypatch.setattr(ExportStage, "_run_cmd", fake_run_cmd, raising=True)
    monkeypatch.setattr(
        ExportStage,
        "verify_with_env",
        lambda self, cfg, job_ctx: None,
        raising=True,
    )

    stg = ExportStage(tools, input_img, src_xmp, stage_xmp, out_tif, 2)

    # Create fake tool files for validation
    fake_gmic = tmp_path / "fake_gmic.exe"
    fake_dt = tmp_path / "fake_dt.exe"
    fake_gmic.write_bytes(b"")
    fake_dt.write_bytes(b"")

    # Create Config and JobContext for new pattern
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
    cfg = Config(path=config_file, verbose=False)
    job_ctx = JobContext(input_path=input_img, output_path=out_tif)
    stg.execute_with_env(cfg, job_ctx)

    args = captured["args"]
    # TIFF bpp should be 16 for stage 2
    assert any("bpp=16" in a for a in args)
    # ICC args should be present for stage 2
    assert "--icc-intent" in args and "PERCEPTUAL" in args
    assert "--icc-type" in args and "SRGB" in args
