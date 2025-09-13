from pathlib import Path

import types

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
    from nind_denoise.pipeline.base import Context

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
    monkeypatch.setattr(ExportStage, "verify", lambda self, ctx=None: None, raising=True)

    stg = ExportStage(tools, input_img, src_xmp, stage_xmp, out_tif, 1)
    stg.execute(Context(verbose=True))

    # Assert command args
    args = captured["args"]
    assert args[0].endswith("darktable-cli")  # uses tools.darktable
    assert args[1].endswith("IMG_0001.ARW")
    assert args[2].endswith("stage1.s1.xmp")
    assert args[3].endswith("stage1.tif")
    # TIFF bpp should be 32 for stage 1
    assert any("bpp=32" in a for a in args)
    assert captured["cwd"] == str(tmp_path)


def test_export_stage_stage2_builds_cmd(monkeypatch, tmp_path):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import Context

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
    monkeypatch.setattr(ExportStage, "verify", lambda self, ctx=None: None, raising=True)

    stg = ExportStage(tools, input_img, src_xmp, stage_xmp, out_tif, 2)
    stg.execute(Context(verbose=False))

    args = captured["args"]
    # TIFF bpp should be 16 for stage 2
    assert any("bpp=16" in a for a in args)
    # ICC args should be present for stage 2
    assert "--icc-intent" in args and "PERCEPTUAL" in args
    assert "--icc-type" in args and "SRGB" in args
