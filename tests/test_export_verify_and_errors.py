from pathlib import Path
import types
import pytest


def _tools_stub(tmp_path: Path):
    return types.SimpleNamespace(darktable=tmp_path / "darktable-cli", gmic=None)


def test_export_verify_renames_tiff_to_requested(tmp_path, monkeypatch):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import Context

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
    stg.verify(Context())
    assert out_tiff.exists()
    assert not alt_tif.exists()


def test_export_missing_xmp_raises(tmp_path):
    from nind_denoise.pipeline.export import ExportStage
    from nind_denoise.pipeline.base import Context, StageError

    tools = _tools_stub(tmp_path)
    input_img = tmp_path / "IMG_0001.ARW"
    input_img.write_bytes(b"")
    src_xmp = tmp_path / "IMG_0001.ARW.xmp"  # do not create
    stage_xmp = tmp_path / "stage1.s1.xmp"
    out_tif = tmp_path / "stage1.tif"

    stg = ExportStage(tools, input_img, src_xmp, stage_xmp, out_tif, 1)

    with pytest.raises(StageError):
        stg.execute(Context())
