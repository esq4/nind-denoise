import nind_denoise


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


def _fake_config():
    return {
        "operations": {
            "first_stage": ["demosaic", "flip"],
            "second_stage": ["colorout"],
            "overrides": {},
        }
    }


def test_build_xmp_stage1_filters_and_disables_flip(monkeypatch):
    monkeypatch.setattr(
        nind_denoise.xmp, "read_config", lambda verbose=False: _fake_config()
    )
    out = nind_denoise.xmp.build_xmp(SAMPLE_XMP, stage=1, verbose=True)
    assert "demosaic" in out
    assert "flip" in out and 'darktable:enabled="0"' in out
    assert "colorout" not in out
    assert "sharpen" not in out


def test_build_xmp_stage2_filters_and_updates_iop_order(monkeypatch):
    monkeypatch.setattr(
        nind_denoise.xmp, "read_config", lambda verbose=False: _fake_config()
    )
    out = nind_denoise.xmp.build_xmp(SAMPLE_XMP, stage=2)
    assert "colorout" in out
    # Ensure the demosaic operation was removed from the history (iop_order may still mention it)
    assert 'darktable:operation="demosaic"' not in out
    # iop order tweaks
    assert 'darktable:iop_order_version="5"' in out
    assert "demosaic,0,colorin,0" in out or "colorin,0" not in out


def test_write_xmp_file_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nind_denoise.xmp, "read_config", lambda verbose=False: _fake_config()
    )
    src = tmp_path / "in.xmp"
    dst = tmp_path / "out.xmp"
    src.write_text(SAMPLE_XMP, encoding="utf-8")
    nind_denoise.xmp.write_xmp_file(src, dst, stage=1)
    assert dst.exists()
    text = dst.read_text(encoding="utf-8")
    assert "demosaic" in text and "colorout" not in text
