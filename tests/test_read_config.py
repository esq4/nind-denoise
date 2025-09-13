import types


def test_read_config_nightmode_toggles(monkeypatch):
    from nind_denoise import config

    base_cfg = {
        "operations": {
            "first_stage": ["demosaic", "flip"],
            "second_stage": ["colorout", "toneequal", "exposure"],
        }
    }

    # Provide controlled config without touching files
    monkeypatch.setattr(config, "load_cli_config", lambda path=None: base_cfg)

    # Without nightmode: unchanged
    out = config.read_config(config_path=None, _nightmode=False, verbose=True)
    assert out["operations"]["first_stage"] == ["demosaic", "flip"]
    assert out["operations"]["second_stage"] == [
        "colorout",
        "toneequal",
        "exposure",
    ]

    # With nightmode: exposure and toneequal moved to first_stage, removed from second_stage
    out_nm = config.read_config(config_path=None, _nightmode=True, verbose=True)
    assert "exposure" in out_nm["operations"]["first_stage"]
    assert "toneequal" in out_nm["operations"]["first_stage"]
    assert "exposure" not in out_nm["operations"]["second_stage"]
    assert "toneequal" not in out_nm["operations"]["second_stage"]
