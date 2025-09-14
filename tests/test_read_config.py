import types

def test_read_config_nightmode_moves_ops(monkeypatch):
    from nind_denoise import config as cfg

    # Provide a minimal base config via packaged loader stub
    base = {
        "operations": {
            "first_stage": ["demosaic", "flip"],
            "second_stage": ["colorout", "toneequal", "exposure"],
        }
    }
    monkeypatch.setattr(cfg, "load_cli_config", lambda _: base, raising=True)

    # Baseline: without nightmode, config should stay as-is
    normal = cfg.read_config(None, _nightmode=False, verbose=True)
    assert normal["operations"]["first_stage"] == ["demosaic", "flip"]
    assert normal["operations"]["second_stage"] == [
        "colorout",
        "toneequal",
        "exposure",
    ]

    # Nightmode: exposure and toneequal should move to first_stage and be removed from second
    night = cfg.read_config(None, _nightmode=True, verbose=True)
    assert "exposure" in night["operations"]["first_stage"]
    assert "toneequal" in night["operations"]["first_stage"]
    # second_stage should have those removed but keep the rest
    assert "exposure" not in night["operations"]["second_stage"]
    assert "toneequal" not in night["operations"]["second_stage"]
    assert "colorout" in night["operations"]["second_stage"]
