from pathlib import Path


def test_resolve_tools_path_fallback(monkeypatch, tmp_path):
    from nind_denoise import config

    # Ensure config loader doesn't override with paths; force PATH fallback
    monkeypatch.setattr(config, "load_cli_config", lambda _: {}, raising=True)

    fake_dt = tmp_path / "darktable-cli"
    fake_dt.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_gmic = tmp_path / "gmic"
    fake_gmic.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_which(name):  # emulate PATH search
        if name.startswith("darktable-cli"):
            return str(fake_dt)
        if name.startswith("gmic"):
            return str(fake_gmic)
        return None

    monkeypatch.setattr(config.shutil, "which", fake_which, raising=True)

    tools = config.resolve_tools(None, None)
    assert tools.darktable == fake_dt
    assert tools.gmic == fake_gmic


def test_resolve_tools_explicit_overrides(monkeypatch, tmp_path):
    from nind_denoise import config

    # Avoid reading actual packaged config
    monkeypatch.setattr(config, "load_cli_config", lambda _: {}, raising=True)

    explicit_dt = tmp_path / "dt.exe"
    explicit_dt.write_text("", encoding="utf-8")

    # gmic not provided and not on PATH
    monkeypatch.setattr(config.shutil, "which", lambda name: None, raising=True)

    tools = config.resolve_tools(explicit_dt, None)
    assert tools.darktable == explicit_dt
    assert tools.gmic is None
