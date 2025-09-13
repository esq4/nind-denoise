import os
import pathlib
import shutil

import pytest
import importlib.machinery
import importlib.util
import typer
from typer.testing import CliRunner


def _load_cli_module():
    path = str(pathlib.Path(__file__).resolve().parents[1] / "src" / "denoise.py")
    loader = importlib.machinery.SourceFileLoader("denoise_cli_local", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("ext", ["jpg", "tiff"])  # real tools; slow/integration
@pytest.mark.integration
def test_raw_pipeline_cli_runs_with_real_tools(tmp_path, ext):
    # Locate a RAW that has a paired sample JPG for reference paths
    raw_dir = pathlib.Path(__file__).parent / "test_raw"
    candidates = []
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        if p.with_suffix(".jpg").exists():
            candidates.append(p)
    if not candidates:
        pytest.skip("No RAW+JPG sample pair found in tests/test_raw")

    raw = candidates[0]

    # Resolve external tools: prefer environment overrides, else PATH
    dt_env = os.environ.get("DT_CLI") or os.environ.get("DARKTABLE_CLI")
    gmic_env = os.environ.get("GMIC")

    def _which(name: str) -> str | None:
        p = shutil.which(name)
        return p if p else None

    dt_path = dt_env or _which("darktable-cli") or _which("darktable-cli.exe")
    if not dt_path:
        pytest.skip("darktable-cli not found; set DT_CLI env var or add to PATH to run integration test")

    gmic_path = gmic_env or _which("gmic") or _which("gmic.exe")

    # Build CLI app
    mod = _load_cli_module()
    app = typer.Typer()
    app.command()(mod.cli)

    outdir = tmp_path
    runner = CliRunner()

    args = [
        str(raw),
        "-o",
        str(outdir),
        "-e",
        ext,
        "--debug",
        "--dt",
        dt_path,
    ]
    if gmic_path:
        args.extend(["--gmic", gmic_path])
    else:
        args.append("--no_deblur")

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout

    out = (outdir / raw.name).with_suffix("." + ext)
    assert out.exists(), f"Output not created: {out}"

    # Ensure the file is readable by PIL/imghdr-like behavior without importing heavy deps
    assert out.stat().st_size > 0
