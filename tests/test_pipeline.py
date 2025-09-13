import pathlib

import pytest
import typer
from typer.testing import CliRunner

from nind_denoise.config import valid_extensions


@pytest.mark.parametrize("ext", ["jpg", "tiff"])  # real tools; slow/integration
@pytest.mark.integration
def test_raw_pipeline_cli_runs_with_real_tools(tmp_path, ext):
    # Locate a RAW sample using supported extensions from config
    raw_dir = pathlib.Path(__file__).parent / "test_raw"
    candidates = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    assert candidates, "No RAW sample with a supported extension found in tests/test_raw"

    raw = candidates[0]

    # Build CLI app directly from module import (no dynamic loader)
    import denoise as mod

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
    ]

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout

    out = (outdir / raw.name).with_suffix("." + ext)
    assert out.exists(), f"Output not created: {out}"

    # Ensure the file is readable by PIL/imghdr-like behavior without importing heavy deps
    assert out.stat().st_size > 0
