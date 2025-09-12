import pathlib

import importlib.machinery
import importlib.util


def load_denoise_module():
    # Load the module from the repository src path explicitly
    path = str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'denoise.py')
    loader = importlib.machinery.SourceFileLoader('denoise_local', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_typer_help_displays_usage():
    mod = load_denoise_module()
    import typer
    from typer.testing import CliRunner

    app = typer.Typer()
    app.command()(mod.cli)

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])    
    assert result.exit_code == 0
    assert "Usage" in result.stdout or "Usage:" in result.stdout
