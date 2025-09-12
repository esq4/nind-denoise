import pathlib
import sys

import pytest

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


def test_get_output_extension_adds_dot_when_missing():
    mod = load_denoise_module()
    assert mod.get_output_extension({'--extension': 'jpg'}) == '.jpg'
    assert mod.get_output_extension({'--extension': '.png'}) == '.png'


def test_get_output_path_uses_arg_when_provided(tmp_path):
    mod = load_denoise_module()
    input_path = tmp_path / 'foo.RAF'
    input_path.write_text('x')
    outdir = tmp_path / 'out'
    outdir.mkdir()
    res = mod.get_output_path({'--output-path': str(outdir)}, input_path)
    assert res == outdir


def test_get_output_path_defaults_to_input_parent(tmp_path):
    mod = load_denoise_module()
    input_path = tmp_path / 'bar.RAF'
    input_path.write_text('x')
    res = mod.get_output_path({'--output-path': None}, input_path)
    assert res == tmp_path


def test_get_command_paths_defaults_are_paths():
    mod = load_denoise_module()
    # empty args for defaults
    dt, gmic = mod.get_command_paths({'--dt': None, '--gmic': None})
    import pathlib as _pl
    assert isinstance(dt, _pl.Path)
    assert isinstance(gmic, _pl.Path)


def test_check_good_input_validations(tmp_path):
    mod = load_denoise_module()
    raw = tmp_path / 'a.RAF'
    xmp = tmp_path / 'a.RAF.xmp'
    raw.write_text('x')
    xmp.write_text('x')
    assert mod.check_good_input(raw, mod.valid_extensions)
    assert not mod.check_good_input(tmp_path / 'b.txt', mod.valid_extensions)
