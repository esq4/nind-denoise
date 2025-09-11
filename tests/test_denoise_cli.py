import pathlib
import sys

import pytest
from docopt import docopt

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


def test_docopt_parsing_help_displays_usage(capsys):
    mod = load_denoise_module()
    with pytest.raises(SystemExit):
        # docopt raises on --help by design
        docopt(mod.__doc__, argv=['--help'], version='__version__')


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


def test_get_command_paths_defaults_are_strings():
    mod = load_denoise_module()
    # empty args for defaults
    dt, gmic = mod.get_command_paths({'--dt': None, '--gmic': None})
    assert isinstance(dt, str)
    assert isinstance(gmic, str)


def test_check_good_input_validations(tmp_path):
    mod = load_denoise_module()
    raw = tmp_path / 'a.RAF'
    xmp = tmp_path / 'a.RAF.xmp'
    raw.write_text('x')
    xmp.write_text('x')
    assert mod.check_good_input(raw, mod.valid_extensions)
    assert not mod.check_good_input(tmp_path / 'b.txt', mod.valid_extensions)
