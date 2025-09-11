import os
import pathlib
import types

import pytest

import importlib.machinery
import importlib.util


def load_denoise_module():
    path = str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'denoise.py')
    loader = importlib.machinery.SourceFileLoader('denoise_local', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_get_stage_filepaths(tmp_path):
    mod = load_denoise_module()
    out = tmp_path / 'a.jpg'
    s1, s1d = mod.get_stage_filepaths(out, 1)
    assert s1.name.endswith('_s1.tif') and s1d.name.endswith('_s1_denoised.tif')
    s2 = mod.get_stage_filepaths(out, 2)
    assert s2.name.endswith('_s2.tif')


def test_denoise_file_orchestration(tmp_path, sample_xmp, fake_exiv2_module, monkeypatch):
    mod = load_denoise_module()

    # monkeypatch exiv2 clone
    mod.exiv2 = fake_exiv2_module

    # fake commands
    dt = tmp_path / 'darktable-cli.exe'
    gmic = tmp_path / 'gmic.exe'  # not used as we set --no_deblur
    dt.write_text('x')
    gmic.write_text('x')

    monkeypatch.setattr(mod, 'get_command_paths', lambda args: (str(dt), str(gmic)))

    # fake config
    cfg = {
        'models': {
            'nind_generator_650.pt': {'path': str(tmp_path / 'models' / 'generator_650.pt') }
        },
        'operations': {
            'first_stage': ['demosaic', 'colorin', 'flip'],
            'second_stage': ['colorin'],
            'overrides': {}
        }
    }
    monkeypatch.setattr(mod, 'read_config', lambda config_path='./src/config/operations.yaml', _nightmode=False, verbose=False: cfg)

    # ensure model file exists to avoid download
    model_path = pathlib.Path(cfg['models']['nind_generator_650.pt']['path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text('weights')

    # subprocess.run should create the expected files
    def fake_run(cmd, cwd=None, check=None):
        # Identify output files by command type length
        if isinstance(cmd[0], str) and os.path.basename(cmd[0]).startswith('darktable-cli'):
            # last arg is output filename when using darktable
            outname = cmd[3]
            outpath = pathlib.Path(cwd) / outname
            outpath.write_text('tiff')
        elif isinstance(cmd[0], str) and cmd[0].endswith('python.exe') or cmd[0].endswith('python'):
            # denoiser call: last two args are --output, path
            outpath = pathlib.Path(cmd[-1])
            outpath.write_text('tiff')
        else:
            # fallback: create specified output if present
            pass
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, 'run', fake_run)

    # input RAW and XMP
    raw = sample_xmp.with_suffix('')  # removes .xmp suffix, yields .RAF
    raw.write_text('raw')

    args = {
        '--output-path': str(tmp_path),
        '--extension': 'jpg',
        '--dt': None,
        '--gmic': None,
        '--sigma': '1',
        '--quality': '90',
        '--iterations': '10',
        '--verbose': False,
        '--no_deblur': True,  # skip gmic step
        '--debug': False,
    }

    mod.denoise_file(args, raw)

    # expect final output exists
    out = tmp_path / raw.name
    out = out.with_suffix('.jpg')
    assert out.exists()
