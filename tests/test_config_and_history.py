import io
import yaml
import pathlib

from bs4 import BeautifulSoup

import importlib.machinery
import importlib.util
import pathlib

# load functions from src/nind_denoise/pipeline.py explicitly
_path = str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'nind_denoise' / 'pipeline.py')
_loader = importlib.machinery.SourceFileLoader('pipeline_local_ch', _path)
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
_mod = importlib.util.module_from_spec(_spec)
# register before exec to satisfy dataclasses
import sys as _sys
_sys.modules[_loader.name] = _mod
_loader.exec_module(_mod)
read_config = _mod.read_config
parse_darktable_history_stack = _mod.parse_darktable_history_stack


def test_read_config_nightmode_moves_ops(tmp_path):
    # Prepare a minimal operations.yaml
    ops = {
        'operations': {
            'first_stage': ['demosaic', 'colorin', 'flip'],
            'second_stage': ['exposure', 'toneequal', 'colorin'],
            'overrides': {
                'colorin': {'darktable:enabled': '1'}
            }
        },
        'models': {
            'nind_generator_650.pt': {
                'path': str(tmp_path / 'models' / 'generator_650.pt')
            }
        }
    }
    cfg_path = tmp_path / 'operations.yaml'
    cfg_path.write_text(yaml.safe_dump(ops), encoding='utf-8')

    base = read_config(config_path=str(cfg_path), _nightmode=False)
    assert base['operations']['first_stage'] == ['demosaic', 'colorin', 'flip']
    assert 'exposure' in base['operations']['second_stage']

    nm = read_config(config_path=str(cfg_path), _nightmode=True)
    assert 'exposure' in nm['operations']['first_stage']
    assert 'toneequal' in nm['operations']['first_stage']
    assert 'exposure' not in nm['operations']['second_stage']
    assert 'toneequal' not in nm['operations']['second_stage']


def test_parse_darktable_history_stack_generates_s1_s2(tmp_path):
    # create a minimal input XMP sidecar
    sample_xmp = tmp_path / 'in.xmp'
    sample_xmp.write_text('''\
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:darktable="http://darktable.sourceforge.net/">
  <rdf:Description darktable:iop_order_version="3" darktable:iop_order_list="demosaic,0,unlistedop,0,colorin,0,exposure,0,toneequal,0,flip,0">
    <darktable:history>
      <rdf:Seq>
        <rdf:li darktable:num="0" darktable:operation="demosaic" darktable:enabled="1" />
        <rdf:li darktable:num="1" darktable:operation="unlistedop" darktable:enabled="1" />
        <rdf:li darktable:num="2" darktable:operation="colorin" darktable:enabled="0" />
        <rdf:li darktable:num="3" darktable:operation="exposure" darktable:enabled="1" />
        <rdf:li darktable:num="4" darktable:operation="toneequal" darktable:enabled="1" />
        <rdf:li darktable:num="5" darktable:operation="flip" darktable:enabled="1" />
      </rdf:Seq>
    </darktable:history>
  </rdf:Description>
</rdf:RDF>
''', encoding='utf-8')

    # config mirrors operations of interest
    config = {
        'operations': {
            'first_stage': ['demosaic', 'colorin', 'exposure', 'toneequal', 'flip'],
            'second_stage': ['colorin', 'toneequal'],
            'overrides': {
                'colorin': {'darktable:enabled': '1'}
            }
        }
    }
    parse_darktable_history_stack(sample_xmp, config=config, verbose=True)
    s1 = sample_xmp.with_suffix('.s1.xmp')
    s2 = sample_xmp.with_suffix('.s2.xmp')
    assert s1.exists() and s2.exists()
    s1_xml = s1.read_text(encoding='utf-8')
    s2_xml = s2.read_text(encoding='utf-8')

    soup1 = BeautifulSoup(s1_xml, 'xml')
    ops1 = [li['darktable:operation'] for li in soup1.find_all('rdf:li')]
    # s1 keeps only first_stage, with flip disabled
    assert 'unlistedop' not in ops1
    flip = [li for li in soup1.find_all('rdf:li') if li['darktable:operation']=='flip'][0]
    assert flip['darktable:enabled'] == '0'

    soup2 = BeautifulSoup(s2_xml, 'xml')
    desc = soup2.find('rdf:Description')
    # iop_order_version set to 5
    assert desc['darktable:iop_order_version'] == '5'
    # colorin moved next to demosaic in list
    assert 'demosaic,0,colorin,0' in desc['darktable:iop_order_list']

    ops2 = [li['darktable:operation'] for li in soup2.find_all('rdf:li')]
    # second stage removes ops not in second_stage if they were in first_stage
    assert 'exposure' not in ops2
    assert 'toneequal' in ops2
