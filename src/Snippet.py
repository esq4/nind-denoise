#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Huy Hoang

Denoise the raw image denoted by <filename> and save the results.
Usage:
    denoise.py [-o <outpath> | --output-path=<outpath>] [-e <e> | --extension=<e>]
                    [-d <darktable> | --dt=<darktable>] [-g <gmic> | --gmic=<gmic>] [ -q <q> | --quality=<q>]
                    [--nightmode ] [ --no_deblur ] [ --debug ] [ --sigma=<sigma> ] [ --iterations=<iter> ]
                    [-v | --verbose] <raw_image>
    denoise.py (help | -h | --help)
    denoise.py --version

Options:
  -o <outpath> --output-path=<outpath>  Where to save the result (defaults to current directory).
  -e <e> --extension=<e>                Output file extension. Supported formats are ....? [default: jpg].
  --dt=<darktable>                      Path to darktable-cli. Use this only if not automatically found.
  -g <gmic> --gmic=<gmic>               Path to gmic. Use this only if not automatically found.
  -q <q> --quality=<q>                  JPEG compression quality. Lower produces a smaller file at the cost of more artifacts. [default: 90].
  --nightmode                           Use for very dark images. Normalizes brightness (exposure, tonequal) before denoise [default: False].
  --no_deblur                           Do not perform RL-deblur [default: false].
  --debug                               Keep intermedia files.
  --sigma=<sigma>                       sigma to use for RL-deblur. Acceptable values are ....? [default: 1].
  --iterations=<iter>                   Number of iterations to perform during RL-deblur. Suggest keeping this to ...? [default: 10].
  -v --verbose
  --version                             Show version.
  -h --help                             Show this screen.
"""
import copy
import io
import pathlib

import exiv2
import yaml
from bs4 import BeautifulSoup

# Define RAW extensions as a constant
VALID_EXTENSIONS = [
    '.' + item.lower()
    if item[0] != '.'
    else item.lower()
    for item in
    ['3FR', 'ARW', 'SR2', 'SRF', 'CR2', 'CR3', 'CRW', 'DNG', 'ERF', 'FFF', 'MRW', 'NEF', 'NRW', 'ORF', 'PEF', 'RAF',
     'RW2']
]


class ImageProcessor:
    def __init__(self, config_path=pathlib.Path('./src/config/operations.yaml'), verbose=False):
        self.config = self.read_config(config_path)
        self.verbose = verbose

    @staticmethod
    def check_good_input(path: pathlib.Path, extensions=None) -> bool:
        """Checks that the given path is a good, raw image file.

        :param path: Path to the input file.
        :type path: pathlib.Path
        :param extensions: List of valid extensions. If not provided, uses VALID_EXTENSIONS.
        :type extensions: list or None
        :return: True if the path is a good raw image file, False otherwise.
        :rtype: bool
        """
        extensions = [extensions] if isinstance(extensions, str) else extensions or VALID_EXTENSIONS
        assert isinstance(extensions, list)

        if not path.is_file():
            print("This isn't a file:", path)
            if not path.exists():
                print("In fact, it doesn't exist.")
            print("Either way, I'm skipping it.\n")
            return False

        elif path.suffix.lower() not in extensions:
            if path.suffix.lower() != '.xmp':
                print(f"Not a (supported) RAW file: {path}, skipping.")
            return False
        else:
            return True

    @staticmethod
    def clone_exif(src_file: pathlib.Path, dst_file: pathlib.Path, verbose=False) -> None:
        """Convenience function to clone the exif metadata from one image file to another.

        :param src_file: Source file with EXIF data.
        :type src_file: pathlib.Path
        :param dst_file: Destination file where EXIF will be copied to.
        :type dst_file: pathlib.Path
        """
        src_image = exiv2.ImageFactory.open(str(src_file))
        src_image.readMetadata()
        dst_image = exiv2.ImageFactory.open(str(dst_file))
        dst_image.setExifData(src_image.exifData())
        dst_image.writeMetadata()

        if verbose:
            print(f'Copied EXIF from {src_file} to {dst_file}')

    def read_config(self, config_path=pathlib.Path('./src/config/operations.yaml'), _nightmode=False) -> dict:
        with io.open(config_path, 'r', encoding='utf-8') as instream:
            var = yaml.safe_load(instream)

        if _nightmode and self.verbose:
            print("\nUpdating ops for nightmode ...")
            nightmode_ops = ['exposure', 'toneequal']
            var["operations"]["first_stage"].extend(nightmode_ops)
            for op in nightmode_ops:
                var["operations"]["second_stage"].remove(op)

        return var

    @staticmethod
    def parse_darktable_history_stack(_input_xmp: pathlib.Path, config: dict, verbose=False):
        """Parses the Darktable history stack and modifies it based on operations defined in the config.

        :param _input_xmp: Path to input XMP file.
        :type _input_xmp: pathlib.Path
        :param config: Configuration dictionary with operation stages.
        :type config: dict
        """
        operations = config["operations"]
        with _input_xmp.open() as f:
            sidecar_xml = f.read()

        sidecar = BeautifulSoup(sidecar_xml, "xml")
        history = sidecar.find('darktable:history')
        history_org = copy.copy(history)
        history_ops = history.find_all('rdf:li')

        # Sort and process first stage operations
        history_ops.sort(key=lambda tag: int(tag['darktable:num']))
        for op in reversed(history_ops):
            if op['darktable:operation'] not in operations['first_stage']:
                op.extract()
                if verbose:
                    print("--removed:", op['darktable:operation'])
            elif op['darktable:operation'] == 'flip':
                op['darktable:enabled'] = "0"
                if verbose:
                    print("default:", op['darktable:operation'])

        # Write first stage XMP file
        _input_xmp.with_suffix('.s1.xmp').touch(exist_ok=False)
        with _input_xmp.with_suffix('.s1.xmp').open('w') as first_stage_xmp_file:
            first_stage_xmp_file.write(sidecar.prettify())

        # Restore the history stack to original and process second stage operations
        history.replace_with(history_org)
        history_ops = history_org.find_all('rdf:li')

        for op in reversed(history_ops):
            if (op['darktable:operation'] not in operations["second_stage"] and
                    op['darktable:operation'] in operations["first_stage"]):
                op.extract()
                if verbose:
                    print("--removed:", op['darktable:operation'])
            elif op['darktable:operation'] in operations["overrides"]:
                for key, val in operations["overrides"][op['darktable:operation']].items():
                    op[key] = val
            if verbose:
                print("default:", op['darktable:operation'], op['darktable:enabled'])

        # Set iop_order_version to 5 (for JPEG)
        description = sidecar.find('rdf:Description')
        description['darktable:iop_order_version'] = '5'

        # Bring colorin right next to demosaic
        if description.has_attr("darktable:iop_order_list"):
            order_list = description['darktable:iop_order_list'].replace('colorin,0,', '').replace('demosaic,0',
                                                                                                   'demosaic,0,colorin,0')
            description['darktable:iop_order_list'] = order_list

# ... existing code ...
