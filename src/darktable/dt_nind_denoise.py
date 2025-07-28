#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Huy Hoang

Split the darktable export into two stages to inject nind-denoise into the history stack

"""

import os, sys, subprocess, argparse, shutil
from argparse import ArgumentParser
import configparser
from bs4 import BeautifulSoup
from pathlib import Path
import copy
import exiv2

# @TODO: test and add all available modules, default to keep unlisted ops in 2nd-stage

# list of operations to stay in first stage
first_ops = [
  'channelmixerrgb',
  'colorin',
  'colorout',
  'demosaic',
  #'exposure',
  'flip',
  'gamma',
  'highlights',
  'hotpixels',
  'mask_manager',
  'rawprepare',
  'temperature',
  #'toneequal',
]

# list of operations to be moved to second stage
second_ops = [
  'ashift',         # rotate & perspective, NOTE: autocrop doesn't auto-apply via non-interactive mode
  'bilat',          # local contrast
  'blurs',
  'borders',        # framing
  'colorbalancergb',
  'colorin',
  'colorout',
  'crop',
  'cacorrectrgb',   # chromatic aberrations
  # 'channelmixerrgb',
  'clahe',          # local contrast
  'denoiseprofile',
  'diffuse',        # diffuse or sharpen
  'dither',         # dithering
  'exposure',
  'filmicrgb',
  'flip',
  'gamma',
  'hazeremoval',
  'invert',
  'lens',           # lens correction
  'levels',
  'liquify',
  'lowlight',       # lowlight vision
  'lut3d',
  'mask_manager',
  'monochrome',
  'nlmeans',        # astro photo denoise
  # 'rawdenoise',
  # 'rawprepare',
  'rgbcurve',
  'rgblevels',
  'rotatepixels',
  'scalepixels',
  'shadhi',         # shadow and highlight
  'sharpen',
  'soften',
  'splittoning',
  'spots',          # spot removal
  # 'temperature',
  'tonecurve',
  'tonemap',        # tone-mapping
  'toneequal',
  'velvia',
  'vibrance',
  'vignette',
  'watermark',
  'zonesystem',
]

second_overrides = {
  'colorin': {
    'darktable:num':"0",
    'darktable:operation':"colorin",
    'darktable:enabled':"1",
    'darktable:modversion':"7",
    'darktable:params':"gz48eJxjZBgFowABWAbaAaNgwAEAEDgABg==",
    'darktable:multi_name':"",
    'darktable:multi_name_hand_edited':"0",
    'darktable:multi_priority':"0",
    'darktable:blendop_version':"14",
    'darktable:blendop_params':"gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="
  }
}


"""
  main program, meant to be called manually or by darktable's lua script

"""

def main(argv):
    parser = ArgumentParser()
    parser.add_argument("filenames", metavar="FILE", nargs='*',
                        help="source image", )
    parser.add_argument("-n", "--nightmode", dest="nightmode", type=bool, action=argparse.BooleanOptionalAction,
                        help="nightmode, normalize brightness (exposure, tonequal) before denoise, default: no")
    parser.add_argument("-r", "--rating", dest="rating", default='012345',
                        help="darktable rating, specified as [012345], default: 012345 (all)")
    parser.add_argument("-d", "--debug", dest="debug", type=bool, action=argparse.BooleanOptionalAction,
                        help="debug mode to print extra info and keep intermedia files, default: no")
    parser.add_argument("-l", "--rldeblur", dest="rldeblur", type=bool, action=argparse.BooleanOptionalAction,
                        default=True, help="whether to enable RL-deblur, default: yes")
    parser.add_argument("-e", "--extension", dest="extension", default='jpg',
                        help="output file extension, default: jpg")
    parser.add_argument("-q", "--quality", dest="quality", type=int, default=90,
                        help="JPEG compression quality, default: 90")
    parser.add_argument("-s", "--sigma", dest="sigma", type=float, default=1,
                        help="RL-deblur sigma, default: 1")
    parser.add_argument("-i", "--iter", dest="iteration", type=int, default=10,
                        help="RL-deblur number of iteration, default: 10")

    args = parser.parse_args()

    # create output folder if needed
    outdir = 'darktable_exported'
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    # read config
    config_filename = os.path.join(os.path.dirname(__file__), 'dt_nind_denoise.ini')

    if not os.path.exists(config_filename):
      print('Error reading ', config_filename)
      exit(1)

    config = configparser.ConfigParser()
    config.read(config_filename)

    cmd_darktable     = config['command']['darktable'].strip()
    cmd_nind_denoise  = config['command']['nind_denoise'] + ' ' + config['command']['nind_denoise_params']

    # verify darktable-cli is valid
    if not Path(cmd_darktable).exists():
      print("\nError: darktable-cli (" + cmd_darktable + ") does not exist or not accessible. Please correct the path in dt_nind_denoise.ini")
      # exit(1)

    # gmic is optional
    cmd_gmic = None
    if 'gmic' in config['command'] and config['command']['gmic'].strip() != '':
      cmd_gmic = config['command']['gmic'].strip()

      # verify gmic is valid
      if not os.path.exists(cmd_gmic):
        print("\nError: gmic (" + cmd_gmic+ ") does not exist, disabled RL-deblur")
        args.rldeblur = False

    else:
      print("\nGMIC is not provided, disabled RL-deblur")
      args.rldeblur = False

    # define RAW extensions
    valid_extensions = ['3FR','ARW','SR2','SRF','CR2','CR3','CRW','DNG','ERF','FFF','MRW','NEF','NRW','ORF','PEF','RAF','RW2']

    # update the first and second ops for night mode
    if args.nightmode:
      print("\nUpdating ops for nightmode ...")
      nightmode_ops = ['exposure', 'toneequal']
      first_ops.extend(nightmode_ops)

      for op in nightmode_ops:
        second_ops.remove(op)

    # main loop: iterate through all provided images
    for filename in args.filenames:
      # skip invalid/non-existing file
      if not os.path.exists(filename):
        print("Non-existing file: ", filename, ", skipping.")
        continue

      print("\n")

      # determine a new filename
      basename, ext = os.path.splitext(filename)

      # skip un-supported files
      if ext.lstrip('.').upper() not in valid_extensions:
        print("Non-RAW file: ", filename, ", skipping.")
        continue

      if args.extension != '':
          ext = '.' + args.extension.lstrip('.')

      i = 0
      out_filename = basename + ext
      while (os.path.exists(os.path.join(outdir, out_filename))):
          i = i + 1
          out_filename = basename + '_' + str(i) + ext

      # read the XMP
      xmp = filename + '.xmp'

      if not os.path.exists(xmp):
        print("Error: cannot find sidecar file ", xmp)
        continue

      with open(xmp, 'r') as f:
          sidecar_xml = f.read()

      sidecar = BeautifulSoup(sidecar_xml, "xml")

      # check rating
      rating = sidecar.find('rdf:Description')['xmp:Rating']

      if args.debug:
        print('Rating: ', rating)

      rating_filter = list(args.rating)
      if rating not in rating_filter:
        print('Rating of', rating, 'does not match rating filter. Skipping.')
        continue

      # read the history stack
      history = sidecar.find('darktable:history')
      history_org = copy.copy(history)
      history_ops = history.find_all('rdf:li')

      # sort history ops
      history_ops.sort(key=lambda tag: int(tag['darktable:num']))

      # remove ops not listed in first_ops
      if args.debug:
        print("\nPrepping first stage ...")

      for op in reversed(history_ops):
        if op['darktable:operation'] not in first_ops:
          # op['darktable:enabled'] = "0"
          op.extract()    # remove the op completely


          if args.debug:
            print("--removed: ", op['darktable:operation'])

        else:
          # for "flip": don't remove, only disable
          if op['darktable:operation'] == 'flip':
            op['darktable:enabled'] = "0"

          if args.debug:
            print("default:    ", op['darktable:operation'])

      with open(filename+'.s1.xmp', 'w') as first_stage:
        first_stage.write(sidecar.prettify())


      # restore the history stack to original
      history.replace_with(history_org)
      history_ops = history_org.find_all('rdf:li')

      # remove ops not listed in second_ops
      # unknown ops NOT in first_ops AND NOT in second_ops, default to keeping them
      # in 1    : N   N   Y   Y
      # in 2    : N   Y   N   Y
      # action  : K   K   R   K

      if args.debug:
        print("\nPrepping second stage ...")

      for op in reversed(history_ops):
        if op['darktable:operation'] not in second_ops and op['darktable:operation'] in first_ops:
          op.extract()    # remove the op completely

          if args.debug:
            print("--removed: ", op['darktable:operation'])
        else:
          # replace with override
          if op['darktable:operation'] in second_overrides:
            for key, val in second_overrides[op['darktable:operation']].items():
              op[key] = val

          if args.debug:
            print("default:    ", op['darktable:operation'], op['darktable:enabled'])


      # set iop_order_version to 5 (for JPEG)
      description = sidecar.find('rdf:Description')
      description['darktable:iop_order_version'] = '5'

      # bring colorin right next to demosaic (early in the stack)
      if description.has_attr("darktable:iop_order_list"):
        description['darktable:iop_order_list'] = description['darktable:iop_order_list'].replace('colorin,0,', '').replace('demosaic,0', 'demosaic,0,colorin,0')

      with open(filename+'.s2.xmp', 'w') as second_stage:
          second_stage.write(sidecar.prettify())


      #========== invoke darktable-cli with first stage ==========

      # https://github.com/darktable-org/darktable/issues/12958
      # leave the intermediate tiff files in the current folder
      s1_filename = basename + '_s1.tif'

      if os.path.exists(s1_filename):
        os.remove(s1_filename)

      cmd = cmd_darktable + ' "' + filename + '" "' + filename + '.s1.xmp" "' + s1_filename + '" ' + \
            '--apply-custom-presets 0 ' + \
            '--core --conf plugins/imageio/format/tiff/bpp=32 '

      if args.debug:
        print('First-stage cmd: ', cmd)

      subprocess.call(cmd, shell=True)

      if not os.path.exists(s1_filename):
        print("Error: first-stage export not found: ", s1_filename)
        continue


      #========== call nind-denoise ==========
      # 32-bit TIFF (instead of 16-bit) is needed to retain highlight reconstruction data from stage 1
      # for modified nind-denoise: tif = 16-bit, tiff = 32-bit
      denoised_filename = basename + '_s1_denoised.tiff'

      if os.path.exists(denoised_filename):
        os.remove(denoised_filename)

      cmd = cmd_nind_denoise + ' --input "' + s1_filename + '" --output "' + denoised_filename + '"'

      if args.debug:
        print('nind-denoise cmd: ', cmd)

      subprocess.call(cmd, shell=True)

      if not os.path.exists(denoised_filename):
        print("Error: denoised image not found: ", denoised_filename)
        continue


      # copy exif from RAW file to denoised image
      exiv_src = exiv2.ImageFactory.open(filename)
      exiv_src.readMetadata()
      exiv_dst = exiv2.ImageFactory.open(denoised_filename)
      exiv_dst.setExifData(exiv_src.exifData())
      exiv_dst.writeMetadata()

      print('Copied EXIF from ' + filename + ' to ' + denoised_filename)


      #========== invoke darktable-cli with second stage ==========

      if args.rldeblur:
        s2_filename = basename + '_s2.tif'

        if os.path.exists(s2_filename):
          os.remove(s2_filename)
      else:
        s2_filename = out_filename

      cmd = cmd_darktable + ' "' + denoised_filename + '" "' + filename + '.s2.xmp" "' + s2_filename + '" ' + \
            '--icc-intent PERCEPTUAL --icc-type SRGB ' + \
            '--apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=16 '

      if args.debug:
        print('Second-stage cmd: ', cmd)

      subprocess.call(cmd, shell=True)


      # call ImageMagick RL-deblur
      if args.rldeblur:
        tmp_rl_filename = out_filename.replace(' ', '_')  # gmic can't handle spaces

        cmd = cmd_gmic + ' "' + s2_filename + '" ' + \
              '-deblur_richardsonlucy ' + str(args.sigma) + ',' + str(args.iteration) + ',1 ' + \
              '-/ 256 cut 0,255 round -o "' + tmp_rl_filename + ',' + str(args.quality) + '"'

        if args.debug:
          print('RL-deblur cmd: ', cmd)

        subprocess.call(cmd, shell=True)

        # rename tmp file
        os.rename(tmp_rl_filename, out_filename)
        print('Applied RL-deblur to:', out_filename)


      # copy exif
      exiv_src = exiv2.ImageFactory.open(s1_filename)
      exiv_src.readMetadata()
      exiv_dst = exiv2.ImageFactory.open(out_filename)
      exiv_dst.setExifData(exiv_src.exifData())
      exiv_dst.writeMetadata()

      print('Copied EXIF from', s1_filename, 'to', out_filename)

      # move output file into outdir
      shutil.move(out_filename, os.path.join(outdir, out_filename))
      print('Moved final output to ' + os.path.join(outdir, out_filename))


      #========== clean up ==========
      if not args.debug:
        os.remove(s1_filename)
        os.remove(denoised_filename)
        os.remove(filename + '.s1.xmp')
        os.remove(filename + '.s2.xmp')

        if (s2_filename != out_filename and os.path.exists(s2_filename)):
          os.remove(s2_filename)



# =================
if __name__ == "__main__":
   main(sys.argv[1:])
