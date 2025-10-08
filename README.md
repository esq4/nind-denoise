nind-denoise
==============

A pytorch based image denoising tool for the removal of noise from real photographs. Implements the models developed in    
Benoit Brummer's [NIND Denoise](https://github.com/trougnouf/nind-denoise.git), and the [Darktable](https://github.com/darktable-org/darktable) workflow pioneered by [Huy Hoang](https://github.com/hqhoang).

This fork makes their work easier to experiment with and accessible to a wider audience. Notable features include:
- **One-click setup**: Download the Lua script and let Darktable handle all Python dependencies automatically
- Support for most hardware via `torch.accelerator`
  - CPU-only (universal), nVidia GPU (CUDA), Intel XPU/GPU, and AMD GPU acceleration
  - Intel Xe graphics are about 6x faster than CPU, though slower than CUDA
- Automatic darktable integration with export workflow
- Automatic model download and environment setup
- Processed images automatically imported and grouped with originals in Darktable library

# Quick Start (Recommended)

The easiest way to use nind-denoise is through the Darktable Lua plugin with automatic environment setup.

## 1. Install Lua Scripts and Download This Script

First, install the community lua-scripts collection through Darktable's built-in interface (this will create the necessary folder structure). See the [official lua-scripts repository](https://github.com/darktable-org/lua-scripts) for installation instructions.

Once installed, download `nind_denoise_rl.lua` from this repository and place it in the contrib folder:

**Linux:**
```bash
curl -o ~/.config/darktable/lua/contrib/nind_denoise_rl.lua \
  https://raw.githubusercontent.com/commreteris/nind-denoise/lua/src/lua-scripts/nind_denoise_rl.lua
```

**macOS:**
```bash
curl -o ~/Library/Application\ Support/darktable/lua/contrib/nind_denoise_rl.lua \
  https://raw.githubusercontent.com/commreteris/nind-denoise/lua/src/lua-scripts/nind_denoise_rl.lua
```

**Windows:**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/commreteris/nind-denoise/lua/src/lua-scripts/nind_denoise_rl.lua" `
  -OutFile "$env:LOCALAPPDATA\darktable\lua\contrib\nind_denoise_rl.lua"
```

Or manually download the file and copy it to the contrib folder:
- Linux: `~/.config/darktable/lua/contrib/`
- macOS: `~/Library/Application Support/darktable/lua/contrib/`
- Windows: `%LOCALAPPDATA%\darktable\lua\contrib\`

## 2. Set Up Python Environment (One-Time Setup)

1. Start Darktable and enable the script from the Script Manager (`lighttable > script manager`)
2. In the lighttable module, find the **"Update Environment"** button
3. Click it and wait while the script automatically:
   - Installs the `uv` package manager (if needed)
   - Creates a Python virtual environment
   - Downloads the denoising model (~50MB)
   - Installs all required dependencies (PyTorch, GMic, tifffile, etc.)

**Note:** First-time setup can take several minutes depending on your internet connection and system. The script will display status updates. This is a one-time process.

## 3. Start Denoising

1. Select one or more images in the lighttable
2. Go to the export module
3. Choose **"NIND-denoise RL"** as the target storage
4. Configure export settings:
   - Select JPEG or TIFF output format
   - Adjust RL deblur parameters (sigma, iterations)
   - Check **"import to darktable"** to automatically import processed images back to your library
5. Click export

Processed images will be automatically grouped with their originals in your Darktable library!

---

# Advanced Usage

## Command-Line Interface

For advanced users, custom workflows, or batch processing, you can use the command-line interface directly.

### Basic Usage

To denoise an image, run:

```console
$ python3 src/denoise.py "/path/to/photo0123.RAW"
```

**Note:** On Windows, if you use forward slashes _do not_ use single forward slashes for paths. Double backslashes are OK:

```powershell 
PS> python3 src\\denoise.py "\\good\\path\\to\\photo0123.RAW"
```

### Full Command-Line Options

```python
"""
Usage:
    denoise.py [-o <outpath> | --output-path=<outpath>] [-e <e> | --extension=<e>]
                    [-d <darktable> | --dt=<darktable>] [-g <gmic> | --gmic=<gmic>] [ -q <q> | --quality=<q>]
                    [--nightmode ] [ --no_deblur ] [ --debug ] [ --sigma=<sigma> ] [ --iterations=<iter> ]
                    [-v | --verbose] <raw_image>
    denoise.py (help | -h | --help)
    denoise.py --version

Options:
  -o <outpath> --output-path=<outpath>  Where to save the result (defaults to current directory).
  -e <e> --extension=<e>                Output file extension [default: jpg].
  --dt=<darktable>                      Path to darktable-cli. Use this only if not automatically found.
  -g <gmic> --gmic=<gmic>               Path to gmic. Use this only if not automatically found.
  -q <q> --quality=<q>                  JPEG compression quality [default: 90].
  --nightmode                           Use for very dark images. Normalizes brightness before denoise [default: False].
  --no_deblur                           Do not perform RL-deblur [default: false].
  --debug                               Keep intermediate files.
  --sigma=<sigma>                       Sigma to use for RL-deblur [default: 1].
  --iterations=<iter>                   Number of iterations for RL-deblur [default: 10].
  -v --verbose
  --version                             Show version.
  -h --help                             Show this screen.
"""
```

## Manual Python Environment Setup

If you prefer to set up the Python environment manually (not using the Darktable auto-setup), follow these steps:

### 1. Clone this Repository

```console
git clone https://github.com/commreteris/nind-denoise.git
cd nind-denoise
```

### 2. Create Virtual Environment

The variant-enabled version of `uv` automatically installs the correct version of PyTorch for your GPU.

**Windows:**
```powershell
PS> powershell -c { $env:INSTALLER_DOWNLOAD_URL = 'https://wheelnext.astral.sh'; irm https://astral.sh/uv/install.ps1 | iex }
PS> uv venv
PS> .venv/Scripts/activate
(nind-denoise) PS> Get-Command python

CommandType   Name          Version    Definition
-----------   ----          -------    ----------
Application   python.exe    3.1x.xx    C:\Users\<user>\...\nind-denoise\.venv/scripts\python.exe
```

Make sure `python.exe` is inside the `.venv` directory before proceeding.

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
uv venv
source .venv/bin/activate
which python
# Should show: /path/to/nind-denoise/.venv/bin/python
```

Make sure the Python path is inside the `.venv` directory before proceeding.

### 3. Install Python Packages

```console
$ uv pip install -r requirements.in --upgrade
```

### 4. Verify GPU Acceleration (Optional)

```console
(nind-denoise) $ python
>>> import torch
>>> torch.accelerator.is_available()
True
>>> torch.cuda.is_available()  # For nVidia GPUs
True
>>> torch.xpu.is_available()   # For Intel GPUs
False
```

---

# Requirements

## For Quick Start (Darktable Integration)
- Darktable installed and working
- Internet connection for first-time setup
- That's it! The Lua script handles everything else automatically.

## For Manual/Advanced Setup

### Software
- Darktable and raw images processed with darktable (with `.xmp` sidecar files)
- GMic CLI ([download here](https://gmic.eu/download.html) for Windows)
- [Variant-enabled uv](https://astral.sh/blog/wheel-variants) package manager
- Proper GPU drivers with OpenCL support

### Darktable Version
- If [this PR](https://github.com/darktable-org/darktable/pull/19189) is still open, you may need to use version [5.2.1](https://github.com/darktable-org/darktable/releases/tag/release-5.0.1) for RL-deblur functionality.

### GPU Drivers

**nVidia:**
- Driver + CUDA toolkit

**AMD:**
- Driver + ROCm 

**Intel:**
- Intel GPU drivers are slightly trickier due to version mismatches. Install minimum system packages and let uv/pip handle dependencies in the venv.
- See Intel's [driver guide](https://dgpu-docs.intel.com/driver/client/overview.html)
- Install `libze-dev` and `intel-ocloc` for PyTorch support
- On Arch Linux: `intel-compute-runtime`, `level-zero-loader`, `level-zero-headers`
- Minimum Linux kernel: 6.14

**Verify GPU setup:**
```console
$ clinfo | grep device
$ darktable-cltest
```

---

# Citation

Please cite Benoit Brummer's original work:

```bibtex
@InProceedings{Brummer_2019_CVPR_Workshops,
author = {Brummer, Benoit and De Vleeschouwer, Christophe},
title = {Natural Image Noise Dataset},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```
