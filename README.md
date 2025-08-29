nind-denoise
==============

A pytorch based image denoising tool for the removal of noise from real photographs. Implements the models developed in    
Benoit Brummer's [NIND Denoise](https://github.com/trougnouf/nind-denoise.git), and the 
[Darktable](https://github.com/darktable-org/darktable) workflow pioneered by [Huy Hoang](https://github.com/hqhoang).

This fork seeks make their work easier to experiment with, and to make it accessible to a wider audience. 
- Improved code portability via ```torch.accelerator```. 
  - In addition to supporting cpu-only and NVIDIA GPUs, this adds support for AMD & Intel GPU's, among others. Denoising (on my Iris Xe MAX) is about 6x
  faster than on cpu. (YMMV) 
- Documented installation process on Windows and linux 
  - Installs in a python virtual environment with [uv](https://github.com/astral-sh/uv)
  - Autodownload pretrained model weights from backblaze b2 -> simpler installation
  - To uninstall, just delete the directory.

- [...loading]

# Usage

To denoise an image, run:

```console
$ python3 src/denoise.py "/path/to/photo0123.RAW"
```

# Requirements

 Make sure the right drivers for your system are installed along with OpenCL. 
 This varies by distribution but should look something like (this is arch linux) 
 `opencl-headers`, ```intel-compute-runtime```, ```extra/intel-graphics-compiler```, ```vulkan-intel``` and ```onednn```.
 I'm not sure if all of those are required - but it's a place to start. Proper operation can be verified with:
 ```clinfo | grep device```


# Installation:

Warning: this is a prototype developmental codebase. 
The following should be considered developer documentation; it is not a user install guide.


To install, run the following commands. This should pull the right version of PyTorch for your gpu/cpu/xpu/whatever, but
if it doesn't, or you just want to avoid the possibility of iterating, continue reading below for how to modify to match
your hardware/environment. 

Good luck. Check out https://download.pytorch.org/whl/torch/ if you want to see a list of pytorch versions. 

### 1 - Clone this repo

```bash
git clone https://github.com/commreteris/nind-denoise.git
cd nind-denoise
```

### 2 - Create a virtual environment with `uv venv` and activate it.

On **Windows** that would look something like this:

```powershell
PS> uv venv
PS> .venv/Scripts/activate
(nind-denoise) PS> Get-Command eventvwr

CommandType   Name          Version    Definition
-----------   ----          -------    ----------
Application   python.exe    3.1x.xx    C:\Users\<user>\...\nind-denoise\.venv/scripts\python.exe


PS>
```

On **Linux**, the following will work for most people. If you are a [fish](https://en.wikipedia.org/wiki/Fish_(Unix_shell)), or
otherwise out of water, go [here](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment) for help. 

 ```bash
 [user@linux]$ uv venv
 [user@linux]$ source .venv/bin/activate
 [user@linux]$ which python
 /home/user/.../nind-denoise/.venv/bin/python
[user@linux]$
 ```

## 3 Install requirements

Installing required packages into your `venv` should be the same for all operating systems, but it may have to be 
tweaked to match your hardware (_i.e.,_ GPU or lack thereof). This command _should_ work for all, but if it doesn't 
just scroll down and pick the right one yourself. 

```console
$ uv add -r requirements.in --upgrade
```

### nVIDIA CUDA
For CUDA 12.9, you would do:
```
uv add -r requirements.in --index "https://download.pytorch.org/whl/cu129 --upgrade"
```

### Intel GPU
```
uv add -r requirements.in --index "https://download.pytorch.org/whl/xpu --upgrade"
```

### AMD ROCM
```
uv add -r requirements.in --index "https://download.pytorch.org/whl/rocm6.3 --upgrade"
```

## Citations

Go cite Benoit

```bibtex
@InProceedings{Brummer_2019_CVPR_Workshops,
author = {Brummer, Benoit and De Vleeschouwer, Christophe},
title = {Natural Image Noise Dataset},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
} 
```