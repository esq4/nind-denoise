# nind-denoise
This fork implements the following changes:
- Improved code portability via ```torch.accelerator```. This should be able to use most hardware.
  - Support for Intel GPU's. Denoising (on mine) is about 6x faster than on cpu. (YMMV)
- Autodownload pretrained model weights from backblaze b2 when they're missing. 
  - This should simplify installation but does remove the ability to BYOM(odel) or pick a different one. Revisit later.

## Requirements

 Make sure the right drivers for your system are installed along with OpenCL. 
 This varies by distribution but should look something like (this is arch linux) 
 ```intel-compute-runtime```, ```extra/intel-graphics-compiler```, ```vulkan-intel``` and ```onednn```.
 I'm not sure if all of those are required - but it's a place to start. Proper operation can be verified with:
 ```clinfo | grep device```

## Installation:

Warning: this is a prototype developmental codebase. 
The following should be considered developer documentation; it is not a user install guide.


To install, run the following commands. This should pull the right version of PyTorch for your gpu/cpu/xpu/whatever, but
if it doesn't, or you just want to avoid the possibility of iterating, continue reading below for how to modify to match
your hardware/environment. 

Good luck. Check out https://download.pytorch.org/whl/torch/ if you want to see a list of pytorch versions. 

```
git clone https://github.com/commreteris/nind-denoise.git
cd nind-denoise
git checkout darktable-cli-xpu
uv venv
.venv/Scripts/activate
uv add -r requirements.in
```

### nVIDIA CUDA
For CUDA 12.9, you would do:
```
uv add -r requirements.in --index "https://download.pytorch.org/whl/cu129"
```

### Intel GPU
```
uv add -r requirements.in --index "https://download.pytorch.org/whl/xpu"
```

### AMD ROCM
```
uv add -r requirements.in --index "https://download.pytorch.org/whl/rocm6.3"
```

## Usage
=======


To denoise an image, run:

```
python3 src/denoise.py "/path/to/photo0123.RAW"
```
