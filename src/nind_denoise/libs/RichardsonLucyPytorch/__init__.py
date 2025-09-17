"""
PyTorch implementation of Richardson–Lucy deconvolution with a Gaussian PSF.

This module provides a small, self-contained backend that mirrors the behavior of
G'MIC's `-deblur_richardsonlucy {sigma},{iterations},1` for typical RGB images.

Key function
------------
- richardson_lucy_gaussian(image, sigma, iterations, device=None):
  Run RL deconvolution with a circularly symmetric Gaussian point-spread function.

Notes
-----
- Input is expected as a torch.Tensor in either HxWxC (uint8/float) or CxHxW (float).
- Values are normalized to [0, 1] for processing and converted back to the input dtype
  range on return.
- The algorithm is depthwise (per-channel) and uses reflection padding to reduce edge
  artifacts.
- Designed for small to moderate kernels (built in spatial domain). For very large
  sigmas, an FFT-based approach may be preferable but is unnecessary for our use case.
"""

from .richardson_lucy_deconvolution import richardson_lucy_gaussian
from .rl_config import RLConfig

__all__ = ["RLConfig", "richardson_lucy_gaussian"]
