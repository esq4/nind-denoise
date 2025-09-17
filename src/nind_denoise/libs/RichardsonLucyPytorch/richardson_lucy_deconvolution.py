from typing import Optional

import torch
import torch.nn.functional as F

from .gaussian_kernel import GaussianKernel
from .rl_config import RLConfig

_EPS = 1e-8


class RichardsonLucyDeconvolution:  # TODO: fix this - it needs the CUT from gmic to put the values in the right range (0,255)
    def __init__(self, sigma: float, iterations: int, pad_mode: str = "reflect"):
        self.config = RLConfig(sigma, iterations, pad_mode)

    @staticmethod
    def _depthwise_conv2d(
        x: torch.Tensor, k: torch.Tensor, pad_mode: str
    ) -> torch.Tensor:
        N, C, H, W = x.shape
        kh, kw = k.shape

        # Limit padding to prevent exceeding input dimensions
        pad_h = min(kh // 2, H - 1)
        pad_w = min(kw // 2, W - 1)

        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)
        w = k.expand(C, 1, kh, kw)
        return F.conv2d(x_pad, w, bias=None, stride=1, padding=0, groups=C)

    def deconvolve(
        self, image: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if self.config.iterations <= 0:
            return image.clone()

        orig_dtype = image.dtype
        layout = "CHW" if image.ndim == 3 and image.shape[0] in (1, 3, 4) else "HWC"

        x = image.permute(2, 0, 1) if layout == "HWC" else image

        x = x.to(torch.float32)
        if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            x = x / 255.0

        device = self._choose_device(device)

        x = x.unsqueeze(0).to(device)

        # Get the spatial dimensions for kernel size limiting
        _, _, H, W = x.shape
        kernel_generator = GaussianKernel(self.config.sigma)
        # Pass image dimensions to limit kernel size appropriately
        k = kernel_generator.generate_kernel(
            device=device, dtype=x.dtype, max_size=(H, W)
        )
        k_flip = torch.flip(k, dims=(0, 1))

        y = x.clone()
        for _ in range(int(self.config.iterations)):
            conv_x = self._depthwise_conv2d(x, k, self.config.pad_mode)
            ratio = y / (conv_x + _EPS)
            corr = self._depthwise_conv2d(ratio, k_flip, self.config.pad_mode)
            x = x * corr
            x = x.clamp_min(0.0)

        x = x.squeeze(0)

        if layout == "HWC":
            x = x.permute(1, 2, 0)

        x = x.clamp(0.0, 1.0)

        if orig_dtype == torch.uint8:
            x = (x * 255.0).round().to(torch.uint8)
        else:
            x = x.to(torch.float32)

        return x

    @staticmethod
    def _choose_device(device: Optional[torch.device]) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel XPU
                return torch.device("xpu")
            else:
                return torch.device("cpu")


def richardson_lucy_gaussian(
    image: torch.Tensor,
    sigma: float,
    iterations: int,
    device: Optional[torch.device] = None,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Perform RL deconvolution with a Gaussian PSF on an image.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor. Accepts HxWxC (uint8/float) or CxHxW (float). Values can be
        in [0, 255] uint8 or arbitrary floats; they will be normalized to [0, 1].
    sigma : float
        Standard deviation of the Gaussian PSF (pixels). Roughly matches Pillow's
        GaussianBlur radius and G'MIC parameterization for small sigmas.
    iterations : int
        Number of RL iterations to perform. Typical values 5–20.
    device : Optional[torch.device]
        Device to execute on. If None, chooses CUDA/XPU/MPS/CPU automatically.
    pad_mode : str
        Padding mode passed to torch.nn.functional.pad (default "reflect").

    Returns
    -------
    torch.Tensor
        Tensor with the same shape layout as the input (HxWxC or CxHxW) and dtype float32
        in [0, 1] if input was float, or uint8 in [0, 255] if input was uint8.
    """
    deconv = RichardsonLucyDeconvolution(sigma, iterations, pad_mode)
    return deconv.deconvolve(image)
