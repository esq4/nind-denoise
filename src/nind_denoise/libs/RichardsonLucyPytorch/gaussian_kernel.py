from typing import Optional, Tuple

import torch

_EPS = 1e-8


class GaussianKernel:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def generate_kernel(
        self,
        device: torch.device,
        dtype: torch.dtype,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.sigma <= 0:
            return torch.ones((1, 1), device=device, dtype=dtype)

        # Calculate default radius based on sigma
        radius = int(max(1, round(3.0 * float(self.sigma))))

        # Limit kernel size based on max_size if provided
        if max_size is not None:
            max_h, max_w = max_size
            # Ensure kernel fits within input dimensions
            max_radius = min(max_h // 2, max_w // 2)
            if max_radius < 1:
                max_radius = 1
            radius = min(radius, max_radius)

        size = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        y = x[:, None]
        g = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        g = g / (g.sum() + _EPS)
        return g
