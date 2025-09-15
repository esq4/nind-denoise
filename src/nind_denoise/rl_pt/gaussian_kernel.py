import torch

_EPS = 1e-8

class GaussianKernel:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def generate_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sigma <= 0:
            return torch.ones((1, 1), device=device, dtype=dtype)

        radius = int(max(1, round(3.0 * float(self.sigma))))
        size = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        y = x[:, None]
        g = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        g = g / (g.sum() + _EPS)
        return g
