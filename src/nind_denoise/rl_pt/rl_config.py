from dataclasses import dataclass

@dataclass(frozen=True)
class RLConfig:
    sigma: float
    iterations: int
    pad_mode: str = "reflect"  # good default for deconvolution stability
