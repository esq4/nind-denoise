from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..base import Context, DenoiseOperation, StageError


@dataclass
class DenoiseOptions:
    """Options for the denoiser stage.

    Fields are kept simple for now; real implementations can use device/tile configs.
    """

    model_path: Path
    device: Optional[str] = None  # e.g., "cpu", "cuda", "mps"
    overlap: int = 6
    batch_size: int = 1


class _NINDPTDenoiser:
    """Minimal placeholder for a PyTorch-based denoiser.

    For now, it simply copies the input TIFF to the output to keep the pipeline
    operational without imposing a heavyweight dependency or runtime.
    """

    def __init__(self, options: DenoiseOptions):
        self.options = options

    def run(self, input_tif: Path, output_tif: Path) -> None:
        output_tif.parent.mkdir(parents=True, exist_ok=True)
        # Minimal behavior: copy input to output
        import shutil

        shutil.copy2(input_tif, output_tif)


class DenoiseStage(DenoiseOperation):
    def __init__(self, input_tif: Path, output_tif: Path, options: DenoiseOptions):
        self.input_tif = input_tif
        self.output_tif = output_tif
        self.options = options

    def describe(self) -> str:  # pragma: no cover - description only
        return "Denoise (NIND PT)"

    def execute(self, ctx: Context) -> None:
        # Store context for helpers/logging
        self._ctx = ctx  # type: ignore[attr-defined]

        if not self.input_tif.exists():
            raise StageError(f"Input TIFF missing for denoise: {self.input_tif}")

        denoiser = _NINDPTDenoiser(self.options)
        denoiser.run(self.input_tif, self.output_tif)
        self.verify()

    def verify(self, ctx: Context | None = None) -> None:
        if not self.output_tif.exists():
            raise StageError(
                f"Denoise stage expected output missing: {self.output_tif}"
            )
