from __future__ import annotations

from pathlib import Path

from .brummer2019 import DenoiseOperation, DenoiseOptions, NIND
from ..base import StageError


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

        denoiser = NIND(self.options)
        denoiser.run(self.input_tif, self.output_tif)
        self.verify()

    def verify(self, ctx: Context | None = None) -> None:
        if not self.output_tif.exists():
            raise StageError(f"Denoise stage expected output missing: {self.output_tif}")
