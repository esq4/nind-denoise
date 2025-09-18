from __future__ import annotations

import pathlib

from nind_denoise.pipeline.orchestrator import logger


#TODO wrong folder

def download_model_if_needed(model_path: pathlib.Path) -> None:
    """Download the model file if it does not exist."""
    from torch import hub

    if not model_path.exists():
        logger.info("Downloading denoiser model to %s", model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        hub.download_url_to_file(
            "https://f005.backblazeb2.com/file/modelzoo/nind/generator_650.pt",
            str(model_path),
        )
