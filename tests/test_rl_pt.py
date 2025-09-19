import importlib.machinery
import importlib.util
import pathlib
import sys

import numpy as np
from PIL import Image, ImageFilter

# Dynamically load pipeline.py from src without installing the package
_path = str(
    pathlib.Path(__file__).resolve().parents[1] / "src" / "nind_denoise" / "pipeline.py"
)
_loader = importlib.machinery.SourceFileLoader("pipeline_local_rlpt", _path)
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
_pipeline = importlib.util.module_from_spec(_spec)
# Important for dataclasses and decorators
sys.modules[_loader.name] = _pipeline
_loader.exec_module(_pipeline)

Context = _pipeline.Context


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    # Peak Signal-to-Noise Ratio (dB)
    mse = np.mean((ref.astype(np.float32) - test.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    PIXMAX = 255.0
    return 20.0 * np.log10(PIXMAX) - 10.0 * np.log10(mse)


def _make_sharp_test_image(size=(96, 96)) -> Image.Image:
    # Create a simple high-contrast pattern: black background with white square + line
    w, h = size
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    # Draw a centered white rectangle
    arr = np.array(img)
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    arr[y0:y1, x0:x1, :] = 255
    # Add a diagonal line
    for i in range(min(w, h)):
        arr[i, i, :] = 255
    return Image.fromarray(arr, mode="RGB")


def test_rlpt_deblur_runs_without_subprocess(tmp_path, monkeypatch):
    # Ensure no subprocess calls are made by the PyTorch backend
    def fail_run(*a, **k):
        raise AssertionError("subprocess.run should not be called by RLDeblurPT")

    monkeypatch.setattr(_pipeline.subprocess, "run", fail_run)

    # Prepare a blurred input TIFF (stage-2)
    sharp = _make_sharp_test_image((64, 64))
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=1.0))

    s2 = tmp_path / "input_s2.tif"
    blurred.save(s2)

    out = tmp_path / "out.jpg"

    # Lazily import class after module load (will be added by our implementation)
    RLDeblurPT = getattr(_pipeline, "RLDeblurPT")

    ctx = Context(
        outpath=out,
        stage_two_output_filepath=s2,
        sigma=1,
        iteration="5",
        quality="90",
        cmd_gmic="gmic",  # ignored by PT backend
        output_dir=tmp_path,
        verbose=True,
    )

    RLDeblurPT().execute(ctx)
    assert out.exists(), "RLDeblurPT did not produce an output file"


def test_rlpt_improves_psnr_on_gaussian_blur(tmp_path, monkeypatch):
    # No subprocess calls should be made
    def fail_run(*a, **k):
        raise AssertionError("subprocess.run should not be called by RLDeblurPT")

    monkeypatch.setattr(_pipeline.subprocess, "run", fail_run)

    # Create sharp reference and a blurred observation
    sharp = _make_sharp_test_image((96, 96))
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=1.0))

    s2 = tmp_path / "input_s2.tif"
    blurred.save(s2)

    out = tmp_path / "out.jpg"

    RLDeblurPT = getattr(_pipeline, "RLDeblurPT")

    ctx = Context(
        outpath=out,
        stage_two_output_filepath=s2,
        sigma=1,  # match blur radius
        iteration="10",  # modest number of iterations
        quality="92",
        cmd_gmic="gmic",
        output_dir=tmp_path,
        verbose=False,
    )

    RLDeblurPT().execute(ctx)
    assert out.exists(), "Output was not created"

    # Compare PSNR w.r.t the original sharp
    sharp_np = np.array(sharp.convert("RGB"), dtype=np.uint8)
    blurred_np = np.array(blurred.convert("RGB"), dtype=np.uint8)
    out_np = np.array(Image.open(out).convert("RGB"), dtype=np.uint8)

    psnr_blur = _psnr(sharp_np, blurred_np)
    psnr_out = _psnr(sharp_np, out_np)

    # Expect a noticeable improvement over the blurred observation
    assert (
        psnr_out > psnr_blur + 0.1
    ), f"PSNR did not improve: blur={psnr_blur:.3f}, out={psnr_out:.3f}"
