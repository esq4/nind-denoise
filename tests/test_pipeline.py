import pathlib
import types
import importlib.machinery
import importlib.util

import numpy as np
from PIL import Image
import pytest

import nind_denoise.pipeline as _pipeline

Context = _pipeline.Context
NoOpDeblur = _pipeline.NoOpDeblur
RLDeblur = _pipeline.RLDeblur


def test_noop_deblur_does_not_invoke_subprocess(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run(*a, **k):
        calls.append((a, k))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(_pipeline.subprocess, "run", fake_run)

    outpath = tmp_path / "x.jpg"
    ctx = Context(
        outpath=outpath,
        stage_two_output_filepath=tmp_path / "x_s2.tif",
        sigma=1,
        iteration="10",
        quality="90",
        cmd_gmic="gmic",
        output_dir=tmp_path,
        verbose=True,
    )
    NoOpDeblur().execute(ctx)
    assert calls == []


def test_rl_deblur_invokes_gmic_and_handles_spaces(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, check=None):
        calls.append((cmd, cwd, check))
        # write output file to simulate gmic behavior
        outname = cmd[-1].split(",")[0]
        (cwd / outname).write_text("data")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(_pipeline.subprocess, "run", fake_run)

    outpath = tmp_path / "my photo.jpg"  # contains space
    outpath.touch()
    ctx = Context(
        outpath=outpath,
        stage_two_output_filepath=tmp_path / "x_s2.tif",
        sigma=2,
        iteration="5",
        quality="85",
        cmd_gmic="gmic",
        output_dir=tmp_path,
        verbose=False,
    )
    RLDeblur().execute(ctx)
    # Ensure a gmic call was made
    assert calls and "gmic" in calls[0][0][0]
    # Original space-containing filename should be restored
    assert (tmp_path / "my photo.jpg").exists()




def _load_denoise_module():
    path = str(pathlib.Path(__file__).resolve().parents[1] / "src" / "denoise.py")
    loader = importlib.machinery.SourceFileLoader("denoise_local_ssim", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _read_image(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    # grayscale SSIM (OpenCV)
    import cv2

    x = a.astype(np.float32)
    y = b.astype(np.float32)
    xg = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255.0
    yg = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY) / 255.0
    ksize = (7, 7)
    sigma = 1.5
    mu_x = cv2.GaussianBlur(xg, ksize, sigma)
    mu_y = cv2.GaussianBlur(yg, ksize, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(xg * xg, ksize, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(yg * yg, ksize, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(xg * yg, ksize, sigma) - mu_xy
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )
    return float(ssim_map.mean())


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    # Peak Signal-to-Noise Ratio
    mse = np.mean((ref.astype(np.float32) - test.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    PIXMAX = 255.0
    return 20.0 * np.log10(PIXMAX) - 10.0 * np.log10(mse)


@pytest.mark.parametrize("ext", ["jpg", "tiff"])
def test_raw_pipeline_matches_sample_ssim(tmp_path, ext):

    # find a RAW that has a paired sample JPG
    raw_dir = pathlib.Path(__file__).parent / "test_raw"
    candidates = []
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _pipeline.valid_extensions:
            continue
        jpg = p.with_suffix(".jpg")
        if jpg.exists():
            candidates.append((p, jpg))
    if not candidates:
        pytest.skip("No RAW+JPG sample pair found in tests/test_raw")

    # just use the first available pair
    raw, sample_jpg = candidates[0]

    # run the pipeline writing to tmp_path
    outdir = tmp_path
    args = {
        "--output-path": str(outdir),
        "--extension": ext,
        "--dt": None,
        "--gmic": None,
        "--sigma": "1",
        "--quality": "90",
        "--iterations": "10",
        "--verbose": False,
        "--debug": True,
    }

    _pipeline.run_pipeline(args, raw)

    # expected output path
    out = (outdir / raw.name).with_suffix("." + ext)
    assert out.exists(), f"Output not created: {out}"

    # SSIM between produced output and provided sample JPG should be near 1
    out_img = _read_image(out)
    sample_img = _read_image(sample_jpg)
    s = _ssim(sample_img, out_img)
    if ext.lower() == "jpg":
        assert s >= 0.999, f"SSIM too low for JPG: {s:.6f}"
    else:
        # converting JPG->TIFF is near-lossless but not exact; be lenient
        assert s >= 0.99, f"SSIM too low for TIFF: {s:.6f}"


def test_raw_pipeline_psnr_improves(tmp_path, monkeypatch):
    raw_dir = pathlib.Path(__file__).parent / "test_raw"
    # pick the same RAW+JPG pair
    for p in raw_dir.iterdir():
        if (
            p.is_file()
            and p.with_suffix(".jpg").exists()
            and p.suffix.lower() in _pipeline.valid_extensions
        ):
            raw = p
            sample_jpg = p.with_suffix(".jpg")
            break
    else:
        pytest.skip("No RAW+JPG sample pair found")

    base_args = {
        "--output-path": None,
        "--extension": "jpg",
        "--dt": None,
        "--gmic": None,
        "--sigma": "1",
        "--quality": "90",
        "--iterations": "10",
        "--verbose": False,
        "--debug": True,
    }

    # 1) Run full pipeline (with denoising)
    out_with = tmp_path / f"{raw.stem}_with.jpg"
    args_with = dict(base_args)
    args_with["--output-path"] = str(out_with)
    _pipeline.run_pipeline(args_with, raw)
    assert out_with.exists(), f"Full pipeline output missing: {out_with}"

    # 2) Run pipeline again but patch the denoising step to be a no-op (copy s1 -> s1_denoised)
    out_no = tmp_path / f"{raw.stem}_no_denoise.jpg"
    args_no = dict(base_args)
    args_no["--output-path"] = str(out_no)

    # Compute expected stage-1 and stage-1-denoised filepaths for this run
    s1_no, s1d_no = _pipeline.get_stage_filepaths(out_no, 1)

    # Keep original run to delegate for everything else
    orig_run = _pipeline.subprocess.run

    def fake_run(cmd, *a, **k):
        # Detect the denoiser invocation by the presence of the denoise_image.py script in argv
        cmd_strs = [str(c) for c in (cmd if isinstance(cmd, (list, tuple)) else [cmd])]
        if any(
            part.endswith("denoise_image.py")
            or "nind_denoise" in part
            and "denoise_image.py" in part
            for part in cmd_strs
        ):
            # Simulate denoiser by copying the stage-1 output to the expected denoised path
            s1_no.parent.mkdir(parents=True, exist_ok=True)
            if not s1_no.exists():
                # If stage-1 export hasn't been simulated yet, create a placeholder identical to sample JPG
                # This keeps the test resilient in environments without darktable.
                from shutil import copyfile

                copyfile(sample_jpg, s1_no)
            from shutil import copyfile

            copyfile(s1_no, s1d_no)
            return types.SimpleNamespace(returncode=0)
        # Delegate to real run for other commands
        return orig_run(cmd, *a, **k)

    monkeypatch.setattr(_pipeline.subprocess, "run", fake_run)

    _pipeline.run_pipeline(args_no, raw)
    assert out_no.exists(), f"No-denoise pipeline output missing: {out_no}"

    # Compare PSNRs against the ground-truth sample JPG
    ref = _read_image(sample_jpg)
    img_with = _read_image(out_with)
    img_no = _read_image(out_no)

    psnr_with = _psnr(ref, img_with)
    psnr_no = _psnr(ref, img_no)
    assert (
        psnr_with > psnr_no
    ), f"PSNR did not improve with denoising: with={psnr_with:.3f}, no={psnr_no:.3f}"
