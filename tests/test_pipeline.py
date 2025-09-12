import importlib.machinery
import importlib.util
import pathlib
import sys
import shutil
import numpy as np
from PIL import Image
import pytest

# Load pipeline from src
_path = str(
    pathlib.Path(__file__).resolve().parents[1] / "src" / "nind_denoise" / "pipeline.py"
)
_loader = importlib.machinery.SourceFileLoader("pipeline_local", _path)
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
_pipeline = importlib.util.module_from_spec(_spec)
# Register module to satisfy dataclasses type resolution
sys.modules[_loader.name] = _pipeline
_loader.exec_module(_pipeline)

Context = _pipeline.Context
NoOpDeblur = _pipeline.NoOpDeblur
RLDeblur = _pipeline.RLDeblur


def test_noop_deblur_runs(tmp_path):
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
    # Should not raise and should not create any files
    NoOpDeblur().execute(ctx)
    assert not outpath.exists()


@pytest.mark.integration
def test_rl_deblur_executes_when_gmic_available(tmp_path):
    # Skip if gmic is not available on PATH
    gmic_path = shutil.which("gmic") or shutil.which("gmic.exe")
    if not gmic_path:
        pytest.skip("gmic not available on PATH")

    # Create a tiny valid TIFF as stage-2 input
    s2 = tmp_path / "x_s2.tif"
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(s2)

    outpath = tmp_path / "out.jpg"
    ctx = Context(
        outpath=outpath,
        stage_two_output_filepath=s2,
        sigma=1,
        iteration="5",
        quality="90",
        cmd_gmic=gmic_path,
        output_dir=tmp_path,
        verbose=False,
    )

    RLDeblur().execute(ctx)
    assert outpath.exists(), f"RL-deblur did not produce output: {outpath}"


def _read_image(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
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


@pytest.mark.parametrize("ext", ["jpg", "tiff"])
@pytest.mark.integration
def test_raw_pipeline_matches_sample_ssim(tmp_path, ext):
    # Ensure darktable-cli is available, otherwise skip
    dt_path = shutil.which("darktable-cli") or shutil.which("darktable-cli.exe")
    if not dt_path:
        pytest.skip("darktable-cli not available on PATH")

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

    raw, sample_jpg = candidates[0]

    # run the pipeline writing to tmp_path
    outdir = tmp_path
    args = {
        "--output-path": str(outdir),
        "--extension": ext,
        "--dt": dt_path,
        "--gmic": None,  # allow pipeline to disable RL-deblur if missing
        "--sigma": "1",
        "--quality": "90",
        "--iterations": "10",
        "--verbose": False,
        "--debug": True,
    }

    _pipeline.run_pipeline(args, raw)

    out = (outdir / raw.name).with_suffix("." + ext)
    assert out.exists(), f"Output not created: {out}"

    out_img = _read_image(out)
    sample_img = _read_image(sample_jpg)
    s = _ssim(sample_img, out_img)
    if ext.lower() == "jpg":
        assert s >= 0.999, f"SSIM too low for JPG: {s:.6f}"
    else:
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
