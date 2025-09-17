from pathlib import Path

from nind_denoise.config.config import Config
from nind_denoise.pipeline.base import JobContext
from nind_denoise.pipeline.deblur import RLDeblur


def test_rldeblur_builds_args_and_renames(monkeypatch, tmp_path):
    # Arrange a fake environment
    output_dir = tmp_path
    stage2 = output_dir / "stage2.tif"
    stage2.write_bytes(b"")
    outpath = output_dir / "final.jpg"

    captured = {}

    def fake_run_cmd(self, args, cwd=None):  # noqa: D401, ANN001
        # Capture and create the expected tmp output file to simulate gmic
        captured["args"] = [str(a) for a in args]
        captured["cwd"] = str(cwd) if cwd is not None else None
        # The last two args are: "-o_jpg", f"{tmp_out.name},{quality}"
        tmp_spec = captured["args"][captured["args"].index("-o_jpg") + 1]
        tmp_name = tmp_spec.split(",")[0]
        (Path(cwd) / tmp_name).write_bytes(b"")

    monkeypatch.setattr(RLDeblur, "_run_cmd", fake_run_cmd, raising=True)

    # Create mock tools for Environment
    cfg = Config(verbose=True)
    job_ctx = JobContext(
        input_path=stage2,
        output_path=outpath,
        sigma=2,
        iterations=5,
        quality=92,
        intermediate_path=stage2,
    )

    # Execute
    RLDeblur().execute_with_env(cfg, job_ctx)

    # Verify args constructed and tmp was renamed to final outpath
    args = captured["args"]
    assert args[0].endswith("gmic")
    assert args[1] == stage2.name
    assert "-fx_sharpen_reinhard" in args
    assert "-o_jpg" in args
    assert outpath.exists()
