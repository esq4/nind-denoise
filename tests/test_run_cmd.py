from pathlib import Path


def test_run_cmd_stringifies_and_uses_cwd(monkeypatch):
    from nind_denoise import config

    captured = {}

    def fake_run(args, cwd=None, check=False, text=None, capture_output=None):  # noqa: ARG001
        # Ensure args are converted to strings and cwd to str
        assert all(isinstance(a, str) for a in args)
        assert isinstance(cwd, str) or cwd is None
        captured["args"] = args
        captured["cwd"] = cwd

        # Simulate success
        class _Res:  # minimal CompletedProcess-like stub
            stdout = ""

        return _Res()

    monkeypatch.setattr(config.subprocess, "run", fake_run)

    # Provide Path arguments and cwd Path to ensure conversion
    config.run_cmd([Path("echo"), "hello"], cwd=Path("."))
    assert captured["args"][0] == "echo"


def test_run_cmd_raises_on_failure(monkeypatch):
    from nind_denoise import config

    def fake_run(args, cwd=None, check=False, text=None, capture_output=None):  # noqa: ARG001
        raise config.subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(config.subprocess, "run", fake_run)

    import pytest
    from nind_denoise import SubprocessError

    with pytest.raises(SubprocessError):
        config.run_cmd(["false"])  # any command; our fake raises regardless
