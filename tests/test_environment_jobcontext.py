"""Test Environment/JobContext split functionality."""

import dataclasses
from pathlib import Path

import pytest

from nind_denoise.config import Tools
from nind_denoise.config.config import Config
from nind_denoise.pipeline.base import JobContext


def test_environment_immutability():
    """Test that Environment is immutable (frozen dataclass)."""
    tools = Tools(darktable=Path("dt"), gmic=Path("gmic"))
    config = {"test": "value"}

    cfg = Config(tools=tools, config=config, verbose=True, device="cpu")

    # Environment should be frozen - attempting to modify should raise error
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.verbose = False

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.device = "cuda"

    # Verify values are correctly set
    assert cfg.tools == tools
    assert cfg.config == config
    assert cfg.verbose is True
    assert cfg.device == "cpu"


def test_jobcontext_mutability():
    """Test that JobContext is mutable and has typed fields."""
    input_path = Path("input.tiff")
    output_path = Path("output.tiff")
    output_dir = Path("out_dir")

    job_ctx = JobContext(
        input_path=input_path,
        output_path=output_path,
        output_dir=output_dir,
        sigma=2,
        iterations=15,
        quality=95,
    )

    # JobContext should be mutable
    job_ctx.sigma = 3
    job_ctx.iterations = 20
    job_ctx.quality = 85

    assert job_ctx.sigma == 3
    assert job_ctx.iterations == 20
    assert job_ctx.quality == 85
    assert job_ctx.input_path == input_path
    assert job_ctx.output_path == output_path
    assert job_ctx.output_dir == output_dir


def test_jobcontext_defaults():
    """Test JobContext default values."""
    input_path = Path("input.tiff")
    output_path = Path("output.tiff")
    output_dir = Path("out_dir")

    job_ctx = JobContext(
        input_path=input_path, output_path=output_path, output_dir=output_dir
    )

    # Check default values
    assert job_ctx.sigma == 1
    assert job_ctx.iterations == 10
    assert job_ctx.quality == 90
    assert job_ctx.intermediate_path is None


def test_environment_device_selection():
    """Test Environment device field for future denoiser device configuration."""
    tools = Tools(darktable=Path("dt"), gmic=Path("gmic"))
    config = {"test": "value"}

    # Test default device (None)
    env_default = Config(tools=tools, config=config)
    assert env_default.device is None

    # Test CPU device
    env_cpu = Config(tools=tools, config=config, device="cpu")
    assert env_cpu.device == "cpu"

    # Test CUDA device
    env_cuda = Config(tools=tools, config=config, device="cuda")
    assert env_cuda.device == "cuda"

    # Test MPS device
    env_mps = Config(tools=tools, config=config, device="mps")
    assert env_mps.device == "mps"


def test_type_safety():
    """Test that JobContext provides type safety benefits."""
    # Required fields must be provided
    with pytest.raises(TypeError):
        JobContext()  # Missing required arguments

    with pytest.raises(TypeError):
        JobContext(input_path=Path("input.tiff"))  # Missing output_path and output_dir

    # All required fields provided should work
    job_ctx = JobContext(
        input_path=Path("input.tiff"),
        output_path=Path("output.tiff"),
        output_dir=Path("output"),
    )
    assert job_ctx is not None


def test_config_access_in_environment():
    """Test that Environment provides clean access to configuration."""
    config = {
        "models": {
            "nind_generator_650.pt": {"path": "models/brummer2019/generator_650.pt"}
        },
        "operations": {
            "first_stage": ["exposure", "demosaic"],
            "second_stage": ["exposure", "colorout"],
        },
    }

    tools = Tools(darktable=Path("dt"), gmic=Path("gmic"))
    cfg = Config(tools=tools, config=config, verbose=False)

    # Test config access
    assert (
        cfg.config["models"]["nind_generator_650.pt"]["path"]
        == "models/brummer2019/generator_650.pt"
    )
    assert "first_stage" in cfg.config["operations"]
    assert "exposure" in cfg.config["operations"]["first_stage"]
