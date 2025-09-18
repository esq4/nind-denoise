"""Test Environment/JobContext split functionality."""

from pathlib import Path

import pytest

from nind_denoise.config.config import Config
from nind_denoise.pipeline.base import JobContext


def test_environment_immutability(tmp_path, create_test_config):
    """Test that Config provides access to tools and configuration."""
    config_file = create_test_config(tmp_path)
    cfg = Config(path=config_file, verbose=True)

    # Config should provide access to tools and models
    assert cfg.tools is not None
    assert cfg.models is not None
    assert cfg.verbose is True

    # Test that we can access the tools
    tools = cfg.tools
    assert tools is not None


def test_jobcontext_mutability():
    """Test that JobContext is mutable and has typed fields."""
    input_path = Path("input.tiff")
    output_path = Path("output.tiff")

    job_ctx = JobContext(
        input_path=input_path,
        output_path=output_path,
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


def test_jobcontext_defaults():
    """Test JobContext default values."""
    input_path = Path("input.tiff")
    output_path = Path("output.tiff")
    job_ctx = JobContext(input_path=input_path, output_path=output_path)

    # Check default values
    assert job_ctx.sigma == 1
    assert job_ctx.iterations == 10
    assert job_ctx.quality == 90
    assert job_ctx.intermediate_path is None


def test_environment_device_selection(tmp_path, create_test_config):
    """Test Config provides tools and operations access."""
    config_file = create_test_config(tmp_path)

    # Test config creation and access
    cfg = Config(path=config_file, verbose=False)
    assert cfg.tools is not None
    assert cfg.operations is not None
    assert cfg.models is not None

    # Test verbose mode
    cfg_verbose = Config(path=config_file, verbose=True)
    assert cfg_verbose.verbose is True


def test_type_safety():
    """Test that JobContext provides type safety benefits."""
    # Required fields must be provided
    with pytest.raises(TypeError):
        JobContext(input_path=Path("input.tiff"))  # Missing output_path

    # All required fields provided should work
    job_ctx = JobContext(
        input_path=Path("input.tiff"),
        output_path=Path("output.tiff"),
    )
    assert job_ctx is not None


def test_config_access_in_environment(tmp_path, create_test_config):
    """Test that Config provides clean access to configuration."""
    models = {
        "nind_generator_650.pt": {
            "path": "models/brummer2019/generator_650.pt",
            "default": True,
        }
    }
    operations = {
        "operations": {
            "first_stage": ["exposure", "demosaic"],
            "second_stage": ["exposure", "colorout"],
        }
    }

    config_file = create_test_config(tmp_path, models=models, operations=operations)
    cfg = Config(path=config_file, verbose=False)

    # Test config access
    assert cfg.models["nind_generator_650.pt"]["path"] == "models/brummer2019/generator_650.pt"
    assert "first_stage" in cfg.operations["operations"]
    assert "exposure" in cfg.operations["operations"]["first_stage"]


# TODO: need to test default model
