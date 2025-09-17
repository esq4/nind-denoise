"""Test Environment/JobContext split functionality."""

<<<<<<< HEAD
=======
import dataclasses
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
from pathlib import Path

import pytest

<<<<<<< HEAD
=======
from nind_denoise.config import Tools
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
from nind_denoise.config.config import Config
from nind_denoise.pipeline.base import JobContext


<<<<<<< HEAD
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
=======
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
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882


def test_jobcontext_mutability():
    """Test that JobContext is mutable and has typed fields."""
    input_path = Path("input.tiff")
    output_path = Path("output.tiff")
<<<<<<< HEAD
=======
    output_dir = Path("out_dir")
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882

    job_ctx = JobContext(
        input_path=input_path,
        output_path=output_path,
<<<<<<< HEAD
=======
        output_dir=output_dir,
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
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
<<<<<<< HEAD
=======
    assert job_ctx.output_dir == output_dir
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882


def test_jobcontext_defaults():
    """Test JobContext default values."""
    input_path = Path("input.tiff")
    output_path = Path("output.tiff")
<<<<<<< HEAD

    job_ctx = JobContext(input_path=input_path, output_path=output_path)
=======
    output_dir = Path("out_dir")

    job_ctx = JobContext(
        input_path=input_path, output_path=output_path, output_dir=output_dir
    )
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882

    # Check default values
    assert job_ctx.sigma == 1
    assert job_ctx.iterations == 10
    assert job_ctx.quality == 90
    assert job_ctx.intermediate_path is None


<<<<<<< HEAD
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
=======
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
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882


def test_type_safety():
    """Test that JobContext provides type safety benefits."""
    # Required fields must be provided
    with pytest.raises(TypeError):
        JobContext()  # Missing required arguments

    with pytest.raises(TypeError):
<<<<<<< HEAD
        JobContext(input_path=Path("input.tiff"))  # Missing output_path
=======
        JobContext(input_path=Path("input.tiff"))  # Missing output_path and output_dir
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882

    # All required fields provided should work
    job_ctx = JobContext(
        input_path=Path("input.tiff"),
        output_path=Path("output.tiff"),
<<<<<<< HEAD
=======
        output_dir=Path("output"),
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
    )
    assert job_ctx is not None


<<<<<<< HEAD
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
    assert (
        cfg.models["nind_generator_650.pt"]["path"]
        == "models/brummer2019/generator_650.pt"
    )
    assert "first_stage" in cfg.operations["operations"]
    assert "exposure" in cfg.operations["operations"]["first_stage"]
=======
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
>>>>>>> a5fd5d04ba398e54626a0e75a9f92231aba11882
