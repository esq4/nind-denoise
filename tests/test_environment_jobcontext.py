"""Test Environment/JobContext split functionality."""

import dataclasses
from pathlib import Path

import pytest

from nind_denoise.config import Tools
from nind_denoise.pipeline.base import Context, Environment, JobContext


def test_environment_immutability():
    """Test that Environment is immutable (frozen dataclass)."""
    tools = Tools(darktable=Path("dt"), gmic=Path("gmic"))
    config = {"test": "value"}

    env = Environment(tools=tools, config=config, verbose=True, device="cpu")

    # Environment should be frozen - attempting to modify should raise error
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.verbose = False

    with pytest.raises(dataclasses.FrozenInstanceError):
        env.device = "cuda"

    # Verify values are correctly set
    assert env.tools == tools
    assert env.config == config
    assert env.verbose is True
    assert env.device == "cpu"


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
    env_default = Environment(tools=tools, config=config)
    assert env_default.device is None

    # Test CPU device
    env_cpu = Environment(tools=tools, config=config, device="cpu")
    assert env_cpu.device == "cpu"

    # Test CUDA device
    env_cuda = Environment(tools=tools, config=config, device="cuda")
    assert env_cuda.device == "cuda"

    # Test MPS device
    env_mps = Environment(tools=tools, config=config, device="mps")
    assert env_mps.device == "mps"


def test_backward_compatibility_conversion():
    """Test that Operation base class converts new context types to legacy Context."""
    from nind_denoise.pipeline.base import Operation

    # Create a mock operation to test the conversion
    class TestOperation(Operation):
        def describe(self) -> str:
            return "Test Operation"

        def execute(self, ctx: Context) -> None:
            # Store the converted context for inspection
            self.last_ctx = ctx

        def verify(self, ctx: Context = None) -> None:
            pass

    tools = Tools(darktable=Path("dt"), gmic=Path("gmic"))
    env = Environment(
        tools=tools, config={"test": "config"}, verbose=True, device="cpu"
    )

    job_ctx = JobContext(
        input_path=Path("input.tiff"),
        output_path=Path("output.jpg"),
        output_dir=Path("output"),
        sigma=2,
        iterations=15,
        quality=85,
        intermediate_path=Path("intermediate.tiff"),
    )

    op = TestOperation()
    op.execute_with_env(env, job_ctx)

    # Check that legacy Context was created correctly
    legacy_ctx = op.last_ctx
    assert legacy_ctx.inpath == job_ctx.input_path
    assert legacy_ctx.outpath == job_ctx.output_path
    assert legacy_ctx.output_dir == job_ctx.output_dir
    assert legacy_ctx.output_filepath == job_ctx.intermediate_path
    assert legacy_ctx.sigma == job_ctx.sigma
    assert legacy_ctx.iteration == str(job_ctx.iterations)
    assert legacy_ctx.quality == str(job_ctx.quality)
    assert legacy_ctx.cmd_gmic == str(env.tools.gmic)
    assert legacy_ctx.verbose == env.verbose


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


def test_legacy_context_compatibility():
    """Test that legacy Context still works as before."""
    # Legacy Context should still allow all Optional fields
    ctx = Context()
    assert ctx.inpath is None
    assert ctx.outpath is None
    assert ctx.verbose is False

    # Legacy Context should be mutable
    ctx.outpath = Path("test.jpg")
    ctx.verbose = True
    assert ctx.outpath == Path("test.jpg")
    assert ctx.verbose is True


def test_config_access_in_environment():
    """Test that Environment provides clean access to configuration."""
    config = {
        "models": {"nind_generator_650.pt": {"path": "models/nind/generator_650.pt"}},
        "operations": {
            "first_stage": ["exposure", "demosaic"],
            "second_stage": ["exposure", "colorout"],
        },
    }

    tools = Tools(darktable=Path("dt"), gmic=Path("gmic"))
    env = Environment(tools=tools, config=config, verbose=False)

    # Test config access
    assert (
        env.config["models"]["nind_generator_650.pt"]["path"]
        == "models/nind/generator_650.pt"
    )
    assert "first_stage" in env.config["operations"]
    assert "exposure" in env.config["operations"]["first_stage"]
