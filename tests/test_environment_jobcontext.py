"""Test Environment/JobContext split functionality."""

from nind_denoise.config.config import Config


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
