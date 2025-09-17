"""Test PyTorch RL deblur integration with the new pipeline architecture."""

from nind_denoise.config.config import Config
from nind_denoise.pipeline import Deblur, get_deblur, register_deblur
from nind_denoise.pipeline.base import JobContext
from nind_denoise.pipeline.deblur import RLDeblurPT


def test_pytorch_deblur_registry():
    """Test that PyTorch RL deblur is properly registered in the deblur registry."""
    # Test that pt_rl is available in registry
    deblur_class = get_deblur("pt_rl")
    assert deblur_class == RLDeblurPT

    # Test that we can instantiate the class
    deblur_instance = deblur_class()
    assert isinstance(deblur_instance, Deblur)
    assert isinstance(deblur_instance, RLDeblurPT)


def test_pytorch_deblur_describe():
    """Test that PyTorch RL deblur has proper description."""
    deblur_instance = RLDeblurPT()
    description = deblur_instance.describe()
    assert "PyTorch" in description
    assert "Deblur" in description


def test_pytorch_deblur_verify_without_context():
    """Test that PyTorch RL deblur verify method works without context."""
    deblur_instance = RLDeblurPT()
    # Should not raise an exception when ctx is None
    deblur_instance.verify(None)


def test_deblur_registry_extensibility():
    """Test that the deblur registry is extensible for custom implementations."""

    class CustomDeblur(Deblur):
        def describe(self) -> str:
            return "Custom Deblur"

        def execute_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
            pass

        def verify_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
            pass

    # Register custom deblur
    register_deblur("custom", CustomDeblur)

    # Verify it can be retrieved
    custom_class = get_deblur("custom")
    assert custom_class == CustomDeblur

    # Test instantiation
    custom_instance = custom_class()
    assert isinstance(custom_instance, Deblur)
    assert custom_instance.describe() == "Custom Deblur"


def test_all_registered_deblurs():
    """Test that all expected deblur implementations are registered."""
    # Test gmic deblur
    gmic_class = get_deblur("gmic")
    gmic_instance = gmic_class()
    assert "gmic" in gmic_instance.describe().lower()

    # Test pytorch deblur
    pt_class = get_deblur("pt_rl")
    pt_instance = pt_class()
    assert "pytorch" in pt_instance.describe().lower()
