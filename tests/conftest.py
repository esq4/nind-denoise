# tests/conftest.py

import pytest


@pytest.fixture
def denoise_module():
    import denoise

    return denoise
