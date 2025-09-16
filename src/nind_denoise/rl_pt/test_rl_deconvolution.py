import torch

from .richardson_lucy_deconvolution import richardson_lucy_gaussian


def test_richardson_lucy_gaussian_basic():
    # Create a simple test image (3x3 with two channels) to ensure at least 2 pixels in each dimension
    image = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                          [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
                          [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]]], dtype=torch.float32)

    sigma = 1.0
    iterations = 5

    result = richardson_lucy_gaussian(image, sigma, iterations)
    assert result.shape == image.shape
    assert (result >= 0).all()
    assert (result <= 1).all()

def test_richardson_lucy_gaussian_with_uint8():
    # Create a simple test image (3x3 with two channels) using uint8
    image = torch.tensor([[[255, 255], [127, 127], [63, 63]],
                          [[191, 191], [127, 127], [63, 63]],
                          [[127, 127], [191, 191], [255, 255]]], dtype=torch.uint8)

    sigma = 0.5
    iterations = 3

    result = richardson_lucy_gaussian(image, sigma, iterations)
    assert result.shape == image.shape
    assert (result >= 0).all()
    assert (result <= 255).all()

def test_richardson_lucy_gaussian_layouts():
    # Test CHW layout
    image_chw = torch.randn((3, 4, 4), dtype=torch.float32)
    sigma = 1.0
    iterations = 5

    result_chw = richardson_lucy_gaussian(image_chw, sigma, iterations)
    assert result_chw.shape == image_chw.shape

    # Test HWC layout
    image_hwc = torch.randn((4, 4, 3), dtype=torch.float32)
    result_hwc = richardson_lucy_gaussian(image_hwc, sigma, iterations)
    assert result_hwc.shape == image_hwc.shape

def test_richardson_lucy_gaussian_zero_iterations():
    # Test with zero iterations (should return the original image)
    image = torch.randn((3, 4, 4), dtype=torch.float32)

    sigma = 1.0
    iterations = 0

    result = richardson_lucy_gaussian(image, sigma, iterations)
    assert torch.allclose(result, image)

if __name__ == "__main__":
    import pytest
    pytest.main(["-v"])
