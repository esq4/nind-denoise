import numpy as np
import pytest
import torch

from nind_denoise.pipeline.orchestrator import filepath_input_connector


class TestFilepathInputConnector:

    @pytest.fixture
    def setup_class(self):
        class DummyClass:
            image = None

            def input_connector(self, *args, **kwargs):
                return self.image

        return DummyClass()

    def test_filepath_input_connector_decorator(self, tmp_path, setup_class):
        img_path = tmp_path / "test_image.npy"
        np_img = np.random.rand(100, 100, 3).astype(np.float32)
        np.save(img_path, np_img)

        # Apply the decorator
        @filepath_input_connector
        class DecoratedClass(DummyClass):
            pass

        decorated_class_instance = DecoratedClass()
        decorated_class_instance.image = img_path

        # Call input_connector
        result_tensor = decorated_class_instance.input_connector()

        expected_tensor = torch.from_numpy(np_img)

        assert torch.equal(
            result_tensor, expected_tensor
        ), f"Expected {expected_tensor}, but got {result_tensor}"

    def test_filepath_input_connector_original_method(self, tmp_path, setup_class):
        img_path = tmp_path / "test_image.npy"
        np_img = np.random.rand(100, 100, 3).astype(np.float32)
        np.save(img_path, np_img)

        # Apply the decorator
        @filepath_input_connector
        class DecoratedClass(DummyClass):
            pass

        decorated_class_instance = DecoratedClass()
        decorated_class_instance.image = img_path

        # Call input_connector and check if original method is called
        result_tensor = decorated_class_instance.input_connector()

        expected_tensor = torch.from_numpy(np_img)

        assert isinstance(
            result_tensor, torch.Tensor
        ), f"Expected type {torch.Tensor}, but got {type(result_tensor)}"
        assert torch.equal(
            result_tensor, expected_tensor
        ), f"Expected {expected_tensor}, but got {result_tensor}"
