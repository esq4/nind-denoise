"""Neural network training submodule for image denoising.

This submodule provides comprehensive functionality for training deep learning models
to denoise images using the NIND (Natural Image Noise Dataset) or custom datasets.
The training system supports various neural network architectures including UNet,
UtNet, and custom models with configurable parameters and training strategies.

Key Components:
    dataset: Dataset loading and preprocessing utilities for training and validation.
        Handles image cropping, noise augmentation, and batch preparation.

    nn_train: Main training orchestration module containing the complete training
        pipeline including model initialization, loss computation, optimization,
        and evaluation workflows.

Features:
    * Multiple neural network architectures (UNet, UtNet, custom models)
    * Configurable training parameters via YAML configuration files
    * Support for clean-noisy image pair datasets
    * Advanced data augmentation including compression and artificial noise
    * Multi-GPU training capabilities
    * Comprehensive evaluation metrics (SSIM, MS-SSIM, PSNR)
    * Model checkpointing and resume functionality
    * Test set reservation for proper evaluation
    * Integration with discriminator networks for GAN-based training

Usage:
    Basic training workflow:

    ```python
    from nind_denoise.train import nn_train, dataset

    # Configure training parameters
    args = nn_train.parse_args()

    # Initialize dataset
    train_dataset = dataset.DenoisingDataset(args)

    # Start training
    nn_train.main(args)
    ```

    Command-line training:

    ```bash
    python -m nind_denoise.train.nn_train --config configs/train_conf_unet.yaml
    ```

Configuration:
    Training behavior is controlled through YAML configuration files that specify:
    - Network architecture and parameters
    - Dataset paths and preprocessing options
    - Training hyperparameters (learning rate, batch size, epochs)
    - Augmentation settings
    - Evaluation and checkpoint intervals

Note:
    The training system expects datasets to follow specific directory structures
    and naming conventions. Refer to the dataset module documentation for
    detailed requirements on data organization and preprocessing.

See Also:
    nind_denoise.pipeline.denoise: For inference and model deployment
    nind_denoise.libs.brummer2019: For core network implementations
"""

from . import dataset, nn_train

__all__ = ["dataset", "nn_train"]
