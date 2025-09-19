"""Directory-based image denoising and evaluation module.

This module provides functionality to denoise entire directories of images using trained
neural network models and evaluate their performance against ground truth images. It
supports batch processing of image datasets and comprehensive quality metrics computation.

The module implements a complete evaluation pipeline that:
1. Processes directories containing noisy images and their corresponding ground truth
2. Applies trained denoising models using image cropping and reassembly techniques
3. Computes quality metrics including SSIM and MS-SSIM scores
4. Stores evaluation results for model comparison and analysis

Key Features:
    * Batch denoising of image directories
    * Automatic ground truth detection (lowest-ISO images)
    * SSIM-based quality evaluation
    * Support for multiple network architectures (UtNet, UNet, etc.)
    * Configurable image cropping and padding strategies
    * Result logging and JSON output generation

Example:
    Basic usage for model evaluation:

    ```bash
    python denoise_dir.py --model_path models/generator_20.pt --network UtNet --cs 552 --ucs 540
    ```

    Denoise a specific directory:

    ```bash
    python denoise_dir.py --noisy_dir datasets/test/ds_fs --model_path models/generator.pt --network UNet
    ```

Note:
    The module expects directory names to follow the format [CROPSIZE]_[USEFULCROPSIZE]
    for proper processing configuration. Ground truth images are automatically identified
    as the lowest ISO images in each directory.
"""

import argparse
import os
import sys

import configargparse
import loss
import torch
import torchvision
from PIL import Image

import nind_denoise.libs.brummer2019.dataset
import nind_denoise.pipeline.denoise.brummer2019
import nind_denoise.train.nn_train
from nind_denoise.pipeline.denoise.brummer2019 import Model

sys.path.append("..")
from nind_denoise import nn_common
from nind_denoise import dataset_torch_3
from nind_denoise.libs.common import json_saver, pt_helpers, utilities


def parse_args():
    """Parse command-line arguments for directory-based image denoising.

    Configures and parses command-line arguments for the directory denoising script,
    supporting both direct directory processing and test reserve-based evaluation.
    The function sets up argument parsing with YAML configuration file support.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - noisy_dir (str): Path to directory containing noisy images to denoise
            - g_network (str): Neural network architecture (UtNet, UNet, etc.)
            - model_path (str): Path to pretrained model file (.pth or .pt)
            - model_parameters (str): Comma-separated model parameters
            - result_dir (str): Output directory for denoised images and results
            - no_scoring (bool): Whether to skip SSIM/MSE evaluation
            - cs (str): Crop size for image processing
            - ucs (str): Useful crop size (area used for computation)
            - skip_existing (bool): Whether to skip already processed files
            - whole_image (bool): Process entire image without cropping
            - pad (int): Padding amount per side for whole image processing
            - max_subpixels (int): Maximum pixel count threshold
            - test_reserve (list): Image sets reserved for testing evaluation
            - orig_data (str): Location of original training data
            - models_dpath (str): Directory containing all model files

    Note:
        The parser loads default configuration from YAML files and supports
        both direct directory denoising and test reserve-based model evaluation.
        Directory names should follow [CROPSIZE]_[USEFULCROPSIZE] format.
    """
    parser = configargparse.ArgumentParser(
        description=__doc__,
        default_config_files=[nind_denoise.train.nn_train.COMMON_CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--noisy_dir",
        type=str,
        help="directory of test dataset (or any directory containing images to be denoised), must end with [CROPSIZE]_[USEFULCROPSIZE]",
    )
    parser.add_argument(
        "--g_network",
        "--network",
        type=str,
        help="Generator network architecture (typically UtNet or UNet)",
    )
    parser.add_argument(
        "--model_path",
        "--model_fpath",
        help="Generator pretrained model path (.pth for model, .pt for dictionary)",
    )
    parser.add_argument(
        "--model_parameters",
        default="",
        type=str,
        help='Model parameters with format "parameter1=value1,parameter2=value2"',
    )
    parser.add_argument(
        "--result_dir",
        default="../../results/NIND/test",
        type=str,
        help='directory where results are saved. Can also be set to "make_subdirs" to make a denoised/<model_directory_name> subdirectory',
    )
    parser.add_argument(
        "--no_scoring",
        action="store_true",
        help="Generate SSIM score and MSE loss unless this is set",
    )
    parser.add_argument(
        "--cs",
        type=str,
        help="Crop size in pixels - size of image patches extracted for processing",
    )
    parser.add_argument(
        "--ucs",
        type=str,
        help="Useful crop size in pixels - central area used for actual denoising computation, must be smaller than cs to account for boundary effects",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing of images that already have denoised output files",
    )
    parser.add_argument(
        "--whole_image",
        action="store_true",
        help="Process entire image without cropping (ignores cs and ucs parameters)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        help="Padding amount in pixels applied to each side when using whole_image mode (defaults to (cs-ucs)/2 when cropping)",
    )
    parser.add_argument("--max_subpixels", type=int, help="Max number of pixels, otherwise abort.")
    parser.add_argument(
        "--test_reserve",
        nargs="*",
        help="Space separated list of image sets reserved for testing, or yaml file path containing a list. Can be used like in training in place of noisy_dir argument.",
    )
    parser.add_argument(
        "--orig_data",
        help="Location of the originally downloaded train data (before cropping); used with test_reserve",
    )
    parser.add_argument("--models_dpath", help="Directory where all models are saved")
    args = parser.parse_args()
    return args


def gen_score(noisy_dir, gt_dir="../../datasets/test/NIND/ds_fs"):
    with torch.accelerator.current_accelerator(
        check_available=True
    ) as device:  # will return None if no gpu, should all still work
        MSE = torch.nn.MSELoss().to(device)
        SSIM = pytorch_ssim.SSIM().to(device)
        with open(os.path.join(noisy_dir, "res.txt"), "w") as f:
            for noisy_img in files(noisy_dir):
                gtpath = find_gt_path(noisy_img, gt_dir)
                noisy_path = os.path.join(noisy_dir, noisy_img)
                gtimg = totensor(Image.open(gtpath)).to(device)
                noisyimg = totensor(Image.open(noisy_path)).to(device)
                gtimg = gtimg.reshape([1] + list(gtimg.shape))
                noisyimg = noisyimg.reshape([1] + list(noisyimg.shape))
                MSELoss = MSE(gtimg, noisyimg).item()
                SSIMScore = SSIM(gtimg, noisyimg).item()
                res = noisy_img + "," + str(SSIMScore) + "," + str(MSELoss)
                print(res)
                f.write(res + "\n")


if __name__ == "__main__":
    """Main execution workflow for directory-based image denoising and evaluation.

    This workflow implements a complete pipeline for:
    1. Processing command-line arguments and model configuration
    2. Setting up input directories and output paths
    3. Batch denoising of images using trained neural networks
    4. Computing quality metrics (SSIM, MS-SSIM) against ground truth
    5. Logging results in JSON format for analysis
    """

    # Parse command-line arguments and validate required parameters
    args = parse_args()
    assert args.model_path is not None, "Model path is required for denoising"

    # Auto-detect network architecture and crop size parameters from model or arguments
    nind_denoise.pipeline.denoise.brummer2019.autodetect_network_cs_ucs(args)

    # Resolve complete model path, ensuring generator model is properly located
    model_path = Model.complete_path(
        args.model_path, keyword="generator", models_dpath=args.models_dpath
    )
    # PHASE 1: Directory Setup and Input Configuration
    # Determine processing mode: direct directory or test reserve evaluation
    if args.noisy_dir is not None:
        # Direct directory processing mode - denoise all images in specified directory
        sets_to_denoise = os.listdir(args.noisy_dir)

        # Check if we're processing a flat directory (containing images directly)
        # vs. a directory containing subdirectories of image sets
        if os.path.isfile(os.path.join(args.noisy_dir, sets_to_denoise[0])):
            sets_to_denoise = ["."]  # Process current directory containing images

        # Configure output directory based on result_dir setting
        if args.result_dir == "make_subdirs":
            # Create structured subdirectory: parent/denoised/model_name/dataset_name
            denoised_save_dir = os.path.join(
                args.noisy_dir,
                "..",
                "denoised",
                utilities.get_file_dname(args.model_path),
                utilities.get_leaf(args.noisy_dir),
            )
            os.makedirs(denoised_save_dir, exist_ok=True)
        else:
            # Use specified result directory with model-specific subdirectory
            denoised_save_dir = os.path.join(args.result_dir, model_path.split("/")[-2])
        test_set_str = utilities.get_root(args.noisy_dir)
    else:
        # Test reserve evaluation mode - use predefined test sets for model evaluation
        sets_to_denoise = nind_denoise.train.nn_train.get_test_reserve_list(args.test_reserve)
        args.noisy_dir = args.orig_data  # Point to original training data location

        # Generate test set identifier for result organization
        if len(args.test_reserve) == 1 and os.path.isfile(args.test_reserve[0]):
            test_set_str = utilities.get_leaf(args.test_reserve[0])
        else:
            test_set_str = str(args.test_reserve)

        # Create output directory structure: model_root/test/model_name/test_set
        denoised_save_dir = os.path.join(
            utilities.get_root(args.model_path),
            "test",
            utilities.get_leaf(args.model_path),
            test_set_str,
        )

    # Ensure output directory exists
    os.makedirs(denoised_save_dir, exist_ok=True)

    # PHASE 2: Batch Image Denoising and Quality Evaluation
    # Initialize containers for quality metrics aggregation
    losses_per_set = list()  # Store average metrics per image set

    # Process each image set in the dataset
    for aset in sets_to_denoise:
        losses_per_img = list()  # Store metrics for each image in current set
        aset_indir = os.path.join(args.noisy_dir, aset)

        # Identify ground truth image (lowest-ISO baseline for quality comparison)
        baseline_fpath = nind_denoise.libs.brummer2019.dataset.get_baseline_fpath(aset_indir)
        images_fn = os.listdir(aset_indir)

        # Process each noisy image in the current set
        for animg in images_fn:
            inimg_path = os.path.join(aset_indir, animg)

            # Skip ground truth image (it's used only for comparison)
            if baseline_fpath == inimg_path:
                continue

            # Configure output path and ensure proper file extension
            outimg_path = os.path.join(denoised_save_dir, animg)
            if outimg_path.endswith("jpg"):
                outimg_path = outimg_path + ".tif"  # Convert to lossless format

            # Apply denoising if output doesn't exist or skip_existing is False
            if not (os.path.isfile(outimg_path) and args.skip_existing):
                # Configure denoising parameters using SimpleNamespace for argument passing
                from types import SimpleNamespace as _NS

                di_args = _NS(
                    cs=int(args.cs) if args.cs is not None else None,  # Crop size
                    ucs=(int(args.ucs) if args.ucs is not None else None),  # Useful crop size
                    overlap=6,  # Overlap between crops to avoid boundary artifacts
                    input=inimg_path,
                    output=outimg_path,
                    batch_size=1,  # Process one image at a time
                    debug=False,
                    exif_method="piexif",  # EXIF data preservation method
                    g_network=args.g_network,  # Neural network architecture
                    model_path=model_path,
                    model_parameters=args.model_parameters,
                    max_subpixels=(
                        int(args.max_subpixels) if args.max_subpixels is not None else None
                    ),
                    whole_image=bool(args.whole_image),  # Skip cropping if True
                    pad=(128 if args.whole_image else args.pad),  # Padding for whole image mode
                    models_dpath=args.models_dpath,
                )
                # Execute denoising using Brummer2019 pipeline
                nind_denoise.pipeline.denoise.brummer2019.run_from_args(di_args)

            # PHASE 3: Quality Metrics Computation
            # Compute SSIM and MS-SSIM scores against ground truth baseline
            cur_losses = pt_helpers.get_losses(baseline_fpath, outimg_path)
            print(f"in: {inimg_path}, out: {outimg_path}, clean: {baseline_fpath}")
            print(cur_losses)
            losses_per_img.append(cur_losses)

        # Aggregate metrics for current image set (average across all images)
        losses_per_set.append(utilities.avg_listofdicts(losses_per_img))

    # Compute overall dataset metrics (average across all image sets)
    losses_per_set = utilities.avg_listofdicts(losses_per_set)

    # Display overall evaluation results
    print(losses_per_set)

    # PHASE 4: Results Logging and Persistence
    # Save evaluation results in multiple JSON formats for analysis and comparison

    # Extract epoch number from model filename for proper result organization
    try:
        # Parse epoch from model filename pattern: "generator_<epoch>.<ext>"
        epoch = int(utilities.get_leaf(args.model_path).split("_")[1].split(".")[0])

        # Save results to training results JSON (primary logging)
        # Note: This file may be overwritten by concurrent training processes
        json_res_fpath = os.path.join(utilities.get_root(args.model_path), "trainres.json")
        jsonsaver = json_saver.JSONSaver(json_res_fpath, step_type="epoch")
        jsonsaver.add_res(step=epoch, res=losses_per_set, key_prefix="test_")
    except ValueError as e:
        print(f"Cannot determine epoch from model_path {args.model_path} ({e})")
        epoch = None
    except FileNotFoundError:
        print(f"Model results json file not found ({json_res_fpath})")

    try:
        # Create dedicated test results backup JSON (more reliable for analysis)
        json_res_fpath = os.path.join(utilities.get_root(args.model_path), "testres.json")
        jsonsaver = json_saver.JSONSaver(json_res_fpath, step_type="epoch")
        jsonsaver.add_res(step=epoch, res=losses_per_set, key_prefix="test_")
    except TypeError as e:
        print(f"something is wrong with the test jsonsaver ({e})")
        # Fallback: Direct JSON dump if JSONSaver fails
        print(f"results will be dumped to {json_res_fpath}.")
        utilities.dict_to_json(losses_per_set, json_res_fpath)

    # Legacy scoring system (marked as obsolete but maintained for compatibility)
    if not args.no_scoring:
        gen_score(denoised_save_dir, args.noisy_dir)
totensor = torchvision.transforms.ToTensor()


def find_gt_path(denoised_fn, gt_dir):
    dsname, setdir = denoised_fn.split("_")[0:2]
    setfiles = os.listdir(os.path.join(gt_dir, setdir))
    ext = setfiles[0].split(".")[-1]
    isos = [fn.split("_")[2][:-4] for fn in setfiles]
    baseiso = sortISOs(isos)[0][0]
    baseiso_fn = dsname + "_" + setdir + "_" + baseiso + "." + ext
    gt_fpath = os.path.join(gt_dir, setdir, baseiso_fn)
    return gt_fpath


def files(path):
    for fn in os.listdir(path):
        if os.path.isfile(os.path.join(path, fn)) and fn != "res.txt":
            yield fn


parser = argparse.ArgumentParser(description="Get SSIM score and MSE loss from test images")
