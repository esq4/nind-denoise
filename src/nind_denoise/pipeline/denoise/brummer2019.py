from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import configargparse
import cv2
import exiv2
import numpy as np
import piexif
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from nind_denoise.config.config import Config
from nind_denoise.libs.common import np_imgops, pt_helpers, utilities
from nind_denoise.pipeline.base import (
    DenoiseOperation,
    JobContext,
    StageError,
)

# Default tile sizes for different neural network architectures
# CS = Crop Size (total tile size including padding)
# UCS = Useful Crop Size (actual useful content size within the tile)

CS_UNET, UCS_UNET = 440, 320  # UNet architecture tile parameters
CS_UTNET, UCS_UTNET = 504, 480  # UtNet architecture tile parameters
CS_UNK, UCS_UNK = 512, 448  # Unknown architecture default tile parameters


@dataclass
class DenoiseOptions:
    """Configuration options for the denoising stage.

    This dataclass contains all the necessary parameters for configuring
    the denoising operation, including model location, compute device,
    and tiling parameters.

    Attributes:
        model_path (Path): Filesystem path to the pretrained denoising model file.
            Should point to a .pt or .pth file containing model weights.
        device (Optional[str]): Target computation device for inference.
            Can be "cpu", "cuda", "mps", or None for automatic selection.
            Defaults to None.
        overlap (int): Overlap size in pixels between adjacent image tiles.
            Higher values reduce grid artifacts but increase computation time.
            Defaults to 6.
        batch_size (int): Number of image tiles to process simultaneously.
            Higher values can improve throughput on capable hardware.
            Currently only batch_size=1 is fully supported. Defaults to 1.
    """

    model_path: Path
    device: Optional[str] = None  # e.g., "cpu", "cuda", "mps"
    overlap: int = 6
    batch_size: int = 1


class DenoiseStage(DenoiseOperation):
    """Denoising pipeline stage implementing the Brummer et al. (2019) method.

    This class orchestrates the neural network-based image denoising process,
    including model loading, image tiling, inference, and reconstruction.

    Attributes:
        s1_tif (Path): Input TIFF image path (legacy attribute).
        s1_denoised_tif (Path): Output denoised TIFF path (legacy attribute).
        opts (DenoiseOptions): Configuration options for the denoising process.
    """

    def __init__(self, s1_tif: Path, s1_denoised_tif: Path, opts: DenoiseOptions):
        """Initialize the denoising stage.

        Args:
            s1_tif (Path): Input TIFF image path (legacy parameter).
            s1_denoised_tif (Path): Output denoised TIFF path (legacy parameter).
            opts (DenoiseOptions): Configuration options for denoising.
        """
        self.s1_tif = s1_tif
        self.s1_denoised_tif = s1_denoised_tif
        self.opts = opts

    def describe(self) -> str:
        """Return a human-readable description of this pipeline stage.

        Returns:
            str: Stage description string.
        """
        return "Denoise"

    def execute_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
        """Execute denoise stage with type-safe Environment + JobContext.

        This method orchestrates the complete denoising pipeline, from loading
        configuration to producing the final denoised image.

        Args:
            cfg (Config): Environment containing global configuration.
            job_ctx (JobContext): Job-specific context with input/output paths.

        Raises:
            StageError: If denoising fails or required resources are unavailable.
        """
        # Input comes from JobContext
        s1_tif = job_ctx.input_path
        s1_denoised_tif = job_ctx.output_path

        # Model path comes from Environment config
        model_config = cfg.config.get("models", {}).get("nind_generator_650.pt", {})
        model_path = Path(
            model_config.get("path", "models/brummer2019/generator_650.pt")
        )

        # Create DenoiseOptions with model path
        opts = DenoiseOptions(
            model_path=model_path,
            overlap=6,  # Could be configurable via JobContext in future
            batch_size=1,  # Could be configurable via JobContext in future
        )

        # Prepare output file (ensure directory exists and unlink stale file)
        self._prepare_output_file(s1_denoised_tif)

        self._denoiser(opts, s1_denoised_tif, s1_tif)

    def _denoiser(self, opts: DenoiseOptions, s1_denoised_tif: Path, s1_tif: Path):
        """Execute the core denoising algorithm using neural networks.

        This method prepares arguments and delegates to the main denoising
        function that handles model loading, image tiling, inference, and
        reconstruction.

        Args:
            opts (DenoiseOptions): Configuration options for denoising.
            s1_denoised_tif (Path): Output path for the denoised image.
            s1_tif (Path): Input path of the noisy image to denoise.
        """
        # Import locally to avoid import-time side effects
        from types import SimpleNamespace as NS

        args = NS(
            cs=None,
            ucs=None,
            overlap=opts.overlap,
            input=str(s1_tif),
            output=str(s1_denoised_tif),
            batch_size=opts.batch_size,
            debug=False,
            exif_method="piexif",
            g_network="UtNet",
            model_path=str(opts.model_path),
            model_parameters=None,
            max_subpixels=None,
            whole_image=False,
            pad=None,
            models_dpath=None,
        )
        run_from_args(args)

    def verify_with_env(self, cfg: Config, job_ctx: JobContext) -> None:
        """Verify that the denoising stage completed successfully.

        Checks that the expected output file was created and exists on disk.
        This verification step ensures the denoising pipeline stage completed
        without silent failures.

        Args:
            cfg (Config): Environment containing global configuration (unused).
            job_ctx (JobContext): Job-specific context with output path to verify.

        Raises:
            StageError: If the expected denoised image file does not exist.
        """
        s1_denoised_tif = job_ctx.output_path

        if not s1_denoised_tif.exists():
            raise StageError(
                f"Denoise stage expected output missing: {s1_denoised_tif}"
            )


def make_output_fpath(input_fpath, model_fpath):
    """Generate output filepath for denoised image based on input and model paths.

    Creates a standardized output path within the model directory structure,
    combining the input image filename with the model filename. Ensures the
    output directory exists before returning the path.

    Args:
        input_fpath (str): Path to the input image file.
        model_fpath (str): Path to the model file used for denoising.

    Returns:
        str: Complete filepath for the output denoised image in TIFF format,
             located at {model_dir}/test/denoised_images/{img_name}_{model_name}.tif
    """
    model_dpath = utilities.get_root(model_fpath)
    model_fn = utilities.get_leaf(model_fpath)
    img_fn = utilities.get_leaf(input_fpath)
    os.makedirs(os.path.join(model_dpath, "test", "denoised_images"), exist_ok=True)
    return os.path.join(
        model_dpath, "test", "denoised_images", f"{img_fn}_{model_fn}.tif"
    )


def autodetect_network_cs_ucs(args) -> None:
    """Automatically detect network architecture and set appropriate tile sizes.

    Infers the neural network architecture from the model path if not specified,
    then sets optimal crop size (cs) and useful crop size (ucs) parameters
    based on the detected or specified architecture. This ensures proper tiling
    parameters for different network types.

    Args:
        args: Argument namespace containing model configuration. Modified in-place
              to set g_network, cs, and ucs attributes based on detection or defaults.

    Note:
        The function modifies the args object in-place, setting:
        - args.g_network: Network architecture ("UNet", "UtNet", or inferred)
        - args.cs: Crop size (total tile size including padding)
        - args.ucs: Useful crop size (actual content size within tile)

    Raises:
        SystemExit: If network architecture cannot be determined from model path
                   and no explicit network type is provided.
    """
    if args.g_network is None:
        print("network parameter not specified")
        if "unet" in args.model_path.lower():
            args.g_network = "UNet"
        elif "utnet" in args.model_path.lower():
            args.g_network = "UtNet"
        else:
            exit(
                'Could not determine network architecture from path. Please specify a "--network" type (typically UNet or UtNet)'
            )
        print(f"Assuming {args.g_network} from path")
    if args.cs is None or args.ucs is None:
        print("cs and/or ucs not set, using defaults ...")
        if args.g_network == "UNet":
            args.cs = CS_UNET
            args.ucs = UCS_UNET
        elif args.g_network == "UtNet":
            args.cs, args.ucs = CS_UTNET, UCS_UTNET
        else:
            print(
                "Warning: cs and ucs not known for this architecture; values may be sub-optimal"
            )
            args.cs, args.ucs = CS_UNK, UCS_UNK
        print(f"cs={args.cs}, ucs={args.ucs}")


class OneImageDS(Dataset):
    """PyTorch dataset for processing a single image with tiling and padding.

    This dataset class divides a single large image into overlapping tiles
    for processing by neural networks that have memory or input size constraints.
    It handles mirror padding at image boundaries to avoid edge artifacts
    and supports both tiled processing and whole-image processing modes.

    The dataset loads images using OpenCV in 8/16-bit depth and performs
    all operations using NumPy. Images are always returned in CxHxW format
    (channels first) as expected by PyTorch models.

    Attributes:
        inimg (np.ndarray): Loaded input image as float32 array in CxHxW format.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        whole_image (bool): Whether to process entire image at once or use tiling.
        pad (int): Padding amount in pixels for mirror padding at boundaries.
        cs (int): Crop size - total tile size including padding (tiling mode only).
        ucs (int): Useful crop size - actual content size within tile (tiling mode only).
        ol (int): Overlap between adjacent tiles in pixels (tiling mode only).
        iperhl (int): Number of tiles per horizontal line (tiling mode only).
        size (int): Total number of tiles/samples in the dataset.
    """

    def __init__(self, inimg_fpath, cs, ucs, ol, whole_image=False, pad=None):
        """Initialize the single-image dataset.

        Args:
            inimg_fpath (str): Path to the input image file.
            cs (int): Crop size - total tile size including padding.
            ucs (int): Useful crop size - actual content size within each tile.
            ol (int): Overlap between adjacent tiles in pixels.
            whole_image (bool, optional): If True, process entire image at once
                without tiling. Defaults to False.
            pad (int, optional): Padding amount for whole-image mode. If None,
                calculated as (cs-ucs)/2 for tiling mode. Defaults to None.
        """
        self.inimg = np_imgops.img_path_to_np_flt(inimg_fpath)
        self.width, self.height = self.inimg.shape[2], self.inimg.shape[1]
        if whole_image:
            self.pad = pad
            if self.pad is None or self.pad == 0:
                self.pad = 0
                print("OneImageDS: Warning: you should really consider (pad>0)")
            self.whole_image = True
            self.size = 1
        else:
            self.whole_image = False
            self.cs, self.ucs, self.ol = (
                cs,
                ucs,
                ol,
            )  # crop size, useful crop size, overlap
            self.iperhl = math.ceil(
                (self.width - self.ucs) / (self.ucs - self.ol)
            )  # i_per_hline, or crops per line
            self.pad = int((self.cs - self.ucs) / 2)
            ipervl = math.ceil((self.height - self.ucs) / (self.ucs - self.ol))
            self.size = (self.iperhl + 1) * (ipervl + 1)
            if self.pad is None or self.pad == 0:
                self.pad = 0
                print(
                    "OneImageDS: Warning: you should really consider padding (cs>ucs)"
                )

    def __getitem__(self, i):
        """Retrieve a single tile or the entire image with padding.

        For tiled mode, generates the i-th tile with appropriate mirror padding
        at boundaries. For whole-image mode, returns the entire image with
        mirror padding around all edges. All tiles include coordinate information
        for reconstruction.

        Args:
            i (int): Index of the tile to retrieve (0 to len(dataset)-1).

        Returns:
            tuple: A 3-tuple containing:
                - torch.Tensor: Image tile as float32 tensor in CxHxW format.
                - torch.IntTensor: Useful dimensions as (x_start, y_start, x_end, y_end)
                  indicating the region within the tile containing actual image content.
                - torch.IntTensor: Useful start coordinates (x_start, y_start) in the
                  original full-size image coordinate system for reconstruction.

        Note:
            Mirror padding is applied at image boundaries to avoid edge artifacts
            during neural network processing. The useful dimension information
            allows proper reconstruction by indicating which parts of each tile
            contain actual image content versus padding.
        """
        # mirrorring can be improved by including the corners
        if self.whole_image:
            xi = yi = x0 = y0 = 0
            x1, y1 = self.width, self.height
            ret = np.zeros((3, x1 + self.pad * 2, y1 + self.pad * 2), dtype=np.float32)
            crop = self.inimg
            # copy image to the center
            ret[
                :, self.pad : self.height + self.pad, self.pad : self.width + self.pad
            ] = crop
            # mirror sides
            if self.pad:
                # left
                ret[:, self.pad : -self.pad, : self.pad] = np.flip(
                    crop[:, :, : self.pad], axis=2
                )
                # right
                ret[:, self.pad : -self.pad, self.width + self.pad :] = np.flip(
                    crop[:, :, self.width - self.pad :], axis=2
                )
                # top
                ret[:, : self.pad, self.pad : -self.pad] = np.flip(
                    crop[:, : self.pad, :], axis=1
                )
                # bottom
                ret[:, self.height + self.pad :, self.pad : -self.pad] = np.flip(
                    crop[:, self.height - self.pad :, :], axis=1
                )
            usefuldim = (
                self.pad,
                self.pad,
                self.width + self.pad,
                self.height + self.pad,
            )
            usefulstart = self.pad, self.pad
        else:
            # x-y indices (0 to iperhl, 0 to ipervl)
            yi = int(math.ceil((i + 1) / (self.iperhl + 1) - 1))  # line number
            xi = i - yi * (self.iperhl + 1)
            # x-y start-end position on fs image
            x0 = self.ucs * xi - self.ol * xi - self.pad
            x1 = x0 + self.cs
            y0 = self.ucs * yi - self.ol * yi - self.pad
            y1 = y0 + self.cs
            ret = np.ndarray((3, self.cs, self.cs), dtype=np.float32)
            # amount padded to have a cs x cs crop
            x0pad = -min(0, x0)
            x1pad = max(0, x1 - self.width)
            y0pad = -min(0, y0)
            y1pad = max(0, y1 - self.height)
            # determine crop of interest
            crop = self.inimg[:, y0 + y0pad : y1 - y1pad, x0 + x0pad : x1 - x1pad]
            # copy crop to center
            ret[:, y0pad : self.cs - y1pad, x0pad : self.cs - x1pad] = crop
            # mirror stuff:
            # this somewhat suspicious stuff can be tested (visualized crops) with the
            # --debu argument
            if x0pad > 0:  # left
                ret[:, y0pad : self.cs - y1pad, :x0pad] = np.flip(
                    self.inimg[:, y0 + y0pad : y1 - y1pad, x0 + x0pad : x0 + x0pad * 2],
                    axis=2,
                )
                if y0pad > 0:  # top-left
                    ret[:, :y0pad, :x0pad] = np.flip(
                        self.inimg[:, :y0pad, :x0pad], axis=(1, 2)
                    )
                if y1pad > 0:  # bottom-left
                    ret[:, -y1pad:, :x0pad] = np.flip(
                        self.inimg[:, -y1pad:, :x0pad], axis=(1, 2)
                    )
            if x1pad > 0:  # right
                ret[
                    :,
                    y0pad : self.cs - y1pad,
                    self.cs - x1pad :,
                ] = np.flip(
                    self.inimg[:, y0 + y0pad : y1 - y1pad, x1 - x1pad * 2 : x1 - x1pad],
                    axis=2,
                )
                if y0pad > 0:  # top-right
                    ret[:, :y0pad, -x1pad:] = np.flip(
                        self.inimg[:, :y0pad, -x1pad:], axis=(1, 2)
                    )
                if y1pad > 0:  # bottom-right
                    ret[:, -y1pad:, -x1pad:] = np.flip(
                        self.inimg[:, -y1pad:, -x1pad:], axis=(1, 2)
                    )
            if y0pad > 0:
                ret[:, :y0pad, x0pad : self.cs - x1pad] = np.flip(
                    self.inimg[:, y0 + y0pad : y0 + y0pad * 2, x0 + x0pad : x1 - x1pad],
                    axis=1,
                )
            if y1pad > 0:
                ret[:, self.cs - y1pad :, x0pad : self.cs - x1pad] = np.flip(
                    self.inimg[:, y1 - y1pad * 2 : y1 - y1pad, x0 + x0pad : x1 - x1pad],
                    axis=1,
                )
            # useful info
            usefuldim = (
                self.pad,
                self.pad,
                self.cs - max(self.pad, x1pad),
                self.cs - max(self.pad, y1pad),
            )
            usefulstart = (x0 + self.pad, y0 + self.pad)
        return (
            torch.tensor(ret),
            torch.IntTensor(usefuldim),
            torch.IntTensor(usefulstart),
        )

    def __len__(self):
        """Return the total number of tiles in the dataset.

        For whole-image mode, always returns 1. For tiled mode, returns the
        total number of overlapping tiles needed to cover the entire image.

        Returns:
            int: Number of tiles/samples in the dataset.
        """
        return self.size


def run_from_args(args):
    """Execute the complete image denoising pipeline from command-line arguments.

    This is the main orchestration function that coordinates all aspects of the
    denoising process: model loading, image tiling, neural network inference,
    tile reconstruction, and metadata preservation. It handles both tiled and
    whole-image processing modes.

    The function performs the following steps:
    1. Auto-detects network architecture and sets appropriate tile sizes
    2. Initializes the neural network model and loads pretrained weights
    3. Creates a dataset for image tiling with overlap and padding
    4. Processes tiles through the network using PyTorch DataLoader
    5. Reconstructs the full denoised image from processed tiles
    6. Saves the result and preserves EXIF metadata

    Args:
        args: Argument namespace containing all configuration parameters including:
            - model_path (str): Path to the pretrained model file
            - input (str): Path to the input noisy image
            - output (str): Path for the output denoised image
            - g_network (str): Network architecture ("UNet", "UtNet", etc.)
            - cs (int): Crop size for tiling
            - ucs (int): Useful crop size within each tile
            - overlap (int): Overlap between adjacent tiles
            - batch_size (int): Number of tiles to process simultaneously
            - whole_image (bool): Whether to process entire image at once
            - debug (bool): Whether to save intermediate debug images
            - exif_method (str): Method for EXIF metadata preservation

    Raises:
        ValueError: If the image exceeds maximum subpixel limits.
        AssertionError: If model_path is None.
        SystemExit: If network architecture cannot be determined.

    Note:
        The function modifies the args object in-place by calling autodetect_network_cs_ucs().
        Debug mode saves intermediate tile images to './dbg/' directory.
    """
    assert args.model_path is not None
    autodetect_network_cs_ucs(args)

    def make_seamless_edges(tcrop, x0, y0):
        if x0 != 0:  # left
            tcrop[:, :, 0 : args.overlap] = tcrop[:, :, 0 : args.overlap].div(2)
        if y0 != 0:  # top
            tcrop[:, 0 : args.overlap, :] = tcrop[:, 0 : args.overlap, :].div(2)
        if x0 + args.ucs < fswidth and args.overlap:  # right
            tcrop[:, :, -args.overlap :] = tcrop[:, :, -args.overlap :].div(2)
        if y0 + args.ucs < fsheight and args.overlap:  # bottom
            tcrop[:, -args.overlap :, :] = tcrop[:, -args.overlap :, :].div(2)
        return tcrop

    if not torch.accelerator.is_available():
        print(
            "warning: PyTorch does not have access to an accelerator (means no gpu found probably). Defaulting to CPU."
        )
    torch.manual_seed(123)
    device = torch.accelerator.current_accelerator()
    if args.output is None:
        args.output = make_output_fpath(args.input, args.model_path)

    # ugly hardcoded hack for now
    if args.model_parameters is None and "activation" in args.model_path:
        args.model_parameters = f"activation={args.model_path.split('activation')[-1].split('_')[1].split('_')[0]}"
        print(f"set model_parameters to {args.model_parameters} based on model_path")
    model = Model.instantiate_model(
        network=args.g_network,
        model_path=args.model_path,
        strparameters=args.model_parameters,
        keyword="generator",
        device=device,
        models_dpath=args.models_dpath,
    )
    model.eval()  # evaluation mode
    model = model.to(device)
    ds = OneImageDS(
        args.input,
        args.cs,
        args.ucs,
        args.overlap,
        whole_image=args.whole_image,
        pad=args.pad,
    )
    DLoader = DataLoader(
        dataset=ds,
        num_workers=(
            0
            if args.batch_size == 1
            else max(min(args.batch_size, os.cpu_count() // 4), 1)
        ),
        drop_last=False,
        batch_size=args.batch_size,
        shuffle=False,
    )
    topil = torchvision.transforms.ToPILImage()
    fsheight, fswidth = cv2.imread(args.input, -1).shape[0:2]
    newimg = torch.zeros(3, fsheight, fswidth, dtype=torch.float32)

    start_time = time.time()
    for n_count, ydat in enumerate(DLoader):
        print(str(n_count) + "/" + str(int(len(ds) / args.batch_size)))
        ybatch, usefuldims, usefulstarts = ydat
        if (
            args.max_subpixels is not None
            and math.prod(ybatch.shape) > args.max_subpixels
        ):
            raise ValueError(
                f"denoise_image: {ybatch.shape=}, {math.prod(ybatch.shape)=} > {args.max_subpixels=} for {args.input=}; aborting"
            )
        ybatch = ybatch.to(device)
        xbatch = model(ybatch)
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        for i in range(ybatch.size(0)):
            ud = usefuldims[i]
            # pytorch represents images as [channels, height, width]
            # TODO test leaving on GPU longer
            # TODO reconstruct image with batch_size > 1
            tensimg = xbatch[i][:, ud[1] : ud[3], ud[0] : ud[2]].cpu().detach()
            if args.whole_image:
                newimg = tensimg
            else:
                absx0, absy0 = tuple(usefulstarts[i].tolist())
                tensimg = make_seamless_edges(tensimg, absx0, absy0)
                if args.debug:
                    os.makedirs("dbg", exist_ok=True)
                    torchvision.utils.save_image(
                        xbatch[i],
                        "dbg/crop" + str(n_count) + "_" + str(i) + "_denoised.jpg",
                    )
                    torchvision.utils.save_image(
                        tensimg,
                        "dbg/crop" + str(n_count) + "_" + str(i) + "_tensimg.jpg",
                    )
                    torchvision.utils.save_image(
                        ybatch[i],
                        "dbg/crop" + str(n_count) + "_" + str(i) + "_noisy.jpg",
                    )
                    print(tensimg.shape)
                    print((absx0, absy0, ud))
                newimg[
                    :,
                    absy0 : absy0 + tensimg.shape[1],
                    absx0 : absx0 + tensimg.shape[2],
                ] = newimg[
                    :,
                    absy0 : absy0 + tensimg.shape[1],
                    absx0 : absx0 + tensimg.shape[2],
                ].add(
                    tensimg
                )
    if args.debug:
        torchvision.utils.save_image(
            xbatch[i].cpu().detach(), args.output + "dbg_inclborders.tif"
        )  # dbg: get img with borders
    pt_helpers.tensor_to_imgfile(newimg, args.output)
    print(f"Denoised image written to {args.output}")
    if args.output[:-4] == ".jpg" and args.exif_method == "piexif":
        piexif.transplant(args.input, args.output)
    elif args.exif_method != "noexif":
        exiv_src = exiv2.ImageFactory.open(args.input)
        exiv_src.readMetadata()
        exiv_dst = exiv2.ImageFactory.open(args.output)
        exiv_dst.setExifData(exiv_src.exifData())
        exiv_dst.writeMetadata()

    print(f"Wrote denoised image to {args.output}")


def main():
    """Command-line interface for neural network-based image denoising.

    This function provides a complete command-line interface for denoising images
    using pretrained neural networks. It supports various network architectures
    (UNet, UtNet), automatic parameter detection, tiled processing for large images,
    and EXIF metadata preservation.

    The interface supports both automatic configuration (network type and tile sizes
    inferred from model path) and manual configuration of all parameters. Images
    can be processed either as whole images (memory-intensive) or using overlapping
    tiles (memory-efficient).

    Command-line Arguments:
        --input: Input image file path
        --output: Output denoised image path (auto-generated if not specified)
        --model_path: Path to pretrained model (.pt or .pth file)
        --network: Network architecture (UNet, UtNet, or auto-detect)
        --cs: Crop size (tile size including padding)
        --ucs: Useful crop size (content size within tile)
        --overlap: Overlap between adjacent tiles in pixels
        --batch_size: Number of tiles to process simultaneously
        --whole_image: Process entire image at once (high memory usage)
        --pad: Padding amount for whole-image mode
        --debug: Save intermediate tile images for debugging
        --exif_method: EXIF metadata preservation method
        --max_subpixels: Maximum subpixel limit for memory control
        --model_parameters: Additional model parameters
        --models_dpath: Root directory for model files

    Examples:
        Whole image processing (high memory):
        python brummer2019.py --input img.tif --model_path model.pt --whole_image

        Tiled processing (memory efficient):
        python brummer2019.py --input img.tif --model_path model.pt --cs 512 --ucs 384

        Auto-detection with debugging:
        python brummer2019.py --input img.tif --model_path utnet_model.pt --debug
    """
    parser = configargparse.ArgumentParser(
        description=__doc__,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--cs",
        type=int,
        help="Tile size (model was probably trained with 128, different values will work with unknown results)",
    )
    parser.add_argument(
        "--ucs",
        type=int,
        help="Useful tile size (should be <=.75*cs for U-Net, a smaller value may result in less grid artifacts but costs computation time",
    )
    parser.add_argument(
        "-ol",
        "--overlap",
        default=6,
        type=int,
        help="Merge crops with this much overlap (Reduces grid artifacts, may reduce sharpness between crops, costs computation time)",
    )
    parser.add_argument(
        "-i", "--input", default="in.jpg", type=str, help="Input image file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file with extension (default: model_dpath/test/denoised_images/fn.tif)",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1)  # TODO >1 is broken
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug (store all intermediate crops in ./dbg, display useful messages)",
    )
    parser.add_argument(
        "--exif_method",
        default="piexif",
        type=str,
        help="How is exif data copied over? (piexif, exiftool, noexif)",
    )
    parser.add_argument(
        "--g_network",
        "--network",
        "--arch",
        type=str,
        help="Generator network (typically UNet or UtNet)",
    )
    parser.add_argument(
        "--model_path",
        help="Generator pretrained model path (.pth for model, .pt for dictionary), required",
    )
    parser.add_argument(
        "--model_parameters",
        type=str,
        help='Model parameters with format "parameter1=value1,parameter2=value2"',
    )
    parser.add_argument(
        "--max_subpixels",
        type=int,
        help="Max. number of sub-pixels, abort if exceeded.",
    )
    parser.add_argument(
        "--whole_image",
        action="store_true",
        help="Ignore cs and ucs, denoise whole image",
    )
    parser.add_argument(
        "--pad",
        type=int,
        help="Padding amt per side, only used for whole image (otherwise (cs-ucs)/2",
    )
    parser.add_argument(
        "--models_dpath",
        help="Directory where all models are saved (used when a model name is provided as model_path)",
    )

    args, _ = parser.parse_known_args()
    run_from_args(args)


if __name__ == "__main__":
    main()


class Model:
    """Wrapper class for neural network model management and utilities.

    This class provides utilities for model instantiation, saving, path resolution,
    and training/evaluation mode switching. It supports both full model serialization
    and state dictionary serialization, with automatic device management.

    Attributes:
        print: Print function for logging (either built-in print or custom printer).
        loss (int): Loss value (legacy attribute, defaults to 1).
        save_dict (bool): Whether to save only state dict or full model.
        device: PyTorch device for model operations.
        debug_options (list): Debug configuration options.
        model: The actual PyTorch model instance (set during instantiation).
    """

    def __init__(
        self,
        save_dict=True,
        device=torch.accelerator.current_accelerator(),
        printer=None,
        debug_options=[],
    ):
        """Initialize the Model wrapper.

        Args:
            save_dict (bool, optional): If True, save only model state dict,
                otherwise save entire model. Defaults to True.
            device: PyTorch device for model operations. Defaults to current accelerator.
            printer: Custom printer object with print method. If None, uses built-in print.
                Defaults to None.
            debug_options (list, optional): List of debug configuration options.
                Defaults to empty list.
        """
        if printer is None:
            self.print = print
        else:
            self.print = printer.print
        self.loss = 1
        self.save_dict = save_dict
        self.device = device
        self.debug_options = debug_options

    def save_model(self, model_dir, epoch, name):
        """Save the model to disk with epoch-specific naming.

        Saves either the model's state dictionary or the complete model object
        based on the save_dict configuration. Creates a standardized filename
        format: {name}_{epoch}.pt for state dict or {name}_{epoch}.pth for full model.

        Args:
            model_dir (str): Directory path where the model should be saved.
            epoch (int): Training epoch number to include in filename.
            name (str): Base name for the saved model file.
        """
        save_path = os.path.join(model_dir, "%s_%u.pt" % (name, epoch))
        if self.save_dict:
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model, save_path + "h")

    @staticmethod
    def complete_path(path, models_dpath, keyword=""):
        """Resolve and complete model file paths for various input formats.

        This method handles multiple path resolution scenarios:
        - Direct file paths (returns as-is)
        - Directory paths (finds best or latest model within)
        - Model names within a models root directory
        - Automatic selection based on training results when available

        The method prioritizes models based on validation loss from training results
        (trainres.json) when available, falling back to the highest epoch number.

        Args:
            path (str): Input path - can be a file, directory, or model name.
            models_dpath (str): Root directory containing all model directories.
            keyword (str, optional): Model type keyword for filtering (e.g., "generator").
                Defaults to empty string.

        Returns:
            str: Complete filepath to the selected model file.

        Raises:
            SystemExit: If the specified path cannot be resolved to a valid model file.

        Note:
            For generator models, the method attempts to use best validation loss
            from trainres.json. For other model types or when trainres.json is
            unavailable, it selects the model with the highest epoch number.
        """

        def find_highest(paths, model_t):
            best = [None, 0]
            for path in paths:
                curval = int(path.split("_")[-1].split(".")[0])
                if curval > best[1] and model_t in path:
                    best = [path, curval]
            return best[0]

        def find_best(dpath, model_t):
            if model_t != "generator":
                return None  # hardcoded rules and not implemented for discriminators
            resdpath = os.path.join(dpath, "trainres.json")
            if not os.path.isfile(resdpath):
                print(f"find_best did not find {resdpath}")
                return None
            with open(resdpath, "r") as fp:
                res = json.load(fp)
                best_epoch = res["best_epoch"]["validation_loss"]
            return os.path.join(dpath, f"generator_{best_epoch}.pt")

        if os.path.isfile(path):
            # path exists; nothing to do
            return path
        elif os.path.isdir(path):
            # path is a directory; try to find best model from json, or return latest
            best_model_path = find_best(path, model_t=keyword)
            if best_model_path is not None:
                return best_model_path
            return os.path.join(path, find_highest(os.listdir(path), keyword))
        elif os.path.isdir(os.path.join(models_dpath, path)):
            # if models_dpath/path is a directory, recurse
            return Model.complete_path(os.path.join(models_dpath, path), keyword)
        else:
            print("Model path not found: %s" % path)
            exit(0)

    @staticmethod
    def instantiate_model(
        models_dpath,
        model_path=None,
        network=None,
        device=torch.accelerator.current_accelerator(check_available=True),
        strparameters=None,
        pfun=print,
        keyword="",
        **parameters,
    ):
        """Instantiate a neural network model with pretrained weights.

        Creates and loads a PyTorch model from either a complete model file (.pth)
        or state dictionary file (.pt). Handles parameter parsing, device placement,
        and automatic path resolution. Supports both loading pretrained models and
        creating new model instances.

        Args:
            models_dpath (str): Root directory path containing model files.
            model_path (str, optional): Path to model file or directory. If None,
                creates a new model instance. Defaults to None.
            network (str, optional): Network architecture class name (e.g., "UNet", "UtNet").
                Required for .pt files and new model creation. Defaults to None.
            device: PyTorch device for model placement. Defaults to current accelerator
                with availability check.
            strparameters (str, optional): Comma-separated parameter string in format
                "param1=value1,param2=value2". Defaults to None.
            pfun (callable, optional): Print function for logging. Defaults to built-in print.
            keyword (str, optional): Model type keyword for path resolution filtering.
                Defaults to empty string.
            **parameters: Additional keyword parameters passed to model constructor.

        Returns:
            torch.nn.Module: Instantiated and loaded model ready for inference or training.

        Raises:
            SystemExit: If model path is invalid or loading fails.
            AssertionError: If network architecture is not specified for .pt files.

        Note:
            - .pth files contain complete model objects and don't require network specification
            - .pt files contain only state dictionaries and require network architecture
            - String parameters override keyword parameters when both are provided
            - Model is automatically moved to the specified device before returning
        """
        model = None
        device = torch.device("cpu") if device is None else device
        if strparameters is not None and strparameters != "":
            parameters.update(
                dict([parameter.split("=") for parameter in strparameters.split(",")])
            )
        if model_path is not None:
            path = Model.complete_path(
                path=model_path, keyword=keyword, models_dpath=models_dpath
            )
            if path.endswith(".pth"):
                model = torch.load(path, map_location=device)
            elif path.endswith("pt"):
                assert network is not None
                model = globals()[network](**parameters)
                model.load_state_dict(torch.load(path, map_location=device))
            else:
                pfun("Error: unable to load invalid model path: %s" % path)
                exit(1)
        else:
            model = globals()[network](**parameters)
        return model.to(device)

    def eval(self):
        """Set the model to evaluation mode.

        Switches the model to evaluation mode, which affects behavior of certain
        layers like dropout and batch normalization. In evaluation mode, dropout
        is disabled and batch normalization uses running statistics rather than
        batch statistics.

        Returns:
            Model: Self reference for method chaining.
        """
        self.model = self.model.eval()
        return self

    def train(self):
        """Set the model to training mode.

        Switches the model to training mode, which enables gradient computation
        and affects behavior of layers like dropout and batch normalization.
        In training mode, dropout is active and batch normalization updates
        running statistics.

        Returns:
            Model: Self reference for method chaining.
        """
        self.model = self.model.train()
        return self
