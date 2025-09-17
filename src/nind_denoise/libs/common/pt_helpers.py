import sys

import cv2
import imageio
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.append("../../common")
from . import np_imgops, pt_losses


def fpath_to_tensor(img_fpath, device=torch.device(type="cpu"), batch=False):
    # totensor = torchvision.transforms.ToTensor()
    # pilimg = Image.open(imgpath).convert('RGB')
    # return totensor(pilimg)  # replaced w/ opencv to handle >8bits
    tensor = torch.tensor(np_imgops.img_path_to_np_flt(img_fpath), device=device)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor


def tensor_to_imgfile(tensor, path):
    if tensor.dtype == torch.float32:
        if path[-4:].lower() in [".jpg", "jpeg"]:  # 8-bit
            return torchvision.utils.save_image(tensor.clip(0, 1), path)
        elif path[-4:].lower() in [".png", ".tif"]:  # 16-bit
            nptensor = (
                (tensor.clip(0, 1) * 65535)
                .round()
                .cpu()
                .numpy()
                .astype(np.uint16)
                .transpose(1, 2, 0)
            )
            nptensor = cv2.cvtColor(nptensor, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, nptensor)
        elif path[-4:].lower() in ["tiff"]:  # 32-bit
            nptensor = tensor.cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            imageio.imwrite(path, nptensor)
        else:
            raise NotImplementedError(f"Extension in {path}")
    elif tensor.dtype == torch.uint8:
        tensor = tensor.permute(1, 2, 0).to(torch.uint8).numpy()
        pilimg = Image.fromarray(tensor)
        pilimg.save(path)
    else:
        raise NotImplementedError(tensor.dtype)


def get_losses(img1_fpath, img2_fpath):
    img1 = fpath_to_tensor(img1_fpath).unsqueeze(0)
    img2 = fpath_to_tensor(img2_fpath).unsqueeze(0)
    assert img1.shape == img2.shape, f"{img1.shape=}, {img2.shape=}"
    res = dict()
    res["mse"] = torch.nn.functional.mse_loss(img1, img2).item()
    res["ssim"] = pt_losses.SSIM_loss()(img1, img2).item()
    res["msssim"] = pt_losses.MS_SSIM_loss()(img1, img2).item()
    return res


def get_device(device_n=None):
    if (
        current_device := torch.accelerator.current_accelerator(check_available=True)
    ) is not None:
        return current_device
    else:
        print("Accelerator (gpu/xpu/etc.) device not available; defaulting to cpu.")
        return torch.device("cpu")
