# TODO: add sharpening as optional data augmentation
# from random import randint, uniform, choice
import sys

# from PIL import Image, ImageOps
from nind_denoise.libs.brummer2019.dataset import tds, vds

sys.path.append("../..")

# TODO save img fun


if __name__ == "__main__":
    print("test dataset:")
    for gt, noisy in tds.get_imgs():
        print("{}, {}".format(gt.shape, noisy.shape))
    print("val dataset:")
    for gt, noisy in vds.get_imgs():
        print("{}, {}".format(gt.shape, noisy.shape))
