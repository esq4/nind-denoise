"""
Train a denoising neural network with the NIND (or any dataset of clean-noisy images)
Optionally uses up to 2 (c)GAN discriminators. These can be turned on with --weight_D{1,2} > 0

egrun:
first check that everything works with a dummy run:
python3 nn_train.py --config configs/train_conf_unet.yaml --debug_options output_val_images output_test_images keep_all_output_images short_run --test_interval 0 --epochs 6
then launch the training with (eg):
python3 nn_train.py --config configs/train_conf_unet.yaml --debug_options output_val_images --test_interval 0 --epochs 600

Note that the discriminators are experimental and currently unmaintained; it is unknown whether they
will function with the current state of the source code.


"""

import math
from enum import Enum

from nind_denoise.pipeline.denoise.brummer2019 import Model

"""
TODO:
models: # TODO https://github.com/DingXiaoH/RepVGG

train w/ clean-clean images too. eg FP limited to ISO200
"""
# This replaces run_nn.py

# TODO reset at the end of epoch if stuck producing garbage
# TODO functions
# TODO full test every full_test_interval (done but too memory hungry at 32GB, could use cropping)

import collections
import datetime
import os
import random
import shutil
import statistics
import sys
import time

import configargparse
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from nind_denoise.libs.common import json_saver, pt_losses, pt_ops

DEFAULT_CONFIG_FPATH = os.path.join("configs", "train_conf_defaults.yaml")


# Create validation set and test it now rather than waiting until the end of the first epoch
def validate_generator(model, validation_set, output_to_dir=None):
    """
    currently doing one image at a time, limited to the same crop size as training.
    TODO check if too slow. possibly use other sizes. (If used once per epoch then shouldn't matter.)
    """
    model.eval()
    losses = []
    for i, (clean, noisy) in enumerate(validation_set):
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        denoised = model.denoise_batch(noisy)
        denoised_fs = denoised.clone().detach().cpu()
        model.compute_loss(
            pt_ops.pt_crop_batch(denoised, args.loss_cs),
            pt_ops.pt_crop_batch(clean, args.loss_cs),
        )
        if output_to_dir is not None:
            os.makedirs(output_to_dir, exist_ok=True)
            # This saves as 8-bit tiff (we don't really care for the preview) and includes borders
            torchvision.utils.save_image(
                denoised_fs, os.path.join(output_to_dir, str(i) + ".tif")
            )
        losses.append(model.get_loss(component="weighted"))
    avgloss = statistics.mean(losses)
    model.train()
    return avgloss


def test_generator(model, test_set, output_to_dir=None):
    """
    This test moves the model to CPU and tests on whole images. It is meant to run extremely slowly
    and should not be used frequently.
    FIXME: add padding to the lossf
    """
    model.eval()
    model.tocpu()
    losses = []
    for i, (clean, noisy) in enumerate(test_set):
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        denoised = model.denoise_batch(noisy)
        model.compute_loss(denoised, clean)
        if output_to_dir is not None:
            os.makedirs(output_to_dir, exist_ok=True)
            torchvision.utils.save_image(
                denoised, os.path.join(output_to_dir, str(i) + ".tif")
            )
        losses.append(model.get_loss(component="weighted"))
    avgloss = statistics.mean(losses)
    model.todevice()
    model.train()
    return avgloss


def delete_outperformed_models(
    dpath: str, keepers: set, model_t: str = "generator", keep_all_output_images=False
):
    """
    remove models whose epoch is not in the keepers set
    """
    removed = list()
    for fn in os.listdir(dpath):
        fpath = os.path.join(dpath, fn)
        if (fn == "val" or fn == "testimages") and not keep_all_output_images:
            for subdir in os.listdir(fpath):
                if int(subdir) not in keepers:
                    val_dpath = os.path.join(fpath, subdir)
                    shutil.rmtree(val_dpath)
                    removed.append(val_dpath)
            continue
        if not fn.startswith(f"{model_t}_"):
            continue
        epoch = int(fn.split("_")[1].split(".")[0])
        if epoch not in keepers:

            os.remove(fpath)
            removed.append(fpath)
    return removed


def get_test_reserve_list(test_reserve):
    """
    input: test_reserve argument (list or yaml path)
    output: test_reserve list
    """
    if len(test_reserve) == 1:
        if test_reserve[0].endswith(".yaml"):
            with open(test_reserve[0], "r") as fp:
                return yaml.safe_load(fp)
        elif test_reserve[0] == "0":
            return []
    return test_reserve


class Generator(Model):
    def __init__(
        self,
        network,
        device,
        weights,
        beta1,
        lr,
        patience,
        models_dpath,
        model_path=None,
        activation="PReLU",
        funit=32,
        printer=None,
        compute_SSIM_anyway=False,
        save_dict=True,
        debug_options=[],
        reduce_lr_factor=0.75,
    ):
        Model.__init__(self, save_dict, device, printer, debug_options=[])
        self.weights = weights
        self.criterions = dict()
        if weights["SSIM"] > 0 or compute_SSIM_anyway:
            self.criterions["SSIM"] = pt_losses.SSIM_loss().to(device)
        if weights["L1"] > 0:
            self.criterions["L1"] = torch.nn.L1Loss(reduction=None).to(device)
        if weights["MSE"] > 0:
            self.criterions["MSE"] = torch.nn.MSELoss(reduction=None).to(device)
        if weights["MSSSIM"] > 0:
            self.criterions["MSSSIM"] = pt_losses.MS_SSIM_loss().to(device)
        if weights["D1"] > 0:
            self.print(
                "DBG/FIXME may need to mess with the losses reduction for discriminator compatibility"
            )
            self.criterions["D1"] = torch.nn.MSELoss().to(device)
        if weights["D2"] > 0:
            self.criterions["D2"] = torch.nn.MSELoss().to(device)
        self.model = self.instantiate_model(
            model_path=model_path,
            network=network,
            pfun=self.print,
            device=device,
            funit=funit,
            keyword="generator",
            models_dpath=models_dpath,
            activation=activation,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, 0.999), amsgrad=True
        )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=reduce_lr_factor, verbose=True, threshold=1e-8, patience=patience)
        self.device = device
        self.loss = {
            "SSIM": 1,
            "L1": 1,
            "D1": 1,
            "D2": 1,
            "MSE": 1,
            "MSSSIM": 1,
            "weighted": 1,
        }
        self.compute_SSIM_anyway = compute_SSIM_anyway

    def get_loss(self, pretty_printed=False, component="weighted"):
        """
        Return a component of the last computed loss (print-only; no compute)
        """
        if pretty_printed:
            return ", ".join(
                [
                    "%s: %.3f" % (key, val) if val != 1 else "NA"
                    for key, val in self.loss.items()
                ]
            )
        return self.loss[component]

    def denoise_batch(self, noisy_batch):
        return self.model(noisy_batch).clip(0, 1)

    def learn(
        self,
        generated_batch_cropped,
        clean_batch_cropped,
        discriminators_predictions=[None, None],
    ):
        """
        input: self-generated clean batch, noisy batch
        compute loss and optimize self
        """
        loss = self.compute_loss(
            generated_batch_cropped, clean_batch_cropped, discriminators_predictions
        )
        #         # too late, must have gotten a bad crop from previous batch
        #         if loss > 0.4:
        #             if self.stable:
        #                 os.makedirs('dbg', exist_ok=True)
        #                 torchvision.utils.save_image(generated_batch_cropped, os.path.join('dbg', str(self)+'_gen.png'))
        #                 torchvision.utils.save_image(clean_batch_cropped, os.path.join('dbg', str(self)+'_gt.png'))
        #                 breakpoint()
        #         elif loss < 0.2 and not self.stable:
        #             self.stable = True
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def compute_loss(
        self,
        generated_batch_cropped,
        clean_batch_cropped,
        discriminators_predictions=[None, None],
    ):
        """
        Compute the loss between a denoised and ground-truth batch. Update internal loss dict and
        return weighted loss tensor.
        """
        loss_weighted = 0
        for loss_name, weight in self.weights.items():
            if weight == 0:
                continue
            elif loss_name[0] == "D":
                discriminator_predictions = discriminators_predictions[
                    int(loss_name[1])
                ]
                if discriminator_predictions is None:
                    continue
                self.loss[loss_name] = self.criterions[loss_name](
                    discriminator_predictions,
                    gen_target_probabilities(
                        True,
                        discriminator_predictions.shape,
                        device=self.device,
                        noisy=False,
                    ),
                )
            else:
                self.loss[loss_name] = self.criterions[loss_name](
                    generated_batch_cropped, clean_batch_cropped
                )
            loss_weighted += self.loss[loss_name] * weight
            self.loss[loss_name] = self.loss[loss_name].mean().item()
        self.loss["weighted"] = loss_weighted.mean().item()
        # Debug. can be removed to increase performance slightly
        if self.loss["weighted"] < 0.25 and loss_weighted.min() > 0.4:
            self.p.print("problematic crop saved to dbg")
            worstval, worstindex = loss_weighted.max()
            os.makedirs("dbg", exist_ok=True)
            torchvision.utils.save_image(
                generated_batch_cropped,
                os.path.join(
                    "dbg",
                    f'{self}_{float(self.loss["weighted"])}_{float(worstval)}_{int(worstindex)}_gen.png',
                ),
            )
            torchvision.utils.save_image(
                clean_batch_cropped,
                os.path.join(
                    "dbg",
                    f'{self}_{float(self.loss["weighted"])}_{float(worstval)}_{int(worstindex)}_gt.png',
                ),
            )
            breakpoint()
        return loss_weighted

    def update_learning_rate(self, lr_decay):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_decay
        return param_group["lr"] * lr_decay

    def zero_grad(self):
        self.optimizer.zero_grad()

    def tocpu(self):
        self.model = self.model.cpu()
        for lossname, lossf in self.criterions.items():
            if lossf is None:
                continue
            self.criterions[lossname] = lossf.cpu()

    def todevice(self):
        self.model = self.model.to(self.device)
        for lossname, lossf in self.criterions.items():
            if lossf is None:
                continue
            self.criterions[lossname] = lossf.to(self.device)


class Discriminator(Model):
    def __init__(
        self,
        models_dpath,
        beta1,
        lr,
        patience,
        network="Hul112Disc",
        weights_dict_path=None,
        model_path=None,
        device=torch.accelerator.current_accelerator(),
        loss_function="MSE",
        activation="PReLU",
        funit=32,
        not_conditional=False,
        printer=None,
        save_dict=True,
        debug_options=[],
        reduce_lr_factor=0.75,
    ):
        Model.__init__(self, save_dict, device, printer, debug_options)
        self.device = device
        self.loss = 1
        self.loss_function = loss_function
        if loss_function == "MSE":
            self.criterion = torch.nn.MSELoss().to(device)
        if not_conditional:
            input_channels = 3
        else:
            input_channels = 6
        self.model = self.instantiate_model(
            model_path=model_path,
            models_dpath=models_dpath,
            network=network,
            pfun=self.print,
            device=device,
            funit=funit,
            input_channels=input_channels,
        )
        # elif network == 'PatchGAN':
        #    self.model = net_d = define_D(input_channels, 2*funit, 'basic', gpu_id=device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.conditional = not not_conditional
        self.predictions_range = None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.75, verbose=True, threshold=1e-8, patience=patience
        )

    def update_learning_rate(self, avg_loss):
        self.scheduler.step(metrics=avg_loss)
        lr = self.optimizer.param_groups[0]["lr"]
        self.print("Learning rate: %f" % lr)
        return lr

    def get_loss(self):
        return self.loss

    def get_predictions_range(self):
        return "range (r-r+f-f+): " + str(self.predictions_range)

    def update_loss(self, loss_fake, loss_real):
        if self.loss_function == "MSE":
            self.loss = (math.sqrt(loss_fake) + math.sqrt(loss_real)) / 2
        else:
            self.print(
                "Error: loss function not implemented: %s" % (self.loss_function)
            )

    def discriminate_batch(self, generated_batch_cropped, noisy_batch_cropped=None):
        if self.conditional:
            fake_batch = torch.cat([noisy_batch_cropped, generated_batch_cropped], 1)
        else:
            fake_batch = generated_batch_cropped
        return self.model(fake_batch)

    def learn(
        self, generated_batch_cropped, clean_batch_cropped, noisy_batch_cropped=None
    ):
        self.optimizer.zero_grad()
        if self.conditional:
            real_batch = torch.cat([noisy_batch_cropped, clean_batch_cropped], 1)
            fake_batch = torch.cat(
                [noisy_batch_cropped, generated_batch_cropped.detach()], 1
            )
        else:
            real_batch = clean_batch_cropped
            fake_batch = generated_batch_cropped.detach()
        if "discriminator_input" in self.debug_options:
            os.makedirs("dbg", exist_ok=True)
            batch_savename = os.path.join("dbg", str(time.time()))
            if self.conditional:
                real_batch_detached = (
                    torch.cat([real_batch[:, :3, :, :], real_batch[:, 3:, :, :]], 0)
                    .detach()
                    .cpu()
                )
                fake_batch_detached = (
                    torch.cat([fake_batch[:, :3, :, :], fake_batch[:, 3:, :, :]], 0)
                    .detach()
                    .cpu()
                )
            else:
                real_batch_detached = real_batch.detach().cpu()
                fake_batch_detached = fake_batch.detach().cpu()
            torchvision.utils.save_image(
                real_batch_detached, batch_savename + "_real.png"
            )
            torchvision.utils.save_image(
                fake_batch_detached, batch_savename + "_fake.png"
            )

        pred_real = self.model(real_batch)
        loss_real = self.criterion(
            pred_real,
            gen_target_probabilities(
                True, pred_real.shape, device=self.device, noisy=True
            ),
        )
        loss_real_detached = loss_real.item()
        loss_real.backward()
        pred_fake = self.model(fake_batch)
        loss_fake = self.criterion(
            pred_fake,
            gen_target_probabilities(
                False, pred_fake.shape, device=self.device, noisy=self.loss < 0.25
            ),
        )
        loss_fake_detached = loss_fake.item()
        loss_fake.backward()
        try:
            self.predictions_range = ", ".join(
                [
                    "{:.2}".format(float(i))
                    for i in (
                        pred_real.min(),
                        pred_real.max(),
                        pred_fake.min(),
                        pred_fake.max(),
                    )
                ]
            )
        except:
            self.predictions_range = "(not implemented)"
        self.update_loss(loss_fake_detached, loss_real_detached)
        self.optimizer.step()


COMMON_CONFIG_FPATH = os.path.join("configs", "common_conf_default.yaml")


class DebugOptions(Enum):
    SHORT_RUN = "short_run"
    CHECK_DATASET = "check_dataset"
    OUTPUT_VAL_IMAGES = "output_val_images"
    OUTPUT_TEST_IMAGES = "output_test_images"
    KEEP_ALL_OUTPUT_IMAGES = "keep_all_output_images"


class Printer:
    def __init__(self, tostdout=True, tofile=True, file_path="log"):
        self.tostdout = tostdout
        self.tofile = tofile
        self.file_path = file_path

    def print(self, msg):
        if self.tostdout:
            print(msg)
        if self.tofile:
            try:
                with open(self.file_path, "a") as f:
                    f.write(str(msg) + "\n")
            except Exception as e:
                print("Warning: could not write to log: %s" % e)


def get_weights(args):
    total = 0
    weights = {"MSSSIM": 0, "L1": 0, "MSE": 0, "SSIM": 0, "D1": 0, "D2": 0}
    if args.weight_SSIM:
        weights["SSIM"] = args.weight_SSIM
        total += args.weight_SSIM
    if args.weight_MSSSIM:
        weights["MSSSIM"] = args.weight_MSSSIM
        total += args.weight_MSSSIM
    if args.weight_L1:
        weights["L1"] = args.weight_L1
        total += args.weight_L1
    if args.weight_D1:
        weights["D1"] = args.weight_D1
        total += args.weight_D1
    if args.weight_D2:
        weights["D2"] = args.weight_D2
        total += args.weight_D2
    if args.weight_MSE:
        weights["MSE"] = args.weight_MSE
        total += args.weight_MSE
    if total == 0:
        raise NotImplementedError
    elif total != 1:
        for akey in weights.keys():
            weights["akey"] /= total
    assert sum(weights.values()) == 1, weights
    print(f"Loss weights: {weights}")
    return weights


if __name__ == "__main__":

    # Training settings
    parser = configargparse.ArgumentParser(
        description=__doc__,
        default_config_files=[COMMON_CONFIG_FPATH, DEFAULT_CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add("-c", "--config", is_config_file=True, help="(yaml) config file path")
    parser.add(
        "-c2", "--config2", is_config_file=True, help="extra (yaml) config file path"
    )
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--time_limit", type=int, help="Time limit in seconds (ends training)"
    )
    parser.add_argument(
        "--g_activation",
        type=str,
        default="PReLU",
        help="Final activation function for generator",
    )
    parser.add_argument(
        "--g_funit", type=int, default=32, help="Filter unit size for generator"
    )
    parser.add_argument(
        "--g_model_path",
        help="Generator pretrained model path (.pth for model, .pt for dictionary)",
    )
    parser.add_argument("--models_dpath", help="Directory where all models are saved")
    parser.add_argument("--beta1", type=float, help="beta1 for adam")
    parser.add_argument(
        "--g_lr", type=float, help="Initial learning rate for adam (generator)"
    )
    parser.add_argument(
        "--weight_SSIM", type=float, help="Weight on SSIM term in objective"
    )
    parser.add_argument(
        "--weight_MSSSIM", type=float, help="Weight on MSSSIM term in objective"
    )
    parser.add_argument(
        "--weight_L1", type=float, help="Weight on L1 term in objective"
    )
    parser.add_argument(
        "--weight_MSE", type=float, help="Weight on L1 term in objective"
    )
    parser.add_argument(
        "--test_reserve",
        nargs="*",
        required=True,
        help='Space separated list of image sets to be reserved for testing, or yaml file path containing a list. Set to "0" to use all available data.',
    )
    parser.add_argument(
        "--train_data",
        nargs="*",
        help="(space-separated) Path(s) to the pre-cropped training data",
    )
    parser.add_argument(
        "--cs",
        "--crop_size",
        type=int,
        help="Crop size fed to NN. default: no additional cropping",
    )
    parser.add_argument(
        "--min_crop_size",
        type=int,
        help="Minimum crop size. Dataset will be checked if this value is set.",
    )
    parser.add_argument(
        "--loss_cs",
        "--loss_crop_size",
        type=int,
        help="Center crop size used in loss function. default: use stride size from dataset directory name",
    )
    parser.add_argument(
        "--debug_options",
        "--debug",
        nargs="*",
        default=[],
        help=f"(space-separated) Debug options (available: {DebugOptions})",
    )
    parser.add_argument("--g_network", type=str, help="Generator network")
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Number of threads for data loader to use",
    )
    parser.add_argument(
        "--min_lr", type=float, help="Minimum learning rate (ends training)"
    )
    parser.add_argument(
        "--epochs", type=int, default=9001, help="Number of epochs (ends training)"
    )
    parser.add_argument(
        "--compute_SSIM_anyway",
        action="store_true",
        help="Compute and display SSIM loss even if not used",
    )
    parser.add_argument(
        "--freeze_generator",
        action="store_true",
        help="Freeze generator until discriminator is useful",
    )
    parser.add_argument(
        "--start_epoch", default=1, type=int, help="Starting epoch (cosmetics)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Number of epochs without improvements before scheduler updates learning rate",
    )
    parser.add_argument(
        "--reduce_lr_factor",
        type=float,
        help="LR is multiplied by this factor when model performs poorly for <patience> epochs",
    )
    parser.add_argument(
        "--validation_interval",
        help="Validation interval in # of epochs. Affects learning rate update and helps to keep the best model in the end. 0 = no validation, default=1",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--test_interval",
        default=0,
        help="Test interval in # of epochs. Performed on CPU with whole images (long and WARNING: uses enormous amount of RAM; keep off without >= 64 GB). 0 = no such tests",
        type=int,
    )
    parser.add_argument(
        "--orig_data",
        help="Location of the originally downloaded train data (before cropping); used when test_interval is set",
    )
    parser.add_argument(
        "--validation_set_yaml",
        help=f"Yaml file containing a list of clean/noisy images used for validation.",
    )
    parser.add_argument(
        "--exp_mult_min",
        type=float,
        help="Minimum exposure multiplicator (data augmentation)",
    )
    parser.add_argument(
        "--exp_mult_max",
        type=float,
        help="Maximum exposure multiplicator (data augmentation)",
    )
    # add clean / hq images to the training
    parser.add_argument(
        "--clean_data_dpath",
        help="Location of the high quality (pre-cropped) clean data which can be used in training",
    )
    parser.add_argument(
        "--clean_data_ratio",
        type=float,
        help="Ratio of clean-clean to clean-noisy training data",
    )
    # discriminator stuff
    parser.add_argument(
        "--d_activation",
        type=str,
        default="PReLU",
        help="Final activation function for discriminator",
    )
    parser.add_argument(
        "--d2_activation",
        type=str,
        default="PReLU",
        help="Final activation function for discriminator",
    )
    parser.add_argument(
        "--d_funit", type=int, default=32, help="Filter unit size for discriminator"
    )
    parser.add_argument(
        "--d2_funit", type=int, default=32, help="Filter unit size for discriminator"
    )
    parser.add_argument(
        "--d_model_path",
        help="Discriminator pretrained model path (.pth for model, .pt for dictionary)",
    )
    parser.add_argument(
        "--d2_model_path",
        help="Discriminator pretrained model path (.pth for model, .pt for dictionary)",
    )
    parser.add_argument(
        "--d_loss_function", type=str, default="MSE", help="Discriminator loss function"
    )
    parser.add_argument(
        "--d2_loss_function",
        type=str,
        default="MSE",
        help="Discriminator loss function",
    )
    parser.add_argument(
        "--d_lr", type=float, help="Initial learning rate for adam (discriminator)"
    )
    parser.add_argument(
        "--d2_lr", type=float, help="Initial learning rate for adam (discriminator)"
    )
    parser.add_argument(
        "--weight_D1", type=float, help="Weight on Discriminator 1 term in objective"
    )
    parser.add_argument(
        "--weight_D2", type=float, help="Weight on Discriminator 2 term in objective"
    )
    parser.add_argument("--d_network", type=str, help="Discriminator network")
    parser.add_argument("--d2_network", type=str, help="Discriminator2 network")
    parser.add_argument(
        "--not_conditional", action="store_true", help="Regular GAN instead of cGAN"
    )
    parser.add_argument(
        "--not_conditional_2", action="store_true", help="Regular GAN instead of cGAN"
    )
    parser.add_argument(
        "--discriminator_advantage",
        type=float,
        default=0.0,
        help="Desired discriminator correct prediction ratio is 0.5+advantage",
    )
    parser.add_argument(
        "--discriminator2_advantage",
        type=float,
        default=0.0,
        help="Desired discriminator correct prediction ratio is 0.5+advantage",
    )

    args = parser.parse_args()

    # with device agnosticism:
    with (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device("cpu")
    ) as device:

        debug_options = [DebugOptions(opt) for opt in args.debug_options]

        weights = get_weights(args)
        use_D = weights["D1"] > 0
        use_D2 = weights["D2"] > 0

        expname = (
            datetime.datetime.now().isoformat()[:-10]
            + "_"
            + "_".join(sys.argv).replace("/", "-")
        )[0:255]
        model_dir = os.path.join(args.models_dpath, expname)
        os.makedirs(model_dir, exist_ok=True)
        txt_path = os.path.join(model_dir, "train.log")
        jsonsaver = json_saver.JSONSaver(
            os.path.join(model_dir, "trainres.json"), step_type="epoch"
        )

        frozen_generator = args.freeze_generator

        p = Printer(file_path=os.path.join(txt_path))

        p.print(args)
        p.print("cmd: python3 " + " ".join(sys.argv))

        args.test_reserve = get_test_reserve_list(args.test_reserve)
        p.print(f"test_reserve: {args.test_reserve}")

        # Train data
        if (
            args.min_crop_size is None or args.min_crop_size == 0
        ) and DebugOptions.CHECK_DATASET in debug_options:
            args.min_crop_size = args.cs
        DDataset = dataset_torch_3.DenoisingDataset(
            args.train_data,
            test_reserve=args.test_reserve,
            cs=args.cs,
            min_crop_size=args.min_crop_size,
            exp_mult_min=args.exp_mult_min,
            exp_mult_max=args.exp_mult_max,
        )
        if args.loss_cs is None:
            args.loss_cs = DDataset.min_crop_size
            assert args.loss_cs is not None
        if args.cs is None:
            args.cs = DDataset.cs
        if DebugOptions.SHORT_RUN in debug_options:
            DDataset.dataset = DDataset.dataset[: 3 * args.batch_size]

        if args.clean_data_ratio is not None and args.clean_data_ratio > 0:
            CCdataset = dataset_torch_3.CleanCleanDataset(
                args.clean_data_dpath, cs=args.cs
            )
            bs_clean = max(1, int(args.batch_size * args.clean_data_ratio))
            bs_std = args.batch_size - bs_clean
            p.print(
                f"Initialized clean dataset of size {len(CCdataset)}. Clean batch_size = {bs_clean}"
            )
            clean_dataloader = DataLoader(
                dataset=CCdataset,
                num_workers=min(max(1, args.threads // 2), bs_clean),
                batch_size=bs_clean,
                shuffle=True,
            )
            clean_dataloader_iterator = iter(clean_dataloader)
        else:
            bs_clean = 0
            bs_std = args.batch_size

        data_loader = DataLoader(
            dataset=DDataset,
            num_workers=args.threads,
            drop_last=True,
            batch_size=bs_std,
            shuffle=True,
        )

        # init models

        if use_D:
            discriminator = Discriminator(
                network=args.d_network,
                model_path=args.d_model_path,
                device=device,
                loss_function=args.d_loss_function,
                activation=args.d_activation,
                funit=args.d_funit,
                beta1=args.beta1,
                lr=args.d_lr,
                not_conditional=args.not_conditional,
                printer=p,
                patience=args.patience,
                debug_options=debug_options,
                models_dpath=args.models_dpath,
                reduce_lr_factor=args.reduce_lr_factor,
            )
        if use_D2:
            discriminator2 = Discriminator(
                network=args.d2_network,
                model_path=args.d2_model_path,
                device=device,
                loss_function=args.d2_loss_function,
                activation=args.d2_activation,
                funit=args.d2_funit,
                beta1=args.beta1,
                lr=args.d2_lr,
                not_conditional=args.not_conditional_2,
                printer=p,
                patience=args.patience,
                debug_options=debug_options,
                models_dpath=args.models_dpath,
                reduce_lr_factor=args.reduce_lr_factor,
            )
        generator = Generator(
            network=args.g_network,
            model_path=args.g_model_path,
            device=device,
            weights=weights,
            activation=args.g_activation,
            funit=args.g_funit,
            beta1=args.beta1,
            lr=args.g_lr,
            printer=p,
            compute_SSIM_anyway=args.compute_SSIM_anyway,
            patience=args.patience,
            debug_options=debug_options,
            models_dpath=args.models_dpath,
            reduce_lr_factor=args.reduce_lr_factor,
        )

        discriminator_predictions, discriminator2_predictions = None, None
        generator_learning_rate = args.g_lr
        discriminator_learning_rate = args.d_lr

        # Validation data
        if args.validation_interval > 0:
            validation_set = dataset_torch_3.ValidationDataset(
                args.validation_set_yaml, device=device, cs=args.cs
            )
            if DebugOptions.OUTPUT_VAL_IMAGES in debug_options:
                get_validation_dpath = lambda epoch: os.path.join(
                    model_dir, "val", str(epoch)
                )
            else:
                get_validation_dpath = lambda epoch: None
            validation_loss = validate_generator(
                generator, validation_set, output_to_dir=get_validation_dpath(0)
            )
            jsonsaver.add_res(0, {"validation_loss": validation_loss}, write=True)
            p.print(f"Validation loss: {validation_loss}")
        # Test data
        if args.test_interval > 0:
            test_set = dataset_torch_3.TestDenoiseDataset(
                data_dpath=args.orig_data, sets=args.test_reserve
            )
            if DebugOptions.OUTPUT_TEST_IMAGES in debug_options:
                get_test_dpath = lambda epoch: os.path.join(
                    model_dir, "testimages", str(epoch)
                )
            else:
                get_test_dpath = lambda epoch: None

        with open(os.path.join(model_dir, "config.yaml"), "w") as fp:
            yaml.dump(vars(args), fp)

        start_time = time.time()
        generator_loss_hist = collections.deque(maxlen=args.patience)

        # Train
        for epoch in range(args.start_epoch, args.epochs):
            loss_D_list = []
            loss_D2_list = []
            loss_G_list = []
            loss_G_SSIM_list = []
            epoch_start_time = time.time()

            for iteration, batch in enumerate(data_loader, 1):
                if bs_clean > 0:
                    try:
                        clean_batch = next(clean_dataloader_iterator)
                    except StopIteration:
                        clean_dataloader_iterator = iter(clean_dataloader)
                        clean_batch = next(clean_dataloader_iterator)
                        p.print("Reloading clean_dataloader")
                    batch[0] = torch.cat((batch[0], clean_batch[0]))
                    batch[1] = torch.cat((batch[1], clean_batch[1]))
                iteration_summary = "Epoch %u batch %u/%u: " % (
                    epoch,
                    iteration,
                    len(data_loader),
                )
                clean_batch_cropped = pt_ops.pt_crop_batch(
                    batch[0].to(device), args.loss_cs
                )
                noisy_batch = batch[1].to(device)
                noisy_batch_cropped = pt_ops.pt_crop_batch(noisy_batch, args.loss_cs)
                generated_batch = generator.denoise_batch(noisy_batch)
                generated_batch_cropped = pt_ops.pt_crop_batch(
                    generated_batch, args.loss_cs
                )
                # train discriminator based on its previous performance
                discriminator_learns = (
                    use_D
                    and (discriminator.get_loss() + args.discriminator_advantage)
                    > random.random()
                ) or frozen_generator
                if discriminator_learns:
                    discriminator.learn(
                        noisy_batch_cropped=noisy_batch_cropped,
                        generated_batch_cropped=generated_batch_cropped,
                        clean_batch_cropped=clean_batch_cropped,
                    )
                    loss_D_list.append(discriminator.get_loss())
                    iteration_summary += "loss D: %f (%s)" % (
                        discriminator.get_loss(),
                        discriminator.get_predictions_range(),
                    )
                # train discriminator2 based on its previous performance
                discriminator2_learns = (
                    use_D2
                    and (discriminator2.get_loss() + args.discriminator2_advantage)
                    > random.random()
                ) or (use_D2 and frozen_generator)
                if discriminator2_learns:
                    discriminator2.learn(
                        noisy_batch_cropped=noisy_batch_cropped,
                        generated_batch_cropped=generated_batch_cropped,
                        clean_batch_cropped=clean_batch_cropped,
                    )
                    loss_D2_list.append(discriminator2.get_loss())
                    if discriminator_learns:
                        iteration_summary += ", "
                    while len(iteration_summary) < 90:
                        iteration_summary += " "
                    iteration_summary += "loss D2: %f (%s)" % (
                        discriminator2.get_loss(),
                        discriminator2.get_predictions_range(),
                    )
                # train generator if discriminator didn't learn or discriminator is somewhat useful
                generator_learns = not frozen_generator and (
                    (not discriminator_learns and not discriminator2_learns)
                    or (
                        discriminator_learns
                        and discriminator2_learns
                        and (
                            discriminator2.get_loss()
                            + args.discriminator2_advantage
                            + discriminator.get_loss()
                            + args.discriminator_advantage
                        )
                        / 2
                        < random.random()
                    )
                    or (
                        discriminator_learns
                        and (not discriminator2_learns)
                        and discriminator.get_loss() + args.discriminator_advantage
                        < random.random()
                    )
                    or (
                        discriminator2_learns
                        and (not discriminator_learns)
                        and discriminator2.get_loss() + args.discriminator2_advantage
                        < random.random()
                    )
                )
                if generator_learns:
                    if discriminator_learns or discriminator2_learns:
                        iteration_summary += ", "
                    pregenres_space = 1 if not use_D else 125
                    pregenres_space = 160 if use_D2 else pregenres_space
                    while len(iteration_summary) < pregenres_space:
                        iteration_summary += " "
                    if use_D:
                        discriminator_predictions = discriminator.discriminate_batch(
                            generated_batch_cropped=generated_batch_cropped,
                            noisy_batch_cropped=noisy_batch_cropped,
                        )
                    if use_D2:
                        discriminator2_predictions = discriminator2.discriminate_batch(
                            generated_batch_cropped=generated_batch_cropped,
                            noisy_batch_cropped=noisy_batch_cropped,
                        )
                    else:
                        discriminator2_predictions = None
                    generator.learn(
                        generated_batch_cropped=generated_batch_cropped,
                        clean_batch_cropped=clean_batch_cropped,
                        discriminators_predictions=[
                            discriminator_predictions,
                            discriminator2_predictions,
                        ],
                    )
                    loss_G_list.append(generator.get_loss(component="weighted"))
                    if generator.weights["SSIM"] > 0 or generator.compute_SSIM_anyway:
                        loss_G_SSIM_list.append(generator.get_loss(component="SSIM"))
                    iteration_summary += "loss G: %s" % generator.get_loss(
                        pretty_printed=True
                    )
                else:
                    generator.zero_grad()
                    if frozen_generator:
                        frozen_generator = discriminator.get_loss() > 0.33 and (
                            (not use_D2) or discriminator2.get_loss() > 0.33
                        )
                p.print(iteration_summary)

            # cleanup previous epochs
            removed = delete_outperformed_models(
                dpath=model_dir,
                keepers=jsonsaver.get_best_steps(),
                model_t="generator",
                keep_all_output_images=DebugOptions.KEEP_ALL_OUTPUT_IMAGES
                in debug_options,
            )
            p.print(f"delete_outperformed_models removed {removed}")

            # Do validation
            if args.validation_interval > 0 and epoch % args.validation_interval == 0:
                validation_loss = validate_generator(
                    generator, validation_set, output_to_dir=get_validation_dpath(epoch)
                )
                jsonsaver.add_res(
                    epoch, {"validation_loss": validation_loss}, write=False
                )
                p.print(f"Validation loss: {validation_loss}")
            if args.test_interval > 0 and epoch % args.test_interval == 0:
                test_loss = test_generator(
                    generator, test_set, output_to_dir=get_test_dpath(epoch)
                )
                jsonsaver.add_res(epoch, {"test_loss": test_loss}, write=False)

            p.print("Epoch %u summary:" % epoch)
            p.print(
                "Time elapsed (s): %u (epoch), %u (total)"
                % (time.time() - epoch_start_time, time.time() - start_time)
            )
            p.print("Generator:")
            if len(loss_G_SSIM_list) > 0:
                p.print("Average SSIM loss: %f" % statistics.mean(loss_G_SSIM_list))
                jsonsaver.add_res(
                    epoch,
                    {"train_SSIM_loss": statistics.mean(loss_G_SSIM_list)},
                    write=False,
                )
            if len(loss_G_list) > 0:
                average_g_weighted_loss = statistics.mean(loss_G_list)
                p.print("Average weighted loss: %f" % average_g_weighted_loss)
                jsonsaver.add_res(
                    epoch, {"train_weighted_loss": average_g_weighted_loss}, write=False
                )
                lr_loss = (
                    validation_loss
                    if validation_loss is not None
                    else average_g_weighted_loss
                )

                if len(generator_loss_hist) > 0 and max(generator_loss_hist) < lr_loss:
                    generator_learning_rate = generator.update_learning_rate(
                        args.reduce_lr_factor
                    )
                    p.print(
                        f"Generator learning rate updated to {generator_learning_rate} because generator_loss_hist={generator_loss_hist} < lr_loss={lr_loss}"
                    )
                generator_loss_hist.append(lr_loss)

                # TODO reset to previous best (or init if epoch 1) if failed (eg lr_loss <= .4)

                jsonsaver.add_res(
                    epoch, {"gen_lr": generator_learning_rate}, write=True
                )
            else:
                p.print("Generator learned nothing")
            if use_D:
                # TODO add discriminator(s) to jsonsaver and use for cleanup if plan to use those
                p.print("Discriminator:")
                if len(loss_D_list) > 0:
                    average_d_loss = statistics.mean(loss_D_list)
                    p.print("Average normalized loss: %f" % (average_d_loss))
                    discriminator_learning_rate = discriminator.update_learning_rate(
                        average_d_loss
                    )
                    discriminator.save_model(model_dir, epoch, "discriminator")
            if use_D2:
                p.print("Discriminator2:")
                if len(loss_D2_list) > 0:
                    average_d2_loss = statistics.mean(loss_D2_list)
                    p.print("Average normalized loss: %f" % (average_d2_loss))
                    discriminator2_learning_rate = discriminator2.update_learning_rate(
                        average_d2_loss
                    )
                    discriminator2.save_model(model_dir, epoch, "discriminator2")
            if not frozen_generator:
                generator.save_model(model_dir, epoch, "generator")
            if args.time_limit and args.time_limit < time.time() - start_time:
                p.print("Time is up")
                exit(0)
            if (
                (not use_D) or discriminator_learning_rate < args.min_lr
            ) and generator_learning_rate < args.min_lr:
                p.print("Minimum learning rate reached")
                exit(0)


def gen_target_probabilities(
    target_real,
    target_probabilities_shape,
    device=None,
    invert_probabilities=False,
    noisy=True,
):
    """
    fuzziness for the discriminator's targets, because blind confidence is not right.
    """
    if (target_real and not invert_probabilities) or (
        not target_real and invert_probabilities
    ):
        if noisy:
            res = 19 / 20 + torch.rand(target_probabilities_shape) / 20
        else:
            res = torch.ones(target_probabilities_shape)
    else:
        if noisy:
            res = torch.rand(target_probabilities_shape) / 20
        else:
            res = torch.zeros(target_probabilities_shape)
    if device is None:
        return res
    else:
        return res.to(device)
