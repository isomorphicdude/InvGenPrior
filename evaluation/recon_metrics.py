"""Implements PSNR, SSIM, and LPIPS metrics for evaluation."""

import os

import logging
import numpy as np
import pandas as pd
import torch
import torch.utils
from torchvision import transforms
import tensorflow as tf
from absl import app, flags
from ml_collections.config_flags import config_flags

from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    learned_perceptual_image_patch_similarity,
)
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from datasets import lmdb_dataset


# compute Peak Signal-to-Noise Ratio (PSNR)
def get_psnr(data_loader):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) for a given data loader.

    The data loader should return a pair of tensors,
    where the first tensor is the output and the second tensor is the truth.

    Using torchmetrics implementation.

    Args:
      data_loader (torch.utils.data.DataLoader): The data loader to evaluate. It should return a pair of tensors,

    Returns:
      list: A list of PSNR values as torch.Tensor.
    """
    ret = []
    for output, truth in tqdm(data_loader):
        # move to cuda
        output, truth = output.cuda(), truth.cuda()
        # assert image shape of N, C, H, W
        assert (
            output.shape == truth.shape
        ), f"Output shape {output.shape} != Truth shape {truth.shape}"
        assert output.shape[1] == 3, f"Output shape {output.shape} not RGB"
        ret.append(peak_signal_noise_ratio(output, truth, reduction='none').to('cpu'))

    return ret


# compute Structural Similarity Index Measure (SSIM)
def get_ssim(data_loader):
    """
    Computes the Structural Similarity Index (SSIM) for a given data loader.

    The data loader should return a pair of tensors,
    where the first tensor is the output and the second tensor is the truth.

    Using torchmetrics implementation.

    Args:
        - data_loader: torch.utils.data.DataLoader, the data loader to evaluate.

    Returns:
        - list of SSIM values as torch.Tensor.
    """
    ret = []
    for output, truth in tqdm(data_loader):
        # move to cuda
        output, truth = output.cuda(), truth.cuda()
        # assert image shape of N, C, H, W
        assert (
            output.shape == truth.shape
        ), f"Output shape {output.shape} != Truth shape {truth.shape}"
        assert output.shape[1] == 3, f"Output shape {output.shape} not RGB"
        ret.append(structural_similarity_index_measure(output, truth).to('cpu'))
    return ret


# compute Learned Perceptual Image Patch Similarity (LPIPS)
def get_lpips(data_loader):
    """
    Computes the Learned Perceptual Image Patch Similarity (LPIPS) for a given data loader.

    The data loader should return a pair of tensors,
    where the first tensor is the output and the second tensor is the truth.

    Using torchmetrics implementation.

    Args:
        - data_loader: torch.utils.data.DataLoader, the data loader to evaluate.

    Returns:
        - list of LPIPS values as torch.Tensor.
    """
    ret = []
    for output, truth in tqdm(data_loader):
        # move to cuda
        output, truth = output.cuda(), truth.cuda()
        # assert image shape of N, C, H, W
        assert (
            output.shape == truth.shape
        ), f"Output shape {output.shape} != Truth shape {truth.shape}"
        assert output.shape[1] == 3, f"Output shape {output.shape} not RGB"
        with torch.no_grad():
            lpip = LearnedPerceptualImagePatchSimilarity(
                net_type="alex",  # choose from ['alex', 'squeeze', 'vgg']
                normalize=True,  # since I have applied inverse scaler in sampling
                reduction="mean",
            ).cuda()(output, truth)

        ret.append(lpip.detach().cpu())

    return ret


class ZippedDataset(torch.utils.data.Dataset):
    """
    Creates a zipped data set where the first entry is the model output
    and the second entry is the ground truth.
    """

    def __init__(self, model_output_dir, ground_truth_dset, transform=None):
        self.model_output_dir = model_output_dir
        self.ground_truth_dset = ground_truth_dset

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

        # sort output images and ground truth images
        # sanity check needed
        model_output_images_list = sorted(os.listdir(model_output_dir))
        self.model_output_images = [
            img for img in model_output_images_list if img.endswith(".png") and img.startswith("sample")
        ]
        

        # allow for only subset
        if len(self.model_output_images) < len(self.ground_truth_dset):
            print(
                f"Warning: Model output images less than ground truth images. Truncating ground truth images."
            )
            self.ground_truth_dset = torch.utils.data.Subset(
                self.ground_truth_dset, range(len(self.model_output_images))
            )

    def __len__(self):
        return len(self.model_output_images)

    def __getitem__(self, idx):
        model_output_image_path = os.path.join(
            self.model_output_dir, self.model_output_images[idx]
        )

        model_output_image = Image.open(model_output_image_path).convert("RGB")
        ground_truth_image = self.ground_truth_dset[idx][0]

        if self.transform is not None:
            model_output_image = self.transform(model_output_image)
            # ground_truth_image = self.transform(ground_truth_image)

        return model_output_image, ground_truth_image


def create_eval_dataloader(
    model_output_dir, ground_truth_dset, transform=None, batch_size=2
):
    """
    Creates a data loader for evaluation.  
    
    Args:
      model_output_dir: str, the directory containing the model output images.
      ground_truth_dset: torch.utils.data.Dataset, the ground truth dataset.
      transform: torchvision.transforms, the transformation to apply to the images.
      batch_size: int, the batch size for the data loader.

    Returns:
      torch.utils.data.DataLoader, the data loader for evaluation.
    """

    dataset = ZippedDataset(model_output_dir, ground_truth_dset, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return data_loader


def _compute_recon_metrics(
    workdir,
    method_name,
    task_name,
    dataset_name,
    dataset_path,
    batch_size=2,
    transform=None,
    eval_folder="eval_samples",
):
    # path to the model output
    model_output_dir = os.path.join(
        workdir, eval_folder, dataset_name, method_name, task_name
    )
    
    # true data set
    true_dset = lmdb_dataset.get_dataset(
        name=dataset_name,
        db_path=dataset_path,
        transform=None,  
    )
    
    # create data loader
    eval_loader = create_eval_dataloader(
        model_output_dir=model_output_dir,
        ground_truth_dset=true_dset,
        transform=transform,
        batch_size=batch_size,
    )
    
    # compute metrics
    psnr = get_psnr(eval_loader)
    ssim = get_ssim(eval_loader)
    lpips = get_lpips(eval_loader)
    
    # convert to torch tensor and compute mean
    mean_psnr = torch.cat(psnr).mean()
    mean_ssim = torch.cat(ssim).mean()
    mean_lpips = torch.cat(lpips).mean()
    
    # save metrics
    torch.save(psnr, os.path.join(model_output_dir, "psnr.pt"))
    torch.save(ssim, os.path.join(model_output_dir, "ssim.pt"))
    torch.save(lpips, os.path.join(model_output_dir, "lpips.pt"))
    
    # write to txt
    with open(os.path.join(model_output_dir, "metrics.txt"), "w") as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"PSNR: {mean_psnr}\n")
        f.write(f"SSIM: {mean_ssim}\n")
        f.write(f"LPIPS: {mean_lpips}\n")


def compute_recon_metrics(config, workdir, eval_folder):
    return _compute_recon_metrics(
        workdir=workdir,
        method_name=config.sampling.gudiance_method,
        task_name=config.degredation.task_name,
        dataset_name=config.data.name,
        dataset_path=config.data.lmdb_file_path,
        batch_size=config.sampling.batch_size,
        transform=None,
        eval_folder=eval_folder,
    )

# FLAGS = flags.FLAGS

# config_flags.DEFINE_config_file(
#     "config", None, "Path to the method configuration file."
# )

# flags.DEFINE_string("workdir", "", "Work directory.")

# flags.DEFINE_string(
#     "eval_folder", "eval_samples", "The folder name for storing evaluation results"
# )

# def main(argv):
#     tf.io.gfile.makedirs(FLAGS.workdir)
#     # Set logger so that it outputs to both console and file
#     gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
#     handler = logging.StreamHandler(gfile_stream)
#     formatter = logging.Formatter(
#         "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
#     )
#     handler.setFormatter(formatter)
#     logger = logging.getLogger()
#     logger.addHandler(handler)
#     logger.setLevel("INFO")
    
#     compute_recon_metrics(
#         workdir=FLAGS.workdir,
#         method_name=FLAGS.config.sampling.gudiance_method,
#         task_name=FLAGS.config.degredation.task_name,
#         dataset_name=FLAGS.config.data.name,
#         dataset_path=FLAGS.config.data.lmdb_file_path,
#         batch_size=FLAGS.config.sampling.batch_size,
#         eval_folder=FLAGS.eval_folder,
#     )
    
