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
        ret.append(peak_signal_noise_ratio(output, truth, reduction="none").to("cpu"))

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
        ret.append(
            structural_similarity_index_measure(output, truth, reduction="none").to(
                "cpu"
            )
        )
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

    def __init__(
        self, model_output_dir, ground_truth_dset, transform=None, random_indices=None
    ):
        self.model_output_dir = model_output_dir
        self.ground_truth_dset = ground_truth_dset

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

        # sort output images and ground truth images
        # sanity check needed
        model_output_images_list = os.listdir(model_output_dir)

        self.model_output_images = [
            img
            for img in model_output_images_list
            if img.endswith(".png") and img.startswith("sample")
        ]
        # sort output according to the indices
        self.model_output_images = sorted(
            self.model_output_images, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        # allow for only subset
        if len(self.model_output_images) < len(self.ground_truth_dset):
            print(
                f"Warning: Model output images less than ground truth images. Truncating ground truth images."
            )
            if random_indices is not None:
                self.ground_truth_dset = torch.utils.data.Subset(
                    self.ground_truth_dset, random_indices
                )
            else:
                raise ValueError(
                    "Sample indices must be provided when model output images are less than ground truth images."
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
    model_output_dir,
    ground_truth_dset,
    transform=None,
    batch_size=2,
    random_indices=None,
):
    """
    Creates a data loader for evaluation.

    Args:
      model_output_dir: str, the directory containing the model output images.
      ground_truth_dset: torch.utils.data.Dataset, the ground truth dataset.
      transform: torchvision.transforms, the transformation to apply to the images.
      batch_size: int, the batch size for the data loader.
      random_indices: list, the indices to sample from the ground truth dataset.

    Returns:
      torch.utils.data.DataLoader, the data loader for evaluation.
    """

    dataset = ZippedDataset(
        model_output_dir,
        ground_truth_dset,
        transform=transform,
        random_indices=random_indices,
    )

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
    batch_size=4,
    transform=None,
    noise_std=0.0,
    model_output_dir="",
    random_indices=None,
    additional_params=None,
):
    # path to the model output
    # model_output_dir = os.path.join(
    #     workdir, eval_folder, dataset_name, method_name, task_name, f"{noise_std}"
    # )

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
        random_indices=random_indices,
    )

    # compute metrics
    psnr = get_psnr(eval_loader)
    ssim = get_ssim(eval_loader)
    lpips = get_lpips(eval_loader)

    # convert to torch tensor and compute mean
    mean_psnr = torch.stack(psnr).mean()
    mean_ssim = torch.stack(ssim).mean()
    mean_lpips = torch.stack(lpips).mean()

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

    # get additional params
    starting_time = additional_params.get("starting_time", 0.0)
    sample_N = additional_params.get("nfe", 100)
    # clamp_to = additional_params.get("clamp_to", 1.0)
    gmres_max_iter = additional_params.get("gmres_max_iter", 100)

    # create txt file in top dir if not present
    aggregate_path = os.path.join(
        workdir, f"{dataset_name}_{task_name}_{noise_std}_aggregated_metrics.txt"
    )
    if not os.path.exists(os.path.join(workdir, aggregate_path)):
        with open(os.path.join(workdir, aggregate_path), "w") as f:
            # create file
            f.write(
                "Method, Task, Dataset, starting_time, sample_N, gmres_max_iter, PSNR, SSIM, LPIPS\n"
            )
            f.write(
                f"{method_name}, {task_name}, {dataset_name}, {starting_time}, {sample_N}, {gmres_max_iter}, {mean_psnr}, {mean_ssim}, {mean_lpips}\n"
            )
    else:
        with open(os.path.join(workdir, aggregate_path), "a") as f:
            # append to file
            f.write(
                f"{method_name}, {task_name}, {dataset_name}, {starting_time}, {sample_N}, {gmres_max_iter}, {mean_psnr}, {mean_ssim}, {mean_lpips}\n"
            )


def compute_recon_metrics(
    config, 
    workdir, 
    model_output_dir,
    noise_std,
    random_indices=None,
    additional_params=None
):

    return _compute_recon_metrics(
        workdir=workdir,
        method_name=config.sampling.gudiance_method,
        task_name=config.degredation.task_name,
        dataset_name=config.data.name,
        dataset_path=config.data.lmdb_file_path,
        batch_size=config.sampling.batch_size,
        transform=None,
        noise_std=noise_std,
        model_output_dir=model_output_dir,
        random_indices=random_indices,
        additional_params=additional_params,
    )


def _get_best_config_df(df):
    # Normalize the metrics (SSIM and PSNR should be maximized, LPIPS minimized)
    df['PSNR_norm'] = df['PSNR'] / df['PSNR'].max()
    df['SSIM_norm'] = df['SSIM'] / df['SSIM'].max()
    df['LPIPS_norm'] = df['LPIPS'].min() / df['LPIPS']  # Lower LPIPS is better, so inverse

    df['combined_score'] = df['PSNR_norm'] + df['SSIM_norm'] + df['LPIPS_norm']


    best_row = df.loc[df['combined_score'].idxmax()]

    return best_row

# def _get_best_config(workdir, dataset_name, task_name, noise_std, method_name):
#     # Read the aggregated metrics file
#     df = pd.read_csv(os.path.join(workdir, f"{dataset_name}_{task_name}_{noise_std}_aggregated_metrics.txt"))

#     best_row = _get_best_config_df(df)
    
#     # write to txt
#     best_param_path =  f"{method_name}_best_params.txt"
#     if not os.path.exists(os.path.join(workdir, best_param_path)):
#         with open(os.path.join(workdir, best_param_path), "w") as f:
#             f.write(best_row.to_string())
#     else:
#         with open(os.path.join(workdir, best_param_path), "a") as f:
#             f.write(best_row.to_string())

#     return best_row


# def get_best_config(config, workdir, noise_std, additional_params=None):
#     return _get_best_config(
#         workdir=workdir,
#         dataset_name=config.data.name,
#         task_name=config.degredation.task_name,
#         noise_std=noise_std,
#         method_name=config.sampling.gudiance_method,
#     )




FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", "", "Work directory.")
config_flags.DEFINE_config_file(
    "config", None, "Sampling configuration.", lock_config=False  # might want to lock
)

def main(argv):
    method = FLAGS.config.sampling.gudiance_method
    data_name = FLAGS.config.data.name
    # task_name = FLAGS.config.degredation.task_name
    # noise_std = FLAGS.config.data.noise_std
    
    # list all the txt results in workdir
    workdir = FLAGS.workdir
    all_txt = os.listdir(workdir)
    all_tx = [f for f in all_txt if f.endswith(".txt") and f.startswith(f"{data_name}_")]
    
    # write to txt
    best_param_path =  f"{method}_best_params.txt"
    for txt_file in all_tx:
        df = pd.read_csv(os.path.join(workdir, txt_file))
        best_row = _get_best_config_df(df)
        if not os.path.exists(os.path.join(workdir, best_param_path)):
            with open(os.path.join(workdir, best_param_path), "w") as f:
                f.write(best_row.to_string())
        else:
            with open(os.path.join(workdir, best_param_path), "a") as f:
                f.write(best_row.to_string())
    