"""Implements PSNR, SSIM, and LPIPS metrics for evaluation."""

import os

import logging
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    learned_perceptual_image_patch_similarity
)
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


# compute Peak Signal-to-Noise Ratio (PSNR)
def get_psnr(data_loader):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) for a given data loader.

    The data loader should return a pair of tensors,
    where the first tensor is the output and the second tensor is the truth.

    Using torchmetrics implementation.
    
    Args:  
        - data_loader: torch.utils.data.DataLoader, the data loader to evaluate.
        
    Returns:  
        - list of PSNR values as torch.Tensor.
    """
    ret = []
    for output, truth in tqdm(data_loader):
        # move to cuda
        output, truth = output.cuda(), truth.cuda()
        # assert image shape of N, C, H, W
        assert output.shape == truth.shape, f"Output shape {output.shape} != Truth shape {truth.shape}"
        assert output.shape[1] == 3, f"Output shape {output.shape} not RGB"
        ret.append(peak_signal_noise_ratio(output, truth, reduction=None))
        
    return ret
    


# compute Structural Similarity Index (SSIM)
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
        assert output.shape == truth.shape, f"Output shape {output.shape} != Truth shape {truth.shape}"
        assert output.shape[1] == 3, f"Output shape {output.shape} not RGB"
        ret.append(structural_similarity_index_measure(output, truth, reduction=None))
        
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
        assert output.shape == truth.shape, f"Output shape {output.shape} != Truth shape {truth.shape}"
        assert output.shape[1] == 3, f"Output shape {output.shape} not RGB"
        with torch.no_grad():
            lpip = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", # choose from ['alex', 'squeeze', 'vgg']
                normalize=True, # since I have applied inverse scaler in sampling
                reduction="mean",
                ).cuda()(output, truth)
            
        ret.append(lpip.detach().cpu())
        
    return ret


class ZippedDataset(torch.utils.data.Dataset):
    """
    Creates a zipped data set where the first entry is the model output
    and the second entry is the ground truth.
    """
    def __init__(self, model_output_dir, ground_truth_dir, transform=None):
        self.model_output_dir = model_output_dir
        self.ground_truth_dir = ground_truth_dir
        
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        
        self.transform = transform
        
        # sort output images and ground truth images
        # sanity check needed
        self.model_output_images = sorted(os.listdir(model_output_dir))
        self.ground_truth_images = sorted(os.listdir(ground_truth_dir))
        
        # allow for only subset
        if len(self.model_output_images) < len(self.ground_truth_images):
            print(f"Warning: Model output images less than ground truth images. Truncating ground truth images.")
            self.ground_truth_images = self.ground_truth_images[:len(self.model_output_images)]


    def __len__(self):
        return len(self.model_output_images)

    def __getitem__(self, idx):
        model_output_image_path = os.path.join(self.model_output_dir, self.model_output_images[idx])
        ground_truth_image_path = os.path.join(self.ground_truth_dir, self.ground_truth_images[idx])
        
        model_output_image = Image.open(model_output_image_path).convert("RGB")
        ground_truth_image = Image.open(ground_truth_image_path).convert("RGB")
        
        if self.transform:
            model_output_image = self.transform(model_output_image)
            ground_truth_image = self.transform(ground_truth_image)
        
        return model_output_image, ground_truth_image
    
    
def create_eval_dataloader(model_output_dir, ground_truth_dir, transform=None, batch_size=2):
    """
    Creates a data loader for evaluation.
    
    Returns:
        - torch.utils.data.DataLoader, the data loader for evaluation.
    """
    
    dataset = ZippedDataset(model_output_dir, ground_truth_dir, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader