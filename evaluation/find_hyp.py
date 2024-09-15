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



def _get_best_config_df(df):
    # Normalize the metrics (SSIM and PSNR should be maximized, LPIPS minimized)
    df['PSNR_norm'] = df['PSNR'] / df['PSNR'].max()
    df['SSIM_norm'] = df['SSIM'] / df['SSIM'].max()
    df['LPIPS_norm'] = df['LPIPS'].min() / df['LPIPS']  # Lower LPIPS is better, so inverse

    df['combined_score'] = df['PSNR_norm'] + df['SSIM_norm'] + df['LPIPS_norm']


    best_row = df.loc[df['combined_score'].idxmax()]

    return best_row




FLAGS = flags.FLAGS

# config_flags.DEFINE_config_file(
#     "config", None, "Sampling configuration.", lock_config=False  # might want to lock
# )
# flags.DEFINE_string("workdir", "", "Work directory.")
flags.DEFINE_string("method", "", "Guidance method.")
flags.DEFINE_string("data_name", "", "Data name.")

def main(argv):
    # method = FLAGS.config.sampling.gudiance_method
    data_name = FLAGS.data_name
    method = FLAGS.method
    # task_name = FLAGS.config.degredation.task_name
    # noise_std = FLAGS.config.data.noise_std
    
    # list all the txt results in workdir
    workdir = ""
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
    