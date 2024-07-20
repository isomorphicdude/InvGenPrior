"""Configuration for Gaussian deblurring on CelebA using TMPD."""
import os
import sys

import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..', '..', '..'))
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project directory to the sys.path
sys.path.append(project_dir)
sys.path.append(config_dir)

import numpy as np

import ml_collections

from celeb_configs import get_config as get_celeb_config

def get_config():
    config = get_celeb_config()
    
    # ckpt name
    config.ckpt_name = "celebA_ckpt.pth"
    
    # data config (for creating degraded images)
    data = config.data
    data.name = "celeba"
    data.lmdb_file_path = "data/celeba-hq/val.lmdb"
    data.split_name = "val"
    
    # degredation config
    config.degredation = degredation = ml_collections.ConfigDict()
    degredation.name = "deblurring"
    degredation.task_name = degredation.name
    degredation.channels = 3
    degredation.img_dim = 256
    degredation.noiser = "gaussian"
    # degredation.sigma = 0.05
    degredation.sigma = config.sampling.degredation_sigma
    degredation.device = config.device
    
    # if using torch implementation
    degredation.kernel_size = 61
    degredation.intensity = 4.0
    
    # if using custom implementation from NVlabs
    def pdf(x, sigma=10):
        """Gaussian PDF."""
        return torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2]))
    sigma = 4
    window = 9
    kernel = torch.Tensor([pdf(t, sigma) for t in range(-(window-1)//2, (window-1)//2)])
    degredation.kernel = kernel / kernel.sum()
    
    # sampling config
    sampling = config.sampling
    sampling.gudiance_method = "tmpd"
    sampling.use_ode_sampler = "euler"
    sampling.clamp_to = 1 # gradient clipping
    sampling.batch_size = 2 
    sampling.sample_N = 50
    
    return config
    
    

