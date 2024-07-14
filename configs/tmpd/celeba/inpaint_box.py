"""Configuration for box-masked inpainting on CelebA using TMPD."""
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..', '..', '..'))
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project directory to the sys.path
sys.path.append(project_dir)
sys.path.append(config_dir)

import numpy as np

import ml_collections

from physics.create_mask import load_mask
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
    degredation.name = "inpainting"
    degredation.task_name = "inpainting_box"
    degredation.channels = 3
    degredation.img_dim = 256
    degredation.noiser = "gaussian"
    # degredation.sigma = 0.05
    degredation.sigma = 0.0
    degredation.device = config.device
    
    # load mask from masks/
    mask_path = "masks/square_box_mask.npz"
    degredation.missing_indices = load_mask(mask_path, device=config.degredation.device)[1]
    
    # sampling config
    sampling = config.sampling
    sampling.gudiance_method = "tmpd"
    sampling.use_ode_sampler = "euler"
    sampling.clamp_to = 1 # gradient clipping for the guidance
    sampling.batch_size = 2
    sampling.sample_N = 50 # NOTE: tune this
    sampling.sigma_variance = 0.0 # NOTE: tune this add noise and denoise?
    # does flow models denoise? can it go off the data manifold?
    sampling.starting_time = 0
    return config
    
    

