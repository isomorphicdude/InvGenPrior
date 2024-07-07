"""Configuration for inpainting on CelebA using TMPD."""

import numpy as np

import ml_collections

from physics.create_mask import load_mask
from ...celeb_configs import get_config as get_celeb_config

def get_config():
    config = get_celeb_config()
    
    # degredation config
    config.degredation = degredation = ml_collections.ConfigDict()
    degredation.name = "inpainting"
    degredation.channels = 3
    degredation.img_dim = 256
    degredation.noiser = "gaussian"
    degredation.sigma = 0.05
    
    # load mask from masks/
    mask_path = "masks/square_box_mask.npz"
    degredation.missing_indices = load_mask(mask_path, device=config.device)[1]
    
    # sampling config
    sampling = config.sampling
    sampling.gudiance_method = "tmpd"
    sampling.use_ode_sampler = "euler"
    sampling.clamp_to = 1 # gradient clipping
    sampling.batch_size = 1 
    
    
    
    
    
