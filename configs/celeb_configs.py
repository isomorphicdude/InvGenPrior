# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training rectified Flow on CelebA HQ."""

import torch
import ml_collections

from default_lsun_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "rectified_flow"
    training.continuous = False
    training.reduce_mean = True
    training.snapshot_freq = 100000
    training.data_dir = "DATA_DIR"

    # sampling
    sampling = config.sampling
    sampling.method = "rectified_flow"
    sampling.init_type = "gaussian"
    sampling.init_noise_scale = 1.0
    
    # configure the guided sampler for inverse problems
    sampling.use_ode_sampler = "euler" # changed for inverse problems
    sampling.batch_size = 1
    
    # sampling starting time, 0.2, 0.4 in the paper
    sampling.starting_time = 0
    # noise to add during sampling
    sampling.sigma_variance = 1.0 # NOTE: tune this add noise and denoise?
    
    # number of steps to run the sampler
    sampling.sample_N = 50 # NOTE: tune this
    sampling.clamp_to = 1.0 # gradient clipping for the guidance
    
    # inverse problem settings
    sampling.degredation_sigma = 0.1
    sampling.degredation_noiser = "gaussian"
    
    # data
    data = config.data
    data.dataset = "CelebA-HQ-Pytorch"
    data.centered = True

    # model
    model = config.model
    model.name = "ncsnpp"
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "output_skip"
    model.progressive_input = "input_skip"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3
    

    return config


# print(get_config())
