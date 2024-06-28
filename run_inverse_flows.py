"""Testing ideas for inverse problems."""

import os
import logging
import numpy as np
import torchvision
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import datasets
from models import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from absl import flags
from ml_collections.config_flags import config_flags

import torch
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)

# flags.DEFINE_string("workdir", None, "Work directory.")

config = FLAGS.config

# Initialize model
score_model = mutils.create_model(config)

optimizer = losses.get_optimizer(config, score_model.parameters())

ema = ExponentialMovingAverage(
    score_model.parameters(), decay=config.model.ema_rate
)
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

# checkpoint_dir = os.path.join(workdir, "checkpoints")
checkpoint_dir = "checkpoints"


scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)


 # Setup SDEs
if config.training.sde.lower() == "rectified_flow":
    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling.init_type,
        noise_scale=config.sampling.init_noise_scale,
        use_ode_sampler=config.sampling.use_ode_sampler,
        sigma_var=config.sampling.sigma_variance,
        ode_tol=config.sampling.ode_tol,
        sample_N=config.sampling.sample_N,
    )
    sampling_eps = 1e-3
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")



sampling_shape = (
        config.eval.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )

sampling_fn = sampling.get_sampling_fn(
    config, sde, sampling_shape, inverse_scaler, sampling_eps
)

# load model checkpoints
ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(10))
state = restore_checkpoint(ckpt_path, state, device=config.device)

samples, n = sampling_fn(score_model)
this_sample_dir = "samples"

torchvision.utils.save_image(
    samples.clamp_(0.0, 1.0),
    os.path.join(this_sample_dir, "%d.png" % r),
    nrow=10,
    normalize=False,
)