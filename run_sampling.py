"""
Implements the sampling script to generate samples and save to a dataset.

First observations are loaded from a dataset and passed to degredation operators.
The guided sampler is then used to generate samples.

This has not been optimised for multiple GPUs.
"""

import gc
import io
import os
import time

import torch
from torchvision.utils import save_image
import logging
import numpy as np
import tensorflow as tf
from absl import app, flags
from ml_collections.config_flags import config_flags
import matplotlib.pyplot as plt


# flow models
from models import sde_lib, losses, ddpm, ncsnv2, ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage

# data
from datasets import lmdb_dataset

# inverse problems
from physics.operators import get_operator
from physics.noisers import get_noise
from guided_samplers import tmpd, dps, pgdm
from guided_samplers.registry import get_guided_sampler


logging.basicConfig(level=logging.INFO)


def create_samples(config, workdir, save_degraded=True, eval_folder="eval_samples"):
    """
    Create samples using the guided sampler.

    Args:
      - config: configuration file, used for ml_collections
      - workdir: working directory, usually just the root directory of repo
      - save_degraded: whether to save the degraded images
      - eval_folder: folder to save the samples, should be a combination of the 
        name of the experiment and the method used to generate the samples
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    # create data
    dset = lmdb_dataset.get_dataset(
        name = config.data.name,
        db_path= config.data.lmdb_file_path,
        transform=None,  # overridden by child class
    )

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config.sampling.batch_size,
        shuffle=False,
        # num_workers=config.data.num_workers,
    )

    # scaler and inverse ([-1, 1] and [0, 1])
    scaler = lmdb_dataset.get_data_scaler(config)
    inverse_scaler = lmdb_dataset.get_data_inverse_scaler(config)

    # Initialise model
    score_model = mutils.create_model_no_parallel(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # load weights
    ckpt_path = os.path.join(checkpoint_dir, config.ckpt_name)
    
    if not os.path.exists(ckpt_path):
        logging.error(f"Checkpoint {ckpt_path} does not exist.")
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    
    state = mutils.restore_checkpoint(ckpt_path, state, device=config.device)

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

    # build degredation operator
    H_func = get_operator(name=config.degredation.name, config=config.degredation)
    noiser = get_noise(name=config.degredation.noiser, config=config.degredation)

    # build sampling function
    sampling_shape = (
        config.sampling.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )

    guided_sampler = get_guided_sampler(
        name=config.sampling.gudiance_method,
        model=score_model,
        sde=sde,
        shape=sampling_shape,
        inverse_scaler=inverse_scaler,
        H_func=H_func,
        noiser=noiser,
        device=config.device,
        sampling_eps=sampling_eps,
    )

    # begin sampling
    score_model.eval()
    logging.info(f"Dataset size is {len(data_loader.dataset)}")

    start_time = time.time()
    for iter_no, (batched_img, img_idx) in enumerate(data_loader):
        logging.info(
            f"Sampling a batch of {config.sampling.batch_size} image {iter_no}"
        )

        # apply scaler
        batched_img = scaler(batched_img)

        # apply degredation operator
        y_obs = H_func.H(batched_img)

        # apply noiser
        y_obs = noiser(y_obs)
        
        # if save the degraded images then return the re-shaped
        if save_degraded:
            y_obs_image = H_func.get_degraded_image(batched_img)
            y_obs_image = noiser(y_obs_image)
            # apply scaler
            y_obs_image = inverse_scaler(y_obs_image)

        # pass to guided sampler
        batched_samples = guided_sampler.sample(
            y_obs=y_obs,
            z=None,  # maybe can use latent encoding
            return_list=False,
            method = config.sampling.use_ode_sampler, # euler or rk45
            # method="euler",
            clamp_to = config.sampling.clamp_to,
            # clamp_to=1,
        )
        print(batched_samples.shape)

        # save the images to eval folder
        logging.info(f"Current batch finished. Saving images...")
        for j in range(config.sampling.batch_size):
            img = batched_samples[j]
            # img = inverse_scaler(img) # already included in sampler
            save_image(
                img,
                os.path.join(eval_dir, f"{iter_no}_{j}.png"),
                # normalize=True,
                # range=(-1, 1),
            )
            
        if save_degraded:
            logging.info(f"Saving degraded images...")
            for j in range(config.sampling.batch_size):
                img = y_obs_image[j]
                # img = inverse_scaler(img) # already included in sampler
                save_image(
                    img,
                    os.path.join(eval_dir, f"{iter_no}_{j}_degraded.png"),
                    # normalize=True,
                    # range=(-1, 1),
                )
                
    # clear memory  
    torch.cuda.empty_cache()
    

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Sampling configuration.", lock_config=False # might want to lock
)

flags.DEFINE_string("workdir", "InvGenPrior", "Work directory.")

flags.DEFINE_string(
    "eval_folder", "eval_samples", "The folder name for storing evaluation results"
)

flags.mark_flag_as_required("config")

# TODO: separate the main and the runlib
def main(argv):
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel("INFO")
    
    create_samples(
        FLAGS.config,
        FLAGS.workdir,
        save_degraded=True,
        eval_folder=FLAGS.eval_folder,
    )
    
if __name__ == "__main__":
    app.run(main)