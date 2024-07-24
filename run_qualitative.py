"""
Performs sampling using each of the method on the same inverse problem 
and compares the results. Ground truth and degraded image are also saved.
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
import ml_collections
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
from guided_samplers import tmpd, dps, pgdm, reddiff, bures_jko
from guided_samplers.registry import get_guided_sampler, __GUIDED_SAMPLERS__


def create_and_compare(config, workdir, data_index = 53):

    # create a different config for each sampler
    configs_copies = {
        sampler_name: None
        for sampler_name in __GUIDED_SAMPLERS__
        and sampler_name
        not in ["bures_jko", "tmpd_og", "tmpd_fixed_cov", "tmpd_exact", "tmpd_d"]
    }

    logging.info("Creating configs for each sampler")
    logging.info("Available samplers: ", configs_copies.keys())
    for sampler_name in __GUIDED_SAMPLERS__:
        # if sampler_name in ["bures_jko", "tmpd_og", "tmpd_fixed_cov", "tmpd_exact", "tmpd_d"]:
        #     # skip these samplers for now
        #     continue
        new_config = ml_collections.ConfigDict()
        new_config.update(config)
        new_config.sampling.guidance_method = sampler_name
        if sampler_name == "reddiff":
            new_config.sampling.clamp_to = None
        configs_copies[sampler_name] = new_config

    # create a folder for saving the results
    eval_dir = os.path.join(workdir, "qualitative_eval")
    tf.io.gfile.makedirs(eval_dir)

    ### below is shared for all samplers ###
    
    # use a dataset but we choose an index
    dset = lmdb_dataset.get_dataset(
        name=config.data.name,
        db_path=config.data.lmdb_file_path,
        transform=None,  # overridden by child class
    )
    logging.info(f"Using dataset {config.data.name}.")
    logging.info("Sampling the index: ", data_index)

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
    
    # get the image
    true_img = dset[data_index]
    true_img = true_img.to(config.device)
    
    # save true image
    save_image(true_img, os.path.join(eval_dir, "true_image.png"))
    
    # apply scaler for using the model
    true_img = scaler(true_img)
    # apply degredation operator
    y_obs = H_func.H(true_img)

    # apply noiser
    y_obs = noiser(y_obs)
    
    # save degraded image
    degraded_img = H_func.get_degraded_image(y_obs)
    degraded_img = inverse_scaler(degraded_img)
    save_image(degraded_img, os.path.join(eval_dir, "degraded_image.png"))

    # shared
    sampling_shape = (
        config.sampling.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )
    
    # common initializations
    start_z = torch.randn(sampling_shape).to(config.device)

    for sampler_name in configs_copies.keys():
        # get individual config
        current_config = configs_copies[sampler_name]
        guided_sampler = get_guided_sampler(
            name=current_config.sampling.gudiance_method,
            model=score_model,
            sde=sde,
            shape=sampling_shape,
            inverse_scaler=inverse_scaler,
            H_func=H_func,
            noiser=noiser,
            device=current_config.device,
            sampling_eps=sampling_eps,
        )
        # dumping the config setting into a txt
        with open(os.path.join(eval_dir, f"{sampler_name}_config.txt"), "w") as f:
            f.write(f"{current_config}")
            
        # run the sampler
        score_model.eval()
        logging.info(f"Using {sampler_name} guided sampler.")
        logging.info(f"Sampling {config.sampling.batch_size} images at a time.")
        
        start_time = time.time()
        y_obs = y_obs.clone().detach()
        
        # pass to guided sampler
        current_sample = guided_sampler.sample(
            y_obs=y_obs,
            z=start_z,  # maybe can use latent encoding
            return_list=False,
            method=config.sampling.use_ode_sampler,  # euler by default
            clamp_to=config.sampling.clamp_to,
            starting_time=config.sampling.starting_time,
        )
        
        # save
        save_image(current_sample, os.path.join(eval_dir, f"{sampler_name}_sample.png"))
        
        end_time = time.time()
        logging.info(f"Sampling took {end_time - start_time} seconds.")
        
        # clear memory
        torch.cuda.empty_cache()
        
    logging.info("Sampling Done.")
    
    # plot the images side by side
    fig, axs = plt.subplots(1, len(configs_copies.keys()) + 2, figsize=(20, 10))
    axs[0].imshow(plt.imread(os.path.join(eval_dir, "true_image.png")))
    axs[0].set_title("True Image")
    axs[1].imshow(plt.imread(os.path.join(eval_dir, "degraded_image.png")))
    axs[1].set_title("Degraded Image")
    for i, sampler_name in enumerate(configs_copies.keys()):
        axs[i+2].imshow(plt.imread(os.path.join(eval_dir, f"{sampler_name}_sample.png")))
        axs[i+2].set_title(f"{sampler_name} Sample")
        
    plt.savefig(os.path.join(eval_dir, "comparison.png"))
        

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Sampling configuration.", lock_config=False  # might want to lock
)

flags.DEFINE_integer("data_index", 53, "Index of the data to sample.")

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
    
    # run
    create_and_compare(
        FLAGS.config,
        FLAGS.workdir,
        data_index=FLAGS.data_index,
    )


if __name__ == "__main__":
    app.run(main)
