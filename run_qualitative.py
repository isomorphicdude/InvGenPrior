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


def create_and_compare(config, workdir, data_index=53, noise_sigma=0.05,
                       sample_N=100, sampling_var=0.1, clamp_to=1.0,
                       starting_time= 0.2,
                       use_svd=False,
                       max_iter=3, compare_iter=False):
    """
    Creates a result for each method and compare them.
    """
    # create a different config for each sampler
    # comment out tmpd and reddiff
    # ignore_list = [ # here only use pgdm for testing the thresholding
    #     "bures_jko",
    #     "tmpd",
    #     "reddiff",
    #     "tmpd_og",
    #     "dps",
    #     "tmpd_fixed_diag",
    #     "tmpd_ablation",
    #     "tmpd_fixed_cov",
    #     "tmpd_exact",
    #     "tmpd_d",
    #     "tmpd_cd",
    #     "pgdm_mod",
    #     "tmpd_row_exact",
    #     "tmpd_trace",
    #     "true_vec"
    # ]
    ignore_list = [
        "bures_jko",
        "tmpd_fixed_diag",
        "tmpd_ablation",
        "true_vec",
        "tmpd_fixed_cov",
        "tmpd_exact",
        "tmpd_d",
        "tmpd_cd",
        "pgdm_mod",
        "tmpd_row_exact",
        "tmpd_trace",
        "tmpd_h",
        "tmpd",
        "tmpd_h_ablate",
        "dps",
        "tmpd_gmres",
        "tmpd_gmres_ablate",
        # "reddiff",
        "tmpd_og",
        # "pgdm",
    ]
    config_keys = [
        sampler_name
        for sampler_name in __GUIDED_SAMPLERS__
        if sampler_name not in ignore_list
    ]
    # print("Available samplers: ", config_keys)

    configs_copies = {sampler_name: None for sampler_name in config_keys}

    # logging.info("Creating configs for each sampler")
    print("Available samplers: ", list(configs_copies.keys()))

    for sampler_name in config_keys:
        # print(f"Creating config for {sampler_name}")
        new_config = ml_collections.ConfigDict()
        new_config = config
        new_config.sampling.guidance_method = sampler_name

        if sampler_name == "reddiff":
            new_config.sampling.clamp_to = None

        configs_copies[sampler_name] = new_config

    # create a folder for saving the results
    eval_dir = os.path.join(
        workdir,
        "qualitative_eval",
        config.data.name,
        config.degredation.task_name,
        "sigma_" + str(noise_sigma),
        "image_" + str(data_index)
    )
    tf.io.gfile.makedirs(eval_dir)

    ### below is shared for all samplers ###

    # use a dataset but we choose an index
    dset = lmdb_dataset.get_dataset(
        name=config.data.name,
        db_path=config.data.lmdb_file_path,
        transform=None,  # overridden by child class
    )
    logging.info(f"Using dataset {config.data.name}.")
    data_index = int(data_index) if not isinstance(data_index, int) else data_index
    # logging.info(f"Sampling image number {data_index}.")

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
            sample_N=sample_N,  # number of steps, here does not defined by the config
        )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # build degredation operator
    H_func = get_operator(name=config.degredation.name, config=config.degredation)
    
    # option to change the noise sigma
    noise_config = ml_collections.ConfigDict()
    noise_config.sigma = noise_sigma
    noise_config.device = config.device
    noiser = get_noise(name=config.degredation.noiser, 
                       config=noise_config)

    # get the image
    true_img = dset[data_index][0]
    true_img = true_img.unsqueeze(0)
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
        1,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )

    # common initializations
    start_z = torch.randn(sampling_shape).to(config.device)

    for sampler_name in configs_copies.keys():
        logging.info(f"Sampling using {sampler_name} guided sampler.")

        guided_sampler = get_guided_sampler(
            name=sampler_name,
            model=score_model,
            sde=sde,
            shape=sampling_shape,
            inverse_scaler=inverse_scaler,
            H_func=H_func,
            noiser=noiser,
            device=config.device,
            sampling_eps=sampling_eps,
        )
        # dumping the config setting into a txt
        with open(os.path.join(eval_dir, f"{sampler_name}_config.txt"), "w") as f:
            f.write(f"{config}\n")

        # run the sampler
        score_model.eval()

        start_time = time.time()
        y_obs = y_obs.clone().detach()

        # pass to guided sampler
        # default is to clamp to 1.0
        if sampler_name == "reddiff":
            clamp_to = None
        else:
            clamp_to = clamp_to

        # fix noise during sampling
        current_sample = guided_sampler.sample(
            y_obs=y_obs,
            z=start_z,  # maybe can use latent encoding
            return_list=False,
            method=config.sampling.use_ode_sampler,  # euler by default
            clamp_to=clamp_to,
            starting_time=starting_time,
            new_noise = torch.randn_like(y_obs),
            data_name = config.data.name,
            use_svd = use_svd,
            gmres_max_iter = max_iter,
            num_hutchinson_samples = max_iter
        )

        # save
        if (sampler_name == "tmpd_cg" or sampler_name == "tmpd_gmres_ablate") and compare_iter:
            save_image(current_sample, os.path.join(eval_dir, f"{sampler_name}_sample_{max_iter}.png"))
        else:
            save_image(current_sample, os.path.join(eval_dir, f"{sampler_name}_sample.png"))

        end_time = time.time()
        logging.info(f"Sampling took {end_time - start_time} seconds.")

        # clear memory
        torch.cuda.empty_cache()

    logging.info("Sampling Done, saving comparison image.")

    if not compare_iter:
        # plot the images side by side
        fig, axs = plt.subplots(1, len(configs_copies.keys()) + 2, figsize=(20, 10))
        axs[0].imshow(plt.imread(os.path.join(eval_dir, "true_image.png")))
        axs[0].set_title("True Image")
        axs[0].axis("off")
        axs[1].imshow(plt.imread(os.path.join(eval_dir, "degraded_image.png")))
        axs[1].set_title("Degraded Image")
        axs[1].axis("off")
        for i, sampler_name in enumerate(configs_copies.keys()):
            axs[i + 2].imshow(
                plt.imread(os.path.join(eval_dir, f"{sampler_name}_sample.png"))
            )
            axs[i + 2].set_title(f"{sampler_name.upper()} Sample")
            axs[i + 2].axis("off")

        plt.savefig(os.path.join(eval_dir, "comparison.png"))
        # plt.close()
        logging.info("Comparison image saved.")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Sampling configuration.", lock_config=False  # might want to lock
)

flags.DEFINE_integer("data_index", 53, "Index of the data to sample.")

flags.DEFINE_integer("sample_N", 100, "Number of sampling steps.")

flags.DEFINE_float("noise_sigma", 0.05, "Noise sigma for the degradation.")

flags.DEFINE_float("sampling_var", 0.1, "Sampling variance.")

flags.DEFINE_float("clamp_to", 1.0, "Clamp to value.")

flags.DEFINE_float("starting_time", 0.0, "Starting time for the sampler.")

flags.DEFINE_integer("max_iter", 3, "Maximum number of iterations.")

flags.DEFINE_string("workdir", "InvGenPrior", "Work directory.")

flags.DEFINE_string(
    "eval_folder", "eval_samples", "The folder name for storing evaluation results"
)

flags.DEFINE_bool("use_svd", False, "Use SVD for the guided sampler.")

flags.DEFINE_bool("compare_iter", False, "Compare the results at different iterations.")

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
    torch.manual_seed(0)
    np.random.seed(0)
    # data_index_list = [2, 15, 53, 109]
    # for data_index in data_index_list:
        # logging.info(f"\nSampling for data index {data_index}.\n")
    if FLAGS.clamp_to == 0.0:
        FLAGS.clamp_to = None
        
    create_and_compare(
        FLAGS.config,
        FLAGS.workdir,
        data_index=FLAGS.data_index,
        noise_sigma=FLAGS.noise_sigma,
        sample_N=FLAGS.sample_N,
        sampling_var=FLAGS.sampling_var,
        clamp_to=FLAGS.clamp_to,
        use_svd=FLAGS.use_svd,
        max_iter=FLAGS.max_iter,
        starting_time=FLAGS.starting_time,
        compare_iter=FLAGS.compare_iter
    )


if __name__ == "__main__":
    app.run(main)
