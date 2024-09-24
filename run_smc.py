"""
Implements the sampling script for the SMC sampler.
"""

import gc
import io
import os
import time

import yaml
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
from models import utils as mutils
from models import sde_lib, dps_unet

# data
from datasets import lmdb_dataset

# inverse problems
from physics.operators import get_operator
from physics.noisers import get_noise
from guided_samplers import smcdiffopt
from guided_samplers.registry import get_guided_sampler




def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def create_and_compare(config, workdir, data_index=53, noise_sigma=0.05,
                       sample_N=100, clamp_to=1.0,
                       model_yaml="configs/dps_ffhq.yaml",
                       data_name="ffhq256",
                       num_particles=10):
    """
    Creates a result for each method and compare them.
    """

    # create a folder for saving the results
    eval_dir = os.path.join(
        workdir,
        "qualitative_eval",
        data_name,
        config.degredation.task_name,
        "sigma_" + str(noise_sigma),
        "image_" + str(data_index)
    )
    tf.io.gfile.makedirs(eval_dir)

    ### below is shared for all samplers ###

    # use a dataset but we choose an index
    dset = lmdb_dataset.get_dataset(
        name=data_name,
        db_path="data/ffhq/val.lmdb",
        transform=None,  # overridden by child class
    )
    logging.info(f"Using dataset {data_name}.")
    data_index = int(data_index) if not isinstance(data_index, int) else data_index
    # logging.info(f"Sampling image number {data_index}.")

    # scaler and inverse ([-1, 1] and [0, 1])
    scaler = lmdb_dataset.get_data_scaler(config)
    inverse_scaler = lmdb_dataset.get_data_inverse_scaler(config)

    # Initialise model
    model_config = load_yaml(model_yaml)
    eps_model = dps_unet.create_model(**model_config)
    eps_model = eps_model.to(config.device)
    eps_model.eval()

    # set up SDE, use default parameters    
    sde = sde_lib.VPSDE(N=sample_N)
    sampling_eps = 1e-3

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
    
    sampler_name = "smcdiffopt"

    logging.info(f"Sampling using {sampler_name} guided sampler.")

    guided_sampler = get_guided_sampler(
        name=sampler_name,
        model=eps_model,
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
    eps_model.eval()

    start_time = time.time()
    y_obs = y_obs.clone().detach()
    

    # fix noise during sampling
    current_sample, _ = guided_sampler.sample(
        y_obs=y_obs,
        z=start_z,  # maybe can use latent encoding
        return_list=True,
        method=config.sampling.use_ode_sampler,  # euler by default
        clamp_to=1.0,
        data_name = data_name,
        num_particles=num_particles,
    )

    # save
    save_image(current_sample[-1], os.path.join(eval_dir, f"{sampler_name}_sample.png"))
    
    # save the list of images
    for i, img in enumerate(current_sample):
        save_image(img, os.path.join(eval_dir, f"{sampler_name}_sample_{i}.png"))

    end_time = time.time()
    logging.info(f"Sampling took {end_time - start_time} seconds.")

    # clear memory
    torch.cuda.empty_cache()

    logging.info("Sampling Done, saving comparison image.")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(plt.imread(os.path.join(eval_dir, "true_image.png")))
    axs[0].set_title("True Image")
    axs[0].axis("off")
    axs[1].imshow(plt.imread(os.path.join(eval_dir, "degraded_image.png")))
    axs[1].set_title("Degraded Image")
    axs[1].axis("off")
    
    axs[2].imshow(
        plt.imread(os.path.join(eval_dir, f"{sampler_name}_sample.png"))
    )
    axs[2].set_title(f"{sampler_name.upper()} Sample")
    axs[2].axis("off")

    plt.savefig(os.path.join(eval_dir, "comparison.png"))
    logging.info("Comparison image saved.")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Sampling configuration.", lock_config=False  # might want to lock
)

flags.DEFINE_integer("data_index", 53, "Index of the data to sample.")

flags.DEFINE_integer("sample_N", 100, "Number of sampling steps.")

flags.DEFINE_float("noise_sigma", 0.05, "Noise sigma for the degradation.")

flags.DEFINE_float("clamp_to", 1.0, "Clamp to value.")

flags.DEFINE_integer("num_particles", 10, "Number of particles for the sampler.")


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
    torch.manual_seed(0)
    np.random.seed(0)
    
    if FLAGS.clamp_to == 0.0:
        FLAGS.clamp_to = None
        
    create_and_compare(
        FLAGS.config,
        FLAGS.workdir,
        data_index=FLAGS.data_index,
        noise_sigma=FLAGS.noise_sigma,
        sample_N=FLAGS.sample_N,
        clamp_to=FLAGS.clamp_to,
        num_particles=FLAGS.num_particles,
    )


if __name__ == "__main__":
    app.run(main)
