"""
Implements the GMM experiment in Cardoso et al. 2023 and Boys et al. 2023.

This script creates samples from guided samplers and store the plots to make GIFs.
It also computes the Sliced Wasserstein Distance to the true posterior and outputs
the results to a CSV file.
"""

import os
import gc
import time
import itertools
import logging

import torch
import ot as pot
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ml_collections.config_flags import config_flags

from physics import noisers, operators, create_mask
from guided_samplers import registry, tmpd, pgdm, dps, reddiff
from models import sde_lib, gmm_model
from models.utils import convert_flow_to_x0, convert_x0_to_flow
from evaluation import plotting_utils


class gmm_flow_model(torch.nn.Module):
    def __init__(self, gmm):
        super().__init__()
        self.gmm = gmm

    def forward(self, x, t):
        if len(t.shape) >= 1:
            # print("Batch of time steps")
            # if t is a batch of time steps
            t = t[0]
        t = t / 999
        return self.gmm.flow_pred(x, t)


def get_y_obs(gmm, H_mat, noiser=None):
    """
    Returns the noisy observation H(x_0) + noise.
    """
    x_t = gmm.sample_from_prior(1).squeeze()
    y_obs = H_mat @ x_t
    return noiser(y_obs)


def create_samples(
    gmm_config,
    return_list=False,
    num_samples=1000,
    dim=8,
    obs_dim=1,
    sigma_y=0.05,
    seed=42,
):
    """
    Return samples from guided samplers as a dictionary of
    {method: samples} along with the true posterior samples (also in the dictionary).
    """

    # modify the config according to the input
    gmm_config.dim = dim
    gmm_config.obs_dim = obs_dim
    gmm_config.sigma = sigma_y

    H_mat, U, S, V = create_mask.get_H_mat(gmm_config.dim, gmm_config.obs_dim)

    # put the matrix into config
    gmm_config.H_mat = H_mat
    gmm_config.U_mat = U
    gmm_config.singulars = S
    gmm_config.V_mat = V

    # set SDE
    sde = sde_lib.RectifiedFlow(
        init_type=gmm_config.init_type,
        noise_scale=gmm_config.noise_scale,
        sample_N=gmm_config.sample_N,
        sigma_var=gmm_config.sde_sigma_var,
    )

    # set GMM model
    gmm = gmm_model.GMM(dim=gmm_config.dim, sde=sde)

    H_func = operators.get_operator(name="gmm_h", config=gmm_config)
    noiser = noisers.get_noise(name="gaussian", config=gmm_config)
    y_obs = get_y_obs(gmm, H_func.H_mat, noiser)
    num_samples = 1000

    # true posterior
    true_posterior_samples = gmm.sample_from_posterior(
        num_samples, y_obs, H_func.H_mat, sigma_y
    )

    # common starting point
    start_z = torch.randn(num_samples, gmm_config.dim)

    # list of methods
    methods = ["tmpd", "pgdm", "dps", "reddiff", "tmpd_exact"]
    # methods = ["tmpd", "pgdm", "dps", "reddiff", "tmpd_og"]
    # methods = ["tmpd", "pgdm", "dps", "tmpd_fixed_cov"]

    # samples dictionary
    samples_dict = {method_name: None for method_name in methods}
    samples_dict["true_posterior"] = true_posterior_samples.detach().numpy()

    for method_name in methods:
        logging.info(f"Running {method_name}...")
        sampler = registry.get_guided_sampler(
            name=method_name,
            model=gmm_flow_model(gmm),
            sde=gmm.sde,
            shape=(num_samples, gmm_config.dim),
            inverse_scaler=None,
            H_func=H_func,
            noiser=noiser,
            device="cpu",
            sampling_eps=1e-3,
        )
        # same observation
        y_obs_batched = y_obs.repeat(num_samples, 1)

        # run the sampler
        batched_list_samples = sampler.sample(
            y_obs_batched,
            clamp_to=None,
            z=start_z,
            return_list=return_list,
            method="euler",
        )
        # convert to numpy
        if return_list:
            samples_dict[method_name] = [sample.detach().numpy() for sample in batched_list_samples]
        else:
            samples_dict[method_name] = batched_list_samples.detach().numpy()

    return samples_dict


def visualise_experiment(
    config, return_list=True, plot=True, num_samples=1000, seed=123
):
    """
    Plots the samples and store the GIFs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples_dict = create_samples(
        config, return_list=return_list, num_samples=num_samples, seed=seed
    )
    names = list(samples_dict.keys())
    if plot:
        logging.info("Visualising the samples...")
        fig, axs = plt.subplots(1, len(names), figsize=(30, 5))
        for i in range(len(samples_dict["dps"])):
            if i % 10 == 0:
                for j, name in enumerate(names):
                    samples = samples_dict[name]
                    if name == "true_posterior":
                        sample = samples
                        axs[j].clear()
                        axs[j].scatter(
                            sample[:, 0],
                            sample[:, 1],
                            alpha=0.5,
                            s=10,
                            color="orange",
                            label="True Posterior",
                        )
                        axs[j].set_title(f"{name.upper()}")
                        axs[j].set_xlim(-20, 20)
                        axs[j].set_ylim(-20, 20)
                        axs[j].legend()
                        plt.draw()
                    else:
                        sample = samples[i]

                        axs[j].clear()
                        # also plot the true posterior
                        axs[j].scatter(
                            samples_dict["true_posterior"][:, 0],
                            samples_dict["true_posterior"][:, 1],
                            alpha=0.2,
                            s=10,
                            color="orange",
                            label="True Posterior",
                        )
                        axs[j].scatter(
                            sample[:, 0],
                            sample[:, 1],
                            alpha=0.2,
                            s=10,
                            label=name.upper(),
                        )
                        axs[j].set_title(f"{name.upper()} time step {i}")
                        axs[j].set_xlim(-20, 20)
                        axs[j].set_ylim(-20, 20)
                        axs[j].legend()
                        plt.draw()

                plt.pause(0.01)
                # plt.clf()
        time.sleep(30)

    else:
        # only save the figures to make GIFs
        logging.info("Saving the images...")
        for i in range(len(samples_dict["dps"])):
            plt.clf()
            fig, axs = plt.subplots(1, len(names), figsize=(30, 5))
            if i % 20 == 0:
                for j, name in enumerate(names):
                    samples = samples_dict[name]
                    if name == "true_posterior":
                        sample = samples
                        axs[j].scatter(
                            sample[:, 0],
                            sample[:, 1],
                            alpha=0.5,
                            s=10,
                            color="orange",
                            label="True Posterior",
                        )
                        axs[j].set_title(f"{name.upper()}")
                        axs[j].set_xlim(-20, 20)
                        axs[j].set_ylim(-20, 20)
                        axs[j].legend()
                    else:
                        sample = samples[i]
                        # also plot the true posterior
                        axs[j].scatter(
                            samples_dict["true_posterior"][:, 0],
                            samples_dict["true_posterior"][:, 1],
                            alpha=0.3,
                            s=10,
                            color="orange",
                            label="True Posterior",
                        )
                        axs[j].scatter(
                            sample[:, 0], sample[:, 1], alpha=0.5, s=10, label=name.upper()
                        )
                        axs[j].set_title(f"{name.upper()} time step {i}")
                        axs[j].set_xlim(-20, 20)
                        axs[j].set_ylim(-20, 20)
                        axs[j].legend()
                plt.savefig(f"temp/GMM_{i}.png")
                plt.close(fig)
                del fig
                gc.collect(generation=2)
                plt.close("all")
                
    # calling the function to create the GIF
    if not plot:
        logging.info("Creating the GIFs...")
        plotting_utils.make_gif_gmm("temp", "temp/GMM.gif", duration=100)
        # remove the images
        # os.system("rm temp/*.png")

def run_exp(config, workdir, return_list=False, num_samples=1000, seed=42):
    """
    Run the GMM experiment with the given configuration.
    """
    # set seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # follows from Boys et al. 2023
    # dim_list = [8, 80, 800]
    # obs_dim_list = [1, 2, 4]
    # noise_list = [0.01, 0.1, 1.0]
    dim_list = [80]
    obs_dim_list = [1]
    noise_list = [0.05]

    # get cartesian product of the lists
    product_list = list(itertools.product(dim_list, obs_dim_list, noise_list))

    results_dict = {name: {
        tup: [] for tup in product_list
    } for name in ["tmpd", "pgdm", "dps", "reddiff"]} #NOTE: add "tmpd_exact" if needed


    # to compute the confidence intervals
    num_iters = 1
    for iter in range(num_iters):
        logging.info(f"Iteration {iter}...")
        for tup in product_list:
            dim, obs_dim, sigma_y = tup
            logging.info(f"Running for dim={dim}, obs_dim={obs_dim}, sigma_y={sigma_y}...")
            samples_dict = create_samples(
                config,
                return_list=return_list,
                num_samples=num_samples,
                dim=dim,
                obs_dim=obs_dim,
                sigma_y=sigma_y,
                seed=seed,
            )

            # compute the Sliced Wasserstein Distance
            for method_name, samples in samples_dict.items():
                if method_name == "true_posterior":
                    continue
                logging.info(f"Computing SWD for {method_name}...")
                swd = pot.sliced_wasserstein_distance(
                    samples,
                    samples_dict["true_posterior"],
                    n_projections=10000,
                    seed=seed,
                )
                logging.info(f"SWD for {method_name} is {swd}")
                results_dict[method_name][tup].append(swd)
            del samples_dict
            gc.collect()

    # compute the mean and std of the SWD
    for method_name in results_dict.keys():
        for key, swd_list in results_dict[method_name].items():
            results_dict[method_name][key] = {
                "mean": np.mean(swd_list),
                "std": np.std(swd_list),
            }
    
    
    # store the results to a CSV file
    results_df = pd.DataFrame(results_dict).T

    results_df.to_csv(os.path.join(workdir, "results.csv"))


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", "configs/gmm_configs.py", "Sampling configuration.", lock_config=False  # might want to lock
)

flags.DEFINE_string("workdir", "temp", "Work directory.")
flags.DEFINE_boolean("return_list", False, "Return a list of samples.")
flags.DEFINE_boolean("plot", True, "Plot the samples but without saving the GIFs.")
flags.DEFINE_boolean("visualise", True, "Visualise the samples instead of running benchmark.")


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

    # visualise the samples
    if FLAGS.visualise:
        visualise_experiment(
            FLAGS.config, return_list=True, plot=FLAGS.plot, num_samples=1000, seed=42 # change to 10
        )
        
    # run the experiment
    else:
        run_exp(
            FLAGS.config,
            FLAGS.workdir,
            return_list=False,
            seed=42,
        )


if __name__ == "__main__":
    app.run(main)
