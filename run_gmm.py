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
import ml_collections

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
    verbose=True,
    methods=["tmpd", "pgdm", "dps", "tmpd_og", "tmpd_exact"],
    clamp_to=20,
):
    """
    Return samples from guided samplers as a dictionary of
    {method: samples} along with the true posterior samples (also in the dictionary).

    Args:
      gmm_config: ml_collections.ConfigDict, configuration for the GMM model.
      return_list: bool, whether to return a list of samples.
      num_samples: int, number of samples to generate.
      dim: int, dimension of the true data.
      obs_dim: int, dimension of the observation
      sigma_y: float, noise scale of observation.
      seed: int, random seed.
      verbose: bool, whether to print the time taken for each method.
      methods: list of strings, list of methods to run.
    """
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if gmm_config is None:
        # creat new config
        gmm_config = ml_collections.ConfigDict()
        gmm_config.dim = dim
        gmm_config.obs_dim = obs_dim
        gmm_config.sigma = sigma_y

        H_mat, U, S, V = create_mask.get_H_mat(gmm_config.dim, gmm_config.obs_dim)

        # sampling specifications
        gmm_config.device = "cpu"
        gmm_config.sampling_eps = 1e-3
        gmm_config.init_type = "gaussian"
        gmm_config.noise_scale = 1.0
        gmm_config.sde_sigma_var = 0.0
        gmm_config.sample_N = 1000  # number of sampling steps

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

    # true posterior
    true_posterior_samples = gmm.sample_from_posterior(
        num_samples, y_obs, H_func.H_mat, gmm_config.sigma
    )

    # common starting point
    start_z = torch.randn(num_samples, gmm_config.dim)

    # list of methods
    # methods = ["tmpd", "pgdm", "dps", "tmpd_og"]
    # methods = ["tmpd", "pgdm", "dps", "reddiff", "tmpd_og"]
    # methods = ["tmpd", "pgdm", "dps", "tmpd_fixed_cov"]

    # samples dictionary
    samples_dict = {method_name: None for method_name in methods}
    samples_dict["true_posterior"] = true_posterior_samples.detach().numpy()

    list_of_times = []
    for method_name in methods:
        logging.info(f"Running {method_name}...")
        start_time = time.time()
        sampler = registry.get_guided_sampler(
            name=method_name,
            model=gmm_flow_model(gmm),
            sde=gmm.sde,
            shape=(num_samples, gmm_config.dim),
            inverse_scaler=None,
            H_func=H_func,
            noiser=noiser,
            device=gmm_config.device,
            sampling_eps=1e-3,
        )
        # same observation
        y_obs_batched = y_obs.repeat(num_samples, 1)

        # run the sampler
        batched_list_samples = sampler.sample(
            y_obs_batched,
            clamp_to=clamp_to, # clampping to the support of the prior
            z=start_z,
            return_list=return_list,
            method="euler",
            gmm_model=gmm,
            gmres_max_iter = 10
            
        )
        # convert to numpy
        if return_list:
            samples_dict[method_name] = [
                sample.detach().numpy() for sample in batched_list_samples
            ]
        else:
            samples_dict[method_name] = batched_list_samples.detach().numpy()
        end_time = time.time()
        list_of_times.append(end_time - start_time)

        if verbose:
            logging.info(f"Time taken for {method_name}: {end_time - start_time}")
    logging.info(f"Total time taken: {np.sum(list_of_times)}")

    return samples_dict


def visualise_experiment(
    config,
    return_list=True,
    plot=True,
    num_samples=1000,
    seed=123,
    methods=["tmpd", "pgdm", "dps", "tmpd_og", "tmpd_exact"],
):
    """
    Plots the samples and store the GIFs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples_dict = create_samples(
        config,
        return_list=return_list,
        num_samples=num_samples,
        seed=seed,
        methods=methods,
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
                            sample[:, 0],
                            sample[:, 1],
                            alpha=0.5,
                            s=10,
                            label=name.upper(),
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


def run_exp(
    config,
    workdir,
    return_list=False,
    num_samples=1000,
    seed=42,
    # methods=["tmpd_fixed_cov", "pgdm", "dps"],
    # methods=["tmpd_h", "tmpd_fixed_diag"],
    methods=['tmpd_gmres'],
    clamp_to=20,
):
    """
    Run the GMM experiment with the given configuration.
    """
    # follows from Boys et al. 2023
    # dim_list = [8, 80, 800]
    # obs_dim_list = [1, 2, 4]
    # noise_list = [0.01, 0.1, 1.0]

    # for testing
    dim_list = [8]
    obs_dim_list = [1,2,4]
    # noise_list = [0.01]
    noise_list = [0.01, 0.1, 1.0]

    # for debugging
    # dim_list = [8]
    # obs_dim_list = [4]
    # noise_list = [1.0]

    # get cartesian product of the lists
    product_list = list(itertools.product(dim_list, obs_dim_list, noise_list))

    # also removed reddiff
    results_dict = {
        name: {tup: [] for tup in product_list}
        for name in methods
    }  # NOTE: add "tmpd_exact" if needed

    # to compute the confidence intervals, default 20
    num_iters = 20
    for iter in range(num_iters):
        iter_seed = iter
        logging.info(f"\nIteration {iter}...\n")
        for tup in product_list:
            dim, obs_dim, sigma_y = tup
            logging.info(
                f"Running for dim={dim}, obs_dim={obs_dim}, sigma_y={sigma_y}..."
            )
            # use specified parameters for experiments, thus set gmm config to None
            samples_dict = create_samples(
                None,
                return_list=return_list,
                num_samples=num_samples,
                dim=dim,
                obs_dim=obs_dim,
                sigma_y=sigma_y,
                seed=iter_seed,
                methods=methods,
                clamp_to=clamp_to,
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
                    seed=iter_seed,
                )
                logging.info(f"SWD for {method_name} is {swd}")
                results_dict[method_name][tup].append(swd)
            del samples_dict
            gc.collect()

    # compute the mean and std of the SWD
    for method_name in results_dict.keys():
        for key, swd_list in results_dict[method_name].items():
            # for debugging
            print(swd_list)
            
            swd_mean = np.mean(swd_list)
            swd_std = np.std(swd_list)
            results_dict[method_name][key] = {
                "mean": swd_mean,
                "std": swd_std,
            }
            logging.info(f"Method: {method_name}, {key}, SWD: {swd_mean} +/- {swd_std}")

    # store the results to a CSV file
    tf.io.gfile.makedirs(workdir)
    results_df = pd.DataFrame(results_dict).T
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(os.path.join(workdir, f"results_{current_time}.csv"))


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "configs/gmm_configs.py",
    "Sampling configuration.",
    lock_config=False,  # might want to lock
)

flags.DEFINE_string("workdir", "temp", "Work directory.")
flags.DEFINE_boolean("return_list", False, "Return a list of samples.")
flags.DEFINE_boolean("plot", True, "Plot the samples but without saving the GIFs.")
flags.DEFINE_boolean(
    "visualise", False, "Visualise the samples instead of running benchmark."
)


def main(argv):
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    current_time = time.strftime("%Y%m%d-%H%M%S")
    gfile_stream = open(os.path.join(FLAGS.workdir, f"stdout_{current_time}.txt"), "w")
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
        logging.info("Visualising the samples...")
        visualise_experiment(
            FLAGS.config,
            return_list=True,
            plot=FLAGS.plot,
            num_samples=1000,
            seed=42,  # change to 10
        )

    # run the experiment
    else:
        logging.info("Running the experiment...")
        run_exp(
            FLAGS.config,
            FLAGS.workdir,
            return_list=False,
            seed=7021,
        )


if __name__ == "__main__":
    app.run(main)
