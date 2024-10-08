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
from guided_samplers import tmpd, dps, pgdm, reddiff, bures_jko, tmpd_cgr
from guided_samplers.registry import get_guided_sampler

# evaluation
from evaluation import recon_metrics


def create_samples(
    config,
    workdir,
    save_degraded=True,
    return_list=False,
    nfe=100,
    eval_folder="eval_samples",
    max_num_samples=None,
    noise_level=None,
    starting_time=0.0,
    gmres_max_iter=1,
    random_subset=False,
    seed=0,
    tmpd_alt_impl=False,
    recycle_start_time=10,
):
    """
    Create samples using the guided sampler.

    Args:
      config: configuration file, used for ml_collections
      workdir: working directory, usually just the root directory of repo
      save_degraded: whether to save the degraded images
      return_list: whether to return a list of samples
      nfe: number of function evaluations for the ODE solver
      eval_folder: folder to save the samples, should be a combination of the
        name of the experiment and the method used to generate the samples
      max_num_samples: maximum number of samples to generate, if None, generate all
      noise_level: noise level to use for the degredation operator
      starting_time: starting time for the sampling process (for PGDM)
      gmres_max_iter: maximum number of iterations for GMRES (for TMPD)  
      random_subset: whether to sample a random subset of the data
      seed: random seed
      tmpd_alt_impl: whether to use the alternative TMPD implementation
      recycle_start_time: starting time for the Krylov recycling process (for TMPD)
      
    Returns:
      eval_dir: directory where the samples are saved
      indices: indices of the samples generated (reconstructed)
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(
        workdir,
        eval_folder,
        config.data.name,
        config.sampling.gudiance_method,
        config.degredation.task_name,
        f"start_{starting_time}_iter_{gmres_max_iter}_nfe_{nfe}_noise_{noise_level}_alt_{tmpd_alt_impl}_cgr_{recycle_start_time}",
    )
    tf.io.gfile.makedirs(eval_dir)

    # create data
    dset = lmdb_dataset.get_dataset(
        name=config.data.name,
        db_path=config.data.lmdb_file_path,
        transform=None,  # overridden by child class
    )
    
    # get random subset of data
    if random_subset and max_num_samples < len(dset):
        np.random.seed(seed)
        indices = np.random.choice(len(dset), max_num_samples, replace=False)
        # print(f"Random indices: {indices}")
        dset = torch.utils.data.Subset(dset, indices)
        
    else:
        if max_num_samples is not None:
            indices = list(range(max_num_samples))
            dset = torch.utils.data.Subset(dset, list(range(max_num_samples)))
        else:
            indices = list(range(len(dset)))
            
    # dump random indices
    with open(os.path.join(eval_dir, "indices.txt"), "w") as f:
        for idx in indices:
            f.write(f"{idx}\n")
        

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
            # sample_N=config.sampling.sample_N,
            sample_N=nfe,
        )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # build degredation operator
    H_func = get_operator(name=config.degredation.name, config=config.degredation)
    
    new_deg_config = ml_collections.ConfigDict()
    new_deg_config.sigma = noise_level
    noiser = get_noise(name=config.degredation.noiser, config=new_deg_config)

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

    # dumping the config setting into a txt
    with open(os.path.join(eval_dir, "config.txt"), "w") as f:
        f.write(f"{config}")

    # begin sampling
    score_model.eval()
    logging.info(f"Using {config.sampling.gudiance_method} guided sampler.")
    logging.info(f"Using dataset {config.data.name}.")
    logging.info(f"Dataset size is {len(data_loader.dataset)}")
    if max_num_samples is not None:
        logging.info(f"Maximum number of samples to generate: {max_num_samples}")
    else:
        max_num_samples = len(data_loader.dataset)
        logging.info(f"Generating all samples.")
        
    logging.info(f"Sampling {config.sampling.batch_size} images at a time.")
    logging.info(f"Task is {config.degredation.task_name}.")
    logging.info(f"The noise level is {noise_level}.")

    # check if certain images are sampled before
    # the img_idx will be written to disk
    if os.path.exists(os.path.join(eval_dir, "sampled_images.txt")):

        logging.info("Some images have been sampled before. Loading...")

        with open(os.path.join(eval_dir, "sampled_images.txt"), "r") as f:
            sampled_images = f.readlines()
            sampled_images = set([int(x.strip()) for x in sampled_images])
        # img counter
        # img_counter = len(sampled_images)
    else:
        logging.info("No previously sampled images, starting over...")
        sampled_images = set()
        # create the file
        with open(os.path.join(eval_dir, "sampled_images.txt"), "w") as f:
            f.write("")

    img_counter = 0

    for iter_no, (batched_img, img_idx) in enumerate(data_loader):

        if img_counter not in sampled_images:
            start_time = time.time()
            logging.info(f"Current batch: {iter_no}")

            # apply scaler
            batched_img = scaler(batched_img)

            # if not already on device
            batched_img = batched_img.to(config.device)

            # apply degredation operator
            y_obs = H_func.H(batched_img)

            # apply noiser
            y_obs = noiser(y_obs)

            # if save the degraded images then return the re-shaped
            if save_degraded:
                y_obs_image = H_func.get_degraded_image(y_obs)
                # apply scaler
                y_obs_image = inverse_scaler(y_obs_image)

            # pass to guided sampler
            start_z = torch.randn(config.sampling.batch_size, *sampling_shape[1:]).to(
                config.device
            )
            batched_samples = guided_sampler.sample(
                y_obs=y_obs,
                z=start_z,  # maybe can use latent encoding
                return_list=return_list,
                method=config.sampling.use_ode_sampler,  # euler or rk45
                # method="euler",
                clamp_to=config.sampling.clamp_to,
                starting_time=starting_time,
                data_name=config.data.name,
                gmres_max_iter=gmres_max_iter,
                alt_impl=tmpd_alt_impl,
                recycle_start_time=recycle_start_time
            )

            # save the images to eval folder
            logging.info(f"Current batch finished. Saving images...")
            if not return_list:
                # logging.info(f"Returning single batch.")
                for j in range(config.sampling.batch_size):
                    img = batched_samples[j]
                    # img = inverse_scaler(img) # already included in sampler
                    save_image(
                        img,
                        # os.path.join(eval_dir, f"{iter_no}_{j}.png"),
                        os.path.join(
                            eval_dir, f"sample_{iter_no * config.sampling.batch_size + j}.png"
                        ),
                        # normalize=True,
                        # range=(-1, 1),
                    )

            else:
                logging.info(f"Returning list of samples.")
                # save list of batched samples
                # [batch1, batch2, ...]
                for i, batch in enumerate(batched_samples):
                    for j in range(config.sampling.batch_size):
                        img = batch[j]
                        # img = inverse_scaler(img) # already included in sampler
                        save_image(
                            img,
                            # os.path.join(eval_dir, f"{iter_no}_{j}_time_{i*sde.sample_N//10}.png"),
                            os.path.join(eval_dir, f"sample_{iter_no * config.sampling.batch_size + j}_time_{i}.png"),
                            # normalize=True,
                            # range=(-1, 1),
                        )

            if save_degraded:
                # logging.info(f"Saving degraded images...")
                for j in range(config.sampling.batch_size):
                    degraded_img = y_obs_image[j]
                    save_image(
                        degraded_img,
                        os.path.join(
                            eval_dir,
                            f"degraded_{iter_no * config.sampling.batch_size + j}.png",
                        ),
                        # normalize=True,
                        # range=(-1, 1),
                    )

            end_time = time.time()

            logging.info(
                f"Batch {iter_no} finished in {end_time - start_time:.3f} seconds."
            )
            # additional time
            logging.info(
                f"Estimated time remaining: {(end_time - start_time) * (len(data_loader) - iter_no):.3f} seconds; Or {(end_time - start_time) * (len(data_loader) - iter_no) / 60:.3f} minutes."
            )

            # write to file to store index
            with open(os.path.join(eval_dir, "sampled_images.txt"), "a") as f:
                for i in range(img_counter, img_counter + config.sampling.batch_size):
                    f.write(f"{i}\n")

            img_counter += config.sampling.batch_size

        else:
            img_sampled_in_batch = [
                img_counter + i for i in range(config.sampling.batch_size)
            ]
            logging.info(f"Skipping image {img_sampled_in_batch}.")
            img_counter += config.sampling.batch_size

        # if iter_no % 4 == 0 and iter_no != 0:
        # if max_num_samples is not None:
        #     if (iter_no + 1) * config.sampling.batch_size >= max_num_samples:
        #         logging.info(f"Finished {iter_no} batches, exiting...")

        #         # clear memory
        #         torch.cuda.empty_cache()
        #         gc.collect()

        #         # exit
        #         break

    logging.info("Sampling finished.")

    # clear memory
    torch.cuda.empty_cache()
    
    return eval_dir, indices


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Sampling configuration.", lock_config=False  # might want to lock
)

flags.DEFINE_string("workdir", "", "Work directory.")

flags.DEFINE_string(
    "eval_folder", "eval_samples", "The folder name for storing evaluation results"
)

flags.DEFINE_integer(
    "max_num_samples",
    None,
    # 10,
    "Maximum number of samples to generate, if None, generate all.",
)

flags.DEFINE_integer("seed", 0, "Random seed.")

# flags.DEFINE_boolean("tune_hyp", False, "Tune hyperparameters.")

flags.DEFINE_integer("nfe", 100, "Number of function evaluations for the ODE solver.")

flags.DEFINE_boolean("random_subset", True, "Sample a random subset of the data.")

flags.DEFINE_float("noise_level", 0.1, "Noise level for degredation operator.")

flags.DEFINE_float("starting_time", 0.0, "Starting time for the sampling process.")

flags.DEFINE_integer("gmres_max_iter", 1, "Maximum number of iterations for GMRES.")

flags.DEFINE_boolean("tmpd_alt_impl", False, "Use alternative TMPD implementation.")

flags.DEFINE_integer("recycle_start_time", 10, "Starting time for the Krylov recycling process.")

flags.DEFINE_boolean("return_list", False, "Return a list of samples.")

flags.DEFINE_boolean("compute_recon_metrics", False, "Compute reconstruction metrics: PSNR, SSIM, LPIPS.")

flags.DEFINE_boolean("compute_fid", False, "Compute FID and KID.")

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

    samples_eval_dir, sample_indices = create_samples(
        FLAGS.config,
        FLAGS.workdir,
        save_degraded=False,
        return_list=FLAGS.return_list,
        eval_folder=FLAGS.eval_folder,
        max_num_samples=FLAGS.max_num_samples,
        gmres_max_iter=FLAGS.gmres_max_iter,
        noise_level=FLAGS.noise_level,
        starting_time=FLAGS.starting_time,
        random_subset=FLAGS.random_subset,
        nfe=FLAGS.nfe,
        seed=FLAGS.seed,
        tmpd_alt_impl=FLAGS.tmpd_alt_impl,
        recycle_start_time=FLAGS.recycle_start_time,
    )
    
    additional_params = {
        "noise_level": FLAGS.noise_level,
        "starting_time": FLAGS.starting_time,
        "gmres_max_iter": FLAGS.gmres_max_iter,
        "nfe": FLAGS.nfe,
        "alt_impl": FLAGS.tmpd_alt_impl,
        "recycle_start_time": FLAGS.recycle_start_time,
    }
    if FLAGS.compute_recon_metrics:
        # compute recon metrics
        recon_metrics.compute_recon_metrics(
            FLAGS.config,
            workdir=FLAGS.workdir,
            model_output_dir=samples_eval_dir,
            noise_std=FLAGS.noise_level,
            random_indices=sample_indices,
            additional_params=additional_params,
        )
        
    # if FLAGS.tune_hyp:
    #     recon_metrics.get_best_config(
    #         config=FLAGS.config,
    #         workdir=FLAGS.workdir,
    #         noise_std=FLAGS.noise_level,
    #         additional_params=additional_params,
    #     )
    
    if FLAGS.compute_fid:
        raise NotImplementedError("FID and KID computation not implemented yet.")
    
    # remove all samples after metrics are computed
    logging.info("Removing all samples...")
    img_dir = os.listdir(samples_eval_dir)
    for img in img_dir:
        if img.endswith(".png"):
            os.remove(os.path.join(samples_eval_dir, img))
    

if __name__ == "__main__":
    app.run(main)
