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

"""All functions and modules related to model definition and loading.
"""

import os

import torch
import logging
import numpy as np
import tensorflow as tf

import models.sde_lib as sde_lib


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    # print(cls, name)
    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    # print(_MODELS)
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)

    num_params = 0
    for p in score_model.parameters():
        num_params += p.numel()
    print("Number of Parameters in the Score Model:", num_params)

    score_model = torch.nn.DataParallel(score_model)
    return score_model


def create_model_no_parallel(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)

    num_params = 0
    for p in score_model.parameters():
        num_params += p.numel()
    print("Number of Parameters in the Score Model:", num_params)

    # score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def convert_flow_to_x0(u_t, x_t, alpha_t, std_t, da_dt, dstd_dt):
    """
    Convert flow to x0.

    Args:  
      - u_t: torch.tensor (b, c, h, w), the flow vector.
	  - x_t: torch.tensor (b, c, h, w), the input image.
      - alpha_t: float, the alpha_t coeff.
      - std_t: float, the sigma_t coeff.
      - da_dt: float, derivative of alpha_t coeff (this is alpha bar in DDPM paper).
      - dstd_dt: float, derivative of sigma_t coeff (see Pokle et al. 2024)

    Returns:
		- x0_hat: torch.tensor (b, c, h, w), the noiseless prediction.
    """
    # da_dt = 1.0
    # dvar_dt = -1.0
    x0_coeff = da_dt - (alpha_t * dstd_dt / std_t)
    xt_coeff = dstd_dt / std_t
    x0_hat = (u_t - xt_coeff * x_t) / x0_coeff
    return x0_hat


def convert_x0_to_flow(x0_hat, x_t, alpha_t, std_t, da_dt, dstd_dt):
    """
    Convert x0 to flow.   
    
    Args:	 
	  - x0_hat: torch.tensor (b, c, h, w), the noiseless prediction.
	  - x_t: torch.tensor (b, c, h, w), the input image.
	  - alpha_t: float, the alpha_t coeff.
	  - std_t: float, the sigma_t coeff.
	  - da_dt: float, derivative of alpha_t coeff (this is alpha bar in DDPM paper).
	  - dstd_dt: float, derivative of sigma_t coeff (see Pokle et al. 2024)  
    
	Returns:    
	  - u_t: torch.tensor (b, c, h, w), the flow vector.  
    """
    # da_dt = 1.0
    # dvar_dt = -1.0
    x0_coeff = da_dt - (alpha_t * dstd_dt / std_t)
    xt_coeff = dstd_dt / std_t

    u_t = x0_hat * x0_coeff + xt_coeff * x_t
    return u_t


def convert_flow_to_noise(u_t, x_t, alpha_t, std_t, da_dt, dstd_dt):
    """
    Convert the flow prediction to noise prediction in DDPM/IM.
    
    Args:  
        - u_t: torch.tensor (b, c, h, w), the flow vector.
        - x_t: torch.tensor (b, c, h, w), the input image.
        - alpha_t: float, the alpha_t coeff.
        - std_t: float, the sigma_t coeff.
        - da_dt: float, derivative of alpha_t coeff (this is alpha bar in DDPM paper).
        - dstd_dt: float, derivative of sigma_t coeff (see Pokle et al. 2024)
        
    Returns:  
        - noise_t: torch.tensor (b, c, h, w), the noise prediction.
    """
    x0_hat = convert_flow_to_x0(u_t, x_t, alpha_t, std_t, da_dt, dstd_dt)
    
    return (x_t - x0_hat * alpha_t) / std_t


def convert_flow_to_score(u_t, x_t, alpha_t, std_t, da_dt, dstd_dt):
    """
    Convert the flow prediction to the score prediction in DDPM/IM.
    
    Args:  
        - u_t: torch.tensor (b, c, h, w), the flow vector.
        - x_t: torch.tensor (b, c, h, w), the input image.
        - alpha_t: float, the alpha_t coeff.
        - std_t: float, the sigma_t coeff.
        - da_dt: float, derivative of alpha_t coeff (this is alpha bar in DDPM paper).
        - dstd_dt: float, derivative of sigma_t coeff (see Pokle et al. 2024)
        
    Returns:  
        - score_t: torch.tensor (b, c, h, w), the score prediction.
    """
    noise = convert_flow_to_noise(u_t, x_t, alpha_t, std_t, da_dt, dstd_dt)
    
    return  (-1) * noise / std_t

def convert_score_to_flow(score_t, x_t, alpha_t, std_t, da_dt, dstd_dt):
    """
    Convert score prediction to flow prediction.   
    """
    coeff_xt = da_dt / alpha_t
    
    coeff_score = (std_t / alpha_t) * (std_t * da_dt - alpha_t * dstd_dt)
    
    return coeff_xt * x_t + coeff_score * score_t

def convert_m0t_to_mst(m_0t, x_t, sde, t, s):
    """
    Convert the estimated m_0t to m_st, that is, convert 
    the estimated E[x0|xt] to E[xs|xt].
    """
    alpha_t = sde.alpha_t(t)
    alpha_s = sde.alpha_t(s)
    std_t = sde.std_t(t)
    std_s = sde.std_t(s)
    
    alpha_ts = alpha_t / alpha_s
    std_ts = (alpha_s * std_t - alpha_t * std_s) / alpha_s
    
    return (1 / alpha_ts) * (x_t + (std_ts**2 / std_t**2) * (alpha_t * m_0t - x_t))

def convert_cov0t_to_covst_func(v, cov_0t, sde, t, s):
    """
    Convert the estimated cov_0t to cov_st, that is, convert 
    the estimated Cov[x0|xt] to Cov[xs|xt].
    Takes input v and outputs matrix-vector product of v with the cov_st: C_{s|t} @ v.
    Here cov_0t is a function also outputs matrix-vector product of v with the cov_0t: C_{0|t} @ v.    
    
    Args:    
      v: torch.tensor (b, c, h, w), the input vector.
    """
    alpha_t = sde.alpha_t(t)
    alpha_s = sde.alpha_t(s)
    std_t = sde.std_t(t)
    std_s = sde.std_t(s)
    
    alpha_ts = alpha_t / alpha_s
    std_ts = (alpha_s * std_t - alpha_t * std_s) / alpha_s
    
    first_term = 1 / alpha_ts * v 
    second_term = (std_ts**2 * alpha_t**2) / (alpha_ts * std_t**4) * cov_0t(v)
    third_term = (-1) * std_ts**2 / (alpha_ts * std_t**2) * v
    
    return (first_term + second_term + third_term) * (std_ts**2 / alpha_ts)

def restore_checkpoint(ckpt_dir, state, device):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])

        # state['model'].load_state_dict(loaded_state['model'], strict=False)
        if isinstance(state["model"], torch.nn.DataParallel):
            print("Model is DataParallel")
            state["model"].load_state_dict(loaded_state["model"], strict=False)
        else:
            # this is for loading model for evaluation
            # torch.func.vjp does not seem to work with DataParallel
            print("Model is not DataParallel")
            model_state_dict = {
                key.replace("module.", ""): value
                for key, value in loaded_state["model"].items()
            }
            state["model"].load_state_dict(model_state_dict, strict=False)
            print("Model state loaded successfully.")

        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        print(f"Loaded checkpoint from {ckpt_dir}")
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)
