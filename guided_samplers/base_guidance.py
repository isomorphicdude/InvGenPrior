"""Implements the base class of algorithms for solving inverse problems with flows."""

from abc import ABC, abstractmethod
import math

import torch

# custom imports
from models import utils as mutils


class GuidedSampler(ABC):
    """Base class for guided samplers."""

    def __init__(
        self,
        model,
        sde,
        shape,
        sampling_eps,
        inverse_scaler,
        H_func,
        noiser,
        device,
        **kwargs,
    ):
        self.model = model
        self.sde = sde
        self.shape = shape
        self.sampling_eps = sampling_eps
        self.inverse_scaler = inverse_scaler
        self.H_func = H_func
        self.noiser = noiser
        self.device = device

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        """
        Computes the posterior samples with respect to y_obs using the guided Euler sampler.

        Args:
            - y_obs (torch.Tensor): Observed data to condition the sampling on. (B, C*H*W)
            - z (torch.Tensor, optional): Optional latent variable for sampling. Default is None. (B, C, H, W)
            - return_list (bool, optional): If True, returns the samples as a list. Default is False.
            - clamp_to (float, optional): If not None, clamps the scores to this value. Default is 1.
            - **kwargs: Additional keyword arguments for the sampling process.

        Returns:
            - torch.Tensor or list: The posterior samples. If `return_list` is True, returns intermediate samples as a list.
        """
        if return_list:
            samples = []

        # Initial sample
        with torch.no_grad():
            if z is None:
                # add warning
                z0 = self.sde.get_z0(
                    torch.zeros(self.shape, device=self.device), train=False
                ).to(self.device)
                x = z0.detach().clone()
            else:
                x = z

            model_fn = mutils.get_model_fn(self.model, train=False)

            ### Uniform
            dt = 1.0 / self.sde.sample_N
            eps = 1e-3  # default: 1e-3
            for i in range(self.sde.sample_N):
                # sampling steps default to 1000
                num_t = i / self.sde.sample_N * (self.sde.T - eps) + eps  # scalar time

                # t_batched = torch.ones(self.shape[0], device=self.device) * num_t

                # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
                sigma_t = self.sde.sigma_t(num_t)

                alpha_t = self.sde.alpha_t(num_t)
                std_t = self.sde.std_t(num_t)
                da_dt = self.sde.da_dt(num_t)
                dstd_dt = self.sde.dstd_dt(num_t)

                guided_vec = self.get_guidance(
                    model_fn,
                    x,
                    num_t,
                    y_obs,
                    alpha_t,
                    std_t,
                    da_dt,
                    dstd_dt,
                    clamp_to=clamp_to,
                    **kwargs,
                )

                x = (
                    x.detach().clone()
                    + guided_vec * dt
                    # + flow_pred * dt
                    # + scaled_grad * dt * gamma_t
                    + sigma_t * math.sqrt(dt) * torch.randn_like(guided_vec).to(self.device)
                ).clip(0, 1) # clipping the image

                if return_list and i % (self.sde.sample_N // 5) == 0:
                    samples.append(x.detach().clone())
                if i == self.sde.sample_N - 1 and return_list:
                    samples.append(x.detach().clone())

        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i])
            nfe = self.sde.sample_N
            return samples, nfe
        else:
            x = self.inverse_scaler(x)
            nfe = self.sde.sample_N
            return x, nfe

    def guided_rk45_sampler(self):
        pass

    def sample(
        self, y_obs, z=None, return_list=False, method="euler", clamp_to=1, **kwargs
    ):
        if method == "euler":
            return self.guided_euler_sampler(
                y_obs, z=z, return_list=return_list, clamp_to=clamp_to, **kwargs
            )[0]
        elif method == "rk45":
            return self.guided_rk45_sampler(
                y_obs, z=z, return_list=return_list, clamp_to=clamp_to, **kwargs
            )[0]
        else:
            raise NotImplementedError(f"Method {method} not yet supported.")

    @abstractmethod
    def get_guidance(self):
        raise NotImplementedError()
