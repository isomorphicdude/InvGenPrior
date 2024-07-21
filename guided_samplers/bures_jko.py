"""
Implements a Bures-Wasserstein JKO sampler, 
following Sarkka (2007) and Lambert et al. (2022) and Yi & Liu (2023)
"""

import math
import torch

from models import utils as mutils
from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler
from models.utils import convert_flow_to_score


@register_guided_sampler(name="bures_jko")
class BuresJKO(GuidedSampler):
    """
    Implements the evolving mean mu_t for Gaussian VI.
    """

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        """
        Weird and probably will not work.
        Requires the score at time 0 but we are working with flow,
        I don't see any reason why particular to flows?

        Here all vectors are flattened unless it's particle.

        Args:
            - y_obs (torch.Tensor): Observed data to condition the sampling on. (B, C*H*W)
            - z (torch.Tensor, optional): Optional latent variable for sampling. Default is None. (B, C, H, W)
            - return_list (bool, optional): If True, returns the samples as a list. Default is False.
            - clamp_to (float, optional): If not None, clamps the scores to this value. Default is 1.
            - **kwargs: Additional keyword arguments for the sampling process.

        Returns:
            - torch.Tensor or list: The posterior samples. If `return_list` is True, returns intermediate samples as a list.
            - int: The number of function evaluations (NFEs) used in the sampling process.
        """
        # number of particles to estimate the Monte Carlo approximation
        N_approx = kwargs.get("N_approx", 4)

        if return_list:
            samples = []
        with torch.no_grad():
            if z is None:
                x = torch.randn(self.shape, device=self.device).detach().reshape(-1)
            else:
                x = z.clone().detach().to(self.device).reshape(-1)

            model_fn = mutils.get_model_fn(self.model, train=False)

            # the mu_t from the variational distribution, (B*C*H*W, )
            mu_t = torch.randn(x.shape, device=self.device).detach()
            sigma_t = 0.1 * torch.ones_like(mu_t)

            ### Uniform
            dt = 1.0 / self.sde.sample_N
            eps = 1e-3  # default: 1e-3

            # get designated time steps
            # tau_1, ..., tau_n to target for divide-and-conquer
            no_taus = 100
            taus_list = range(0, self.sde.sample_N, self.sde.sample_N // no_taus)
            
            # steps_per_tau = self.sde.sample_N // no_taus
            steps_per_tau = 10

            # for i in range(self.sde.sample_N):
            for tau in taus_list:
                print(f"tau: {tau}")
                for step in range(steps_per_tau):
                    print(f"step: {step}")
                    # num_t = i / self.sde.sample_N * (self.sde.T - eps) + eps
                    num_t = tau / self.sde.sample_N * (self.sde.T - eps) + eps

                    # pass through the model function as (batch * N_approx, C, H, W)
                    t_batched = (
                        torch.ones(self.shape[0] * N_approx, device=self.device) * num_t
                    )

                    alpha_t = self.sde.alpha_t(num_t)
                    std_t = self.sde.std_t(num_t)
                    da_dt = self.sde.da_dt(num_t)
                    dstd_dt = self.sde.dstd_dt(num_t)

                    print(f"mu_t: {mu_t.mean()}")
                    print(f"sigma_t: {sigma_t.mean()}")
                    # sample from variational distribution
                    q_t_samples = mu_t.repeat(N_approx, 1)
                    print(f"q_t_samples: {q_t_samples.mean()}")
                    q_t_samples += torch.sqrt(sigma_t) * torch.randn_like(
                        q_t_samples
                    )
                    print(f"sqrt(sigma_t): {torch.sqrt(sigma_t).mean()}")
                    
                    print(f"q_t_samples: {q_t_samples.mean()}")

                    # compute the derivative with Monte Carlo approximation
                    sigma_y = self.noiser.sigma

                    print(f"q_t_samples: {q_t_samples.mean()}")
                    # first compute grad_x (-1/2sigma_y^2 ||y - H(x)||^2)
                    x_t = (
                        q_t_samples.reshape(N_approx * self.shape[0], *self.shape[1:])
                        .clone()
                        .detach()
                    )
                    
                    print(f"x_t: {x_t.mean()}")

                    with torch.enable_grad():
                        x_t.requires_grad_(True)
                        H_x = self.H_func.H(x_t)
                        norm_diff_sum_batched = (
                            torch.linalg.norm(y_obs.repeat(N_approx, 1) - H_x) ** 2
                        )

                    grad_term = torch.autograd.grad(
                        norm_diff_sum_batched, x_t, create_graph=True
                    )[0]
                    grad_term = grad_term.detach()

                    x_t = x_t.detach()

                    flow_pred = model_fn(
                        x_t,
                        t_batched * 999,
                    )

                    score_pred = convert_flow_to_score(
                        flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt
                    )

                    dmu_dt = torch.mean(
                        (score_pred - (0.5 / sigma_y**2) * grad_term).reshape(
                            N_approx, self.shape[0], -1
                        ),
                        dim=0,
                    )
                    
                    mean_diff = x_t.reshape(N_approx, self.shape[0], -1) - mu_t.reshape(self.shape[0], -1).unsqueeze(0)
                    
                    dsigma_dt = torch.mean(
                        (
                            (score_pred
                            - (0.5 / sigma_y**2)
                            * grad_term).reshape(N_approx, self.shape[0], -1)
                            + (((0.5 / sigma_t**2).reshape(self.shape[0], -1).unsqueeze(0))
                            * mean_diff)
                        )
                        * mean_diff.reshape(N_approx, self.shape[0], -1) * 2,
                        dim=0,
                    )
                    
                    # print(f"dmu_dt: {dmu_dt.mean()}")
                    
                    print(f"dsigma_dt: {dsigma_dt.mean()}")
                    
                    print(f"sigma_t: {sigma_t.mean()}")

                    mu_t = mu_t + dt * dmu_dt.reshape(mu_t.shape)
                    sigma_t = sigma_t + dt * dsigma_dt.reshape(sigma_t.shape)

                if return_list:
                    samples.append(
                        x_t.reshape(N_approx, self.shape[0], *self.shape[1:])[0]
                    )

        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i])
            return samples, self.sde.sample_N
        else:
            return self.inverse_scaler(mu_t.reshape(self.shape)), self.sde.sample_N


    def get_guidance(self):
        raise NotImplementedError("This method is not implemented for Bures-JKO.")

    def guided_rk45_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        raise NotImplementedError("This method is not implemented for Bures-JKO.")
