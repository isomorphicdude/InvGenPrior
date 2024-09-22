"""Implements the TMPD with recycling CG guidance."""

import math
import gc

import pickle
import torch
import functorch
import numpy as np
from tqdm import tqdm

import models.utils as mutils
from models.utils import convert_flow_to_x0
from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler
from guided_samplers.linalg import gmres, conjugate_gradient


@register_guided_sampler(name="tmpd_recycle")
class TMPD_cgr(GuidedSampler):
    def get_guidance(
        self,
        model_fn,
        x_t,
        num_t,
        y_obs,
        alpha_t,
        std_t,
        da_dt,
        dstd_dt,
        clamp_to,
        clamp_condition,
        init_guess=None,
        **kwargs
    ):
        """
        Using GMRES to solve the linear system for the guidance.
        """
        data_name = kwargs.get("data_name", None)
        gmres_max_iter = kwargs.get("gmres_max_iter", 1)
        return_basis = kwargs.get("return_basis", False)

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def estimate_h_x_0(x):
            flow_pred = model_fn(x, t_batched * 999)

            # pass to model to get x0_hat prediction
            x0_hat = convert_flow_to_x0(
                u_t=flow_pred,
                x_t=x,
                alpha_t=alpha_t,
                std_t=std_t,
                da_dt=da_dt,
                dstd_dt=dstd_dt,
            )

            x0_hat_obs = self.H_func.H(x0_hat)

            return (x0_hat_obs, flow_pred)

        def estimate_x_0(x):
            flow_pred = model_fn(x, t_batched * 999)

            x0_hat = convert_flow_to_x0(
                u_t=flow_pred,
                x_t=x,
                alpha_t=alpha_t,
                std_t=std_t,
                da_dt=da_dt,
                dstd_dt=dstd_dt,
            ).reshape(self.shape[0], -1)

            return (x0_hat, flow_pred)


        x0_hat_obs, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        x_0_hat = convert_flow_to_x0(
            u_t=flow_pred,
            x_t=x_t,
            alpha_t=alpha_t,
            std_t=std_t,
            da_dt=da_dt,
            dstd_dt=dstd_dt,
        )

        coeff_C_yy = std_t**2 / (alpha_t)

        difference = y_obs - x0_hat_obs

        def cov_y_xt(v):
            return (
                self.noiser.sigma**2 * v
                + self.H_func.H(vjp_estimate_h_x_0(v)[0]) * coeff_C_yy
            )
            
        _grad_ll = gmres(
            A=cov_y_xt,
            b=difference,
            x=init_guess,
            maxiter=gmres_max_iter,
        )

        grad_ll = vjp_estimate_h_x_0(_grad_ll)[0]
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None and clamp_condition:
            if num_t < 0.1:
                guided_vec = torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred

            else:
                guided_vec = scaled_grad + flow_pred
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        return guided_vec, _grad_ll

    

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        recycle_start_time = kwargs.get("recycle_start_time", 10) # default: 10
        
        if return_list:
            samples = []

        if self.return_cov:
            list_mean_0t = []
            list_cov_yt = []

        # Initial sample
        with torch.no_grad():
            if z is None:
                # default Gaussian latent
                z0 = self.sde.get_z0(
                    torch.zeros(self.shape, device=self.device), train=False
                ).to(self.device)
                x = z0.detach().clone()
            else:
                # latent variable taken to be alpha_t y + sigma_t \epsilon
                x = z

            model_fn = mutils.get_model_fn(self.model, train=False)

            ### Uniform
            dt = 1.0 / self.sde.sample_N
            eps = 1e-3  # default: 1e-3
            
            init_guess = torch.zeros_like(y_obs).to(self.device)

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

                guided_vec, prev_soln = self.get_guidance(
                    model_fn,
                    x,
                    num_t,
                    y_obs,
                    alpha_t,
                    std_t,
                    da_dt,
                    dstd_dt,
                    clamp_to=clamp_to,
                    clamp_condition=True,
                    init_guess=init_guess,
                    **kwargs,
                )
                
                init_guess = prev_soln if i > recycle_start_time else torch.zeros_like(y_obs).to(self.device)

                x = (
                    x.detach().clone()
                    + guided_vec * dt
                    + sigma_t
                    * math.sqrt(dt)
                    * torch.randn_like(guided_vec).to(self.device)
                )

                if return_list:
                    samples.append(x.detach().clone())

        if not self.return_cov:
            if return_list:
                for i in range(len(samples)):
                    samples[i] = self.inverse_scaler(samples[i])
                nfe = self.sde.sample_N
                return samples, nfe
            else:
                x = self.inverse_scaler(x)
                nfe = self.sde.sample_N
                return x, nfe
        else:
            return samples, list_mean_0t, list_cov_yt