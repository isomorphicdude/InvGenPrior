"""Implements the PGDM guidance functions."""

import math
import torch

from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler
from models.utils import convert_flow_to_x0


@register_guided_sampler(name="pgdm")
class PiGDM(GuidedSampler):
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
        noiseless = False,
        **kwargs
    ):
        """Compute the PiGDM guidance (Song et al., 2022)."""
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t
        
        # r_t_2 as in Song et al. 2022
        r_t_2 = std_t**2 / (alpha_t**2 + std_t**2)
        
        # get the noise level of observation
        sigma_y = self.noiser.sigma
        
        if noiseless:
            # using the author's implementation with pseudo inverse
            with torch.enable_grad():
                x_t = x_t.clone().to(x_t.device).requires_grad_()
            
                flow_pred = model_fn(x_t, t_batched * 999)
            
                # pass to model to get x0_hat prediction
                x0_hat = convert_flow_to_x0(
                    u_t=flow_pred,
                    x_t=x_t,
                    alpha_t=alpha_t,
                    std_t=std_t,
                    da_dt=da_dt,
                    dstd_dt=dstd_dt,
                )
                mat = (self.H_func.H_pinv(y_obs) - self.H_func.H_pinv(self.H_func.H(x0_hat))).reshape(x_t.shape[0], -1)

                mat_x = (mat.detach() * x0_hat.reshape(x_t.shape[0], -1)).sum()
                
                grad_term = torch.autograd.grad(mat_x, x_t, retain_graph=True)[0] / r_t_2
            
        else:
            with torch.enable_grad():
                x_t = x_t.clone().to(x_t.device).requires_grad_()
                flow_pred = model_fn(x_t, t_batched * 999)
            
                # pass to model to get x0_hat prediction
                x0_hat = convert_flow_to_x0(
                    u_t=flow_pred,
                    x_t=x_t,
                    alpha_t=alpha_t,
                    std_t=std_t,
                    da_dt=da_dt,
                    dstd_dt=dstd_dt,
                )
                # get Sigma_^-1 @ vec
                s_times_vec = self.H_func.HHt_inv(y_obs - self.H_func.H(x0_hat),
                                                    r_t_2 = r_t_2,
                                                    sigma_y_2 = sigma_y**2)
                # get vec.T @ Sigma_^-1 @ vec
                mat = (((y_obs - self.H_func.H(x0_hat)).reshape(x_t.shape[0], -1)) * s_times_vec).sum()
                
            grad_term = torch.autograd.grad(mat, x_t, retain_graph=True)[0] * (-1)
            
        grad_term = grad_term.detach()
        
        # compute gamma_t scaling
        # only use for images but not GMM example
        # (using OT path)
        if len(y_obs.shape) > 2:
            gamma_t = 1.0
        else:
            gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))
        
        # print(gamma_t)
        scaled_grad = grad_term * (std_t**2) * (1 / alpha_t + 1 / std_t) * gamma_t
         
        # print("scaled_grad", scaled_grad.mean())
        if clamp_to is not None:
            scaled_grad = torch.clamp(scaled_grad, -clamp_to, clamp_to)
        
        return scaled_grad + flow_pred
        
        
        
        
        
        
        