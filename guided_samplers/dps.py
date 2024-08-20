"""Implements the DPS guidance functions."""

import math
import torch

from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler
from models.utils import convert_flow_to_x0


@register_guided_sampler(name="dps")
class DPS(GuidedSampler):
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
        **kwargs
    ):
        """Compute the DPS guidance (Chung et al., 2022)."""
        
        dps_scaling_const = kwargs.get("dps_scaling_const", 1.0)
        data_name = kwargs.get("data_name", None)
        
        with torch.enable_grad():
            x_t = x_t.clone().to(x_t.device).requires_grad_()
            t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t
            
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
            # NOTE: while the authors claim that the stepsize is 1/||y-H(x0_hat)||^2
            # we note that this does not work well for GMM (? maybe the seed is bad)
            # but we will use the official implementation for images
            
            
            if len(y_obs.shape) > 2:
                norm_diff = torch.linalg.norm(y_obs - self.H_func.H(x0_hat))
            else:
                norm_diff = torch.linalg.norm(y_obs - self.H_func.H(x0_hat))**2
            
            # norm_diff = torch.linalg.norm(y_obs - self.H_func.H(x0_hat))
            
        grad_term = torch.autograd.grad(norm_diff, x_t, retain_graph=True)[0]
        
        
        grad_term = (-1) * grad_term.detach()
        
        corrected_grad = grad_term * (std_t**2) * (1 / alpha_t + 1 / std_t)
        scaled_grad = dps_scaling_const * corrected_grad
        if clamp_to is not None and clamp_condition:
            if data_name == "celeba":
                threshold_time = 2.0
            elif data_name == "afhq":
                threshold_time = 2.0
            else:
                threshold_time = 2.0    
            if num_t < threshold_time:
                if data_name == "celeba":
                    guided_vec = torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred
                elif data_name == "afhq":
                    guided_vec = torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred
                else:
                    guided_vec = torch.clamp(scaled_grad + flow_pred, -clamp_to, clamp_to)
                    
            else:
                guided_vec = scaled_grad + flow_pred
        else:
            return scaled_grad + flow_pred
        
        return guided_vec
        
        
        
        
        
        
        