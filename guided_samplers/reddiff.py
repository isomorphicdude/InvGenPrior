"""Implements the REDdiff guidance method."""

import math
import torch

from models import utils as mutils
from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler
from models.utils import convert_flow_to_noise


@register_guided_sampler(name="reddiff")
class REDdiff(GuidedSampler):
    """
    Implements the variational diffusion sampling method by Mardani et al. 2024.

    Does not have guidance function method, only the Euler sampler.
    """

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        """
        Essentially running the flow model as diffusion.  
        
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
        if return_list:
            samples = []
            
        # official implementation does not use SDEdit initialization (Meng et al. 2021)
        # hence init from Gaussian
        if z is None:
            x = torch.randn(self.shape, device = self.device).detach()
        else:
            x = z.clone().detach().to(self.device)

        # mu is the mean of the variational distribution and needs to be optimized
        mu = torch.autograd.Variable(x, requires_grad=True)
        optimizer = torch.optim.Adam(
            [mu], lr=kwargs.get("lr", 0.1), betas=(0.9, 0.99), weight_decay=0.0
        )
        
        model_fn = mutils.get_model_fn(self.model, train=False)
        ### Uniform
        dt = 1.0 / self.sde.sample_N
        eps = 1e-3  # default: 1e-3
        
        for i in range(self.sde.sample_N):
            num_t = i / self.sde.sample_N * (self.sde.T - eps) + eps  # scalar time
            
            # batched time to pass into model function
            t_batched = torch.ones(self.shape[0], device=self.device) * num_t
            
            sigma_t = self.sde.sigma_t(num_t)
            alpha_t = self.sde.alpha_t(num_t)
            std_t = self.sde.std_t(num_t)
            da_dt = self.sde.da_dt(num_t)
            dstd_dt = self.sde.dstd_dt(num_t)
            
            # initialize, with just x0_pred=mu
            x0_pred = mu
            if clamp_to is not None:
                x0_pred = torch.clamp(x0_pred, -clamp_to, clamp_to)
            
            # conditional distribution q(x_t | y) by diffusing the variational distribution
            noise_xt = torch.randn_like(mu).to(self.device)
            x_t = alpha_t * x0_pred + std_t * noise_xt  
            
            # reconstruction loss
            e_obs = y_obs - self.H_func.H(x0_pred)
            loss_obs = (e_obs**2).mean()/2
            
            # Regularization loss
            # grad = E_{t~U(0,1) and noise~N(0,1)} [lambda_t (score - noise)]
            # thus loss is lambda_t (network_out - noise)^T mu
            
            et = convert_flow_to_noise(
                u_t = model_fn(x_t, t_batched * 999),
                x_t = x_t,
                alpha_t=alpha_t,
                std_t=std_t,
                da_dt=da_dt,
                dstd_dt=dstd_dt
            )
            
            # stopped gradient, do not propagate through network
            et = et.detach()
            
            loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()
            
            # signal to noise ratio inverted
            snr_inv = std_t/alpha_t  #1d torch tensor
            
            grad_term_weight = kwargs.get("grad_term_weight", 0.25)
            obs_weight = kwargs.get("obs_weight", 1.0)
            
            
            w_t = grad_term_weight*snr_inv
            v_t = obs_weight
            
            # weighted loss
            loss = w_t*loss_noise + v_t*loss_obs

            #adam step
            optimizer.zero_grad()  #initialize
            loss.backward()
            optimizer.step()
            
            
            # if return_list and i % (self.sde.sample_N // 10) == 0:
            #     samples.append(x_t.clone().detach())
            # if i == self.sde.sample_N - 1 and return_list:
            #     samples.append(x_t.clone().detach())
            if return_list:
                samples.append(x0_pred.clone().detach())

        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i])
            nfe = self.sde.sample_N
            return samples, nfe
        else:
            x0_pred = self.inverse_scaler(x0_pred)
            nfe = self.sde.sample_N
            return x0_pred, nfe
        
    def guided_rk45_sampler(self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs):
        raise NotImplementedError("REDdiff does not support RK45 sampling.")
        
    def get_guidance(self):
        raise NotImplementedError("REDdiff does not have a guidance function.")