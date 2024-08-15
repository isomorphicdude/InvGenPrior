"""Implements the PGDM guidance functions."""

import math
import torch

from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler
import models.utils as mutils
from models.utils import convert_flow_to_x0, get_model_fn


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
        clamp_condition,
        noiseless=False,
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
                mat = (
                    self.H_func.H_pinv(y_obs)
                    - self.H_func.H_pinv(self.H_func.H(x0_hat))
                ).reshape(x_t.shape[0], -1)

                mat_x = (mat.detach() * x0_hat.reshape(x_t.shape[0], -1)).sum()

                grad_term = (
                    torch.autograd.grad(mat_x, x_t, retain_graph=True)[0] / r_t_2
                )

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
                s_times_vec = self.H_func.HHt_inv(
                    y_obs - self.H_func.H(x0_hat), r_t_2=r_t_2, sigma_y_2=sigma_y**2
                )
                # get vec.T @ Sigma_^-1 @ vec
                mat = (
                    ((y_obs - self.H_func.H(x0_hat)).reshape(x_t.shape[0], -1))
                    * s_times_vec
                ).sum()

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

        guided_vec = scaled_grad + flow_pred
        if clamp_to is not None and clamp_condition:
            # clamp_to = flow_pred.flatten().abs().max().item()   
            # scaled_grad = torch.clamp(scaled_grad, -clamp_to, clamp_to)
        # print("Clamping to", clamp_to)
            guided_vec = torch.clamp(guided_vec, -clamp_to, clamp_to)
            # guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
            
        if not self.return_cov:
            return guided_vec
        else:
            # re-compute the C_yy matrix here
            assert self.H_func.H_mat is not None
            C_yy = r_t_2 * self.H_func.H_mat @ self.H_func.H_mat.T + sigma_y**2
            # note here C_yy is shared across the batch
            return guided_vec, x0_hat.detach().mean(dim=0), C_yy


@register_guided_sampler(name="pgdm_mod")
class PiGDM_modified(GuidedSampler):

    def get_x0_pred(self, x_t, num_t):
        """
        An additional abstraction to get the x0 prediction from the flow prediction.
        """
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t
        model_fn = get_model_fn(self.model, train=False)

        # get the coefficients
        alpha_t = self.sde.alpha_t(num_t)
        std_t = self.sde.std_t(num_t)
        da_dt = self.sde.da_dt(num_t)
        dstd_dt = self.sde.dstd_dt(num_t)
        flow_pred = model_fn(x_t, t_batched * 999)

        x0_hat = convert_flow_to_x0(
            u_t=flow_pred,
            x_t=x_t,
            alpha_t=alpha_t,
            std_t=std_t,
            da_dt=da_dt,
            dstd_dt=dstd_dt,
        )

        return x0_hat

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
        noiseless=False,
        **kwargs
    ):
        """
        Compute the PiGDM guidance (Song et al., 2022).

        We modify the guidance by first converting the score prediction
        to a diffusion path with a_t = t and std_t = sqrt(1 - t^2), then
        we use the original coefficients to compute the guidance.

        The motivation is from an incorrect implementation in GMM example,
        where x0 is not correctly predicted:

        x0 = 1/t (x_t + (1 - t) * score)

        instead of 1/t (x_t + (1 - t^2) * score) corresponding to the diffusion path.

        Converting to flow using a_t = t, std_t = 1-t yields the same flow vector field
        as the diffusion path, which means prior sampling is not affected.

        But the guidance is computed using the incorrect x0 along with OT coefficients,
        which surprisingly works well for the GMM example:
            - It is able to find multiple modes in the GMM example.
            - And has apparently less bias

        To implement this we follow the steps below:
            - Define a function that gives the x0 prediction from the flow prediction
            - Convert the x_0 prediction of the model to that of the diffusion path
            - Compute the score prediction using the diffusion path
            - Compute the wrong x0 pred and use it for guidance
            - Keep all other coefficients the same
        """
        desired_time = num_t
        sample_time = desired_time / (math.sqrt(1 - desired_time**2) + desired_time)
        t_batched = (desired_time / sample_time)

        # r_t_2 as in Song et al. 2022
        r_t_2 = std_t**2 / (alpha_t**2 + std_t**2)

        # get the noise level of observation
        sigma_y = self.noiser.sigma

        
        with torch.enable_grad():
            x_t = x_t.clone().to(x_t.device).requires_grad_()

            # convert x0_hat to the diffusion path
            x0_hat = self.get_x0_pred(
                x_t=x_t / t_batched,
                num_t = sample_time
            )
            
            # compute score prediction using the diffusion path
            score_pred = (num_t * x0_hat - x_t) / (1 - num_t**2)
            
            # compute the wrong x0 prediction
            wrong_x0_hat = (1 / num_t) * (x_t + (1 - num_t) * score_pred)
            
            # get Sigma_^-1 @ vec
            s_times_vec = self.H_func.HHt_inv(
                y_obs - self.H_func.H(wrong_x0_hat), r_t_2=r_t_2, sigma_y_2=sigma_y**2
            )
            # get vec.T @ Sigma_^-1 @ vec
            mat = (
                ((y_obs - self.H_func.H(wrong_x0_hat)).reshape(x_t.shape[0], -1))
                * s_times_vec
            ).sum()

        grad_term = torch.autograd.grad(mat, x_t, retain_graph=True)[0] * (-1)

        grad_term = grad_term.detach()
        
        # also compute the flow prediction using diffusion path
        # flow_pred = mutils.convert_x0_to_flow(
        #     x0_hat=x0_hat.detach(),
        #     x_t=x_t / t_batched,
        #     alpha_t=sample_time,
        #     std_t=math.sqrt(1 - sample_time**2),
        #     da_dt=1.0,
        #     dstd_dt= -sample_time / math.sqrt(1 - sample_time**2)
        # )
        
        flow_pred = model_fn(x_t, num_t *torch.ones(x_t.shape[0], device=self.device) * 999)

        # print((flow_pred - flow_pred2).mean())
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
