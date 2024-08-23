"""Implements the TMPD guidance functions."""

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


@register_guided_sampler(name="tmpd")
class TMPD(GuidedSampler):
    # TMPD does not seem to require any additional hyperparameters
    def _get_guidance(
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
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation with diagonal of Jacobian approximation.

        The diagonal is estimated using Bekas et al. 2005

        Args:
          - model_fn: model function that takes x_t and t as input and returns the flow prediction
          - x_t: current state x_t ~ p_t(x_t|z, y)
          - num_t: current time step
          - y_obs: observed data
          - alpha_t: alpha_t
          - std_t: std_t, the sigma_t in Pokle et al. 2024
          - da_dt: derivative of alpha w.r.t. t
          - dstd_dt: derivative of std w.r.t. t
          - clamp_to: gradient clipping for the guidance

        Returns:
         - guided_vec: guidance vector with flow prediction and guidance combined
        """
        num_hutchinson_samples = kwargs.get("num_hutchinson_samples", 50)
        new_noise = kwargs.get("new_noise", None)
        data_name = kwargs.get("data_name", None)
        # alt_clamp_to = kwargs.get("alt_clamp_to", None)
        # num_hutchinson_samples = 150

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        # x_t = x_t.clone().detach()

        # if alt_clamp_to is not None:
        #     x_t = x_t.detach().clone().clamp(-alt_clamp_to, alt_clamp_to)

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

            return x0_hat, flow_pred

        x_0_pred, vjp_estimate_x_0, flow_pred = torch.func.vjp(
            estimate_x_0, x_t, has_aux=True
        )

        def v_vjp_est(x):
            return self.H_func.Vt(vjp_estimate_x_0(self.H_func.V(x))[0])

        # # compute the diagonal of the Jacobian
        # if len(self.shape) > 2:
        #     diagonal_est = self.hutchinson_diag_est(
        #         vjp_est=v_vjp_est,
        #         shape=(self.shape[0], math.prod(self.shape[1:])),
        #         num_samples=num_hutchinson_samples,
        #     )

        # elif len(self.shape) <= 2:
        #     # print("Using parallel hutchinson")
        #     diagonal_est = self.parallel_hutchinson_diag_est(
        #         vjp_est=v_vjp_est,
        #         shape=(self.shape[0], math.prod(self.shape[1:])),
        #         num_samples=num_hutchinson_samples,
        #         chunk_size=num_hutchinson_samples,
        #     )

        coeff_C_yy = std_t**2 / (alpha_t)

        # difference
        # add noise to the observation
        new_noise_std = 0.25
        # y_obs = y_obs + new_noise * new_noise_std
        difference = y_obs - self.H_func.H(x_0_pred)

        # diagonal_est = (num_t) * diagonal_est + (1 - num_t) * 1.0
        diagonal_est = (1 - num_t) * torch.ones_like(x_t.reshape(x_t.shape[0], -1))
        
        if len(self.shape) > 2:
            # if self.noiser.sigma <= 0.01:
            #     vjp_product = self.H_func.HHt_inv_diag(
            #         vec=difference,
            #         diag=coeff_C_yy * diagonal_est,
            #         sigma_y_2=self.noiser.sigma**2 + new_noise_std**2,
            #     )
            # else:
            #     vjp_product = self.H_func.HHt_inv_diag(
            #         vec=difference,
            #         diag=coeff_C_yy * diagonal_est,
            #         sigma_y_2=self.noiser.sigma**2,
            #     )
            if (
                self.H_func.__class__.__name__
                == "Inpainting"
                # or self.H_func.__class__.__name__ == "SuperResolution"
            ):
                vjp_product = self.H_func.HHt_inv_diag(
                    vec=difference,
                    diag=coeff_C_yy * diagonal_est,
                    sigma_y_2=self.noiser.sigma**2,
                )
            else:
                vjp_product = self.H_func.HHt_inv_diag(
                    vec=difference,
                    diag=coeff_C_yy * diagonal_est,
                    sigma_y_2=self.noiser.sigma**2,
                )
        elif len(self.shape) <= 2:
            vjp_product = self.H_func.HHt_inv_diag(
                vec=difference,
                diag=coeff_C_yy * diagonal_est,
                sigma_y_2=self.noiser.sigma**2,
            )

        grad_ll = vjp_estimate_x_0(self.H_func.Ht(vjp_product))[0]
        # grad_ll = vjp_estimate_h_x_0(vjp_product)[0]
        # print(f"grad_ll mean: {grad_ll.mean()}")
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None and clamp_condition:
            # clamp_to = flow_pred.flatten().abs().max().item()
            if (
                self.H_func.__class__.__name__ == "Inpainting"
                or self.H_func.__class__.__name__ == "SuperResolution"
            ):
                if data_name == "celeba":
                    threshold_time = 2.0
                elif data_name == "afhq":
                    threshold_time = 2.0
                else:
                    threshold_time = 2.0
            else:
                if data_name == "celeba":
                    threshold_time = 2.0
                elif data_name == "afhq":
                    threshold_time = 2.0
                else:
                    threshold_time = 2.0
            if num_t < threshold_time:
                if data_name == "celeba":
                    guided_vec = (
                        torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred
                    )
                elif data_name == "afhq":
                    guided_vec = (
                        torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred
                    )
                    # guided_vec = torch.clamp(
                    #     scaled_grad + flow_pred, -clamp_to, clamp_to
                    # )
                else:
                    guided_vec = torch.clamp(
                        scaled_grad + flow_pred, -clamp_to, clamp_to
                    )

            else:
                guided_vec = scaled_grad + flow_pred

            # else:
            #     guided_vec = scaled_grad + flow_pred
            # guided_vec = (scaled_grad + flow_pred)
            # re-normalisation?
            # flow_pred_norm = torch.linalg.vector_norm(flow_pred, dim=1)
            # guided_vec_norm = torch.linalg.vector_norm(scaled_grad, dim=1)
            # guided_vec = (flow_pred_norm / guided_vec_norm)[:,None] * scaled_grad + flow_pred
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        if not self.return_cov:
            return guided_vec
        else:
            x0_pred = convert_flow_to_x0(
                u_t=flow_pred,
                x_t=x_t,
                alpha_t=alpha_t,
                std_t=std_t,
                da_dt=da_dt,
                dstd_dt=dstd_dt,
            )
            assert self.H_func.H_mat is not None
            # out_C_yy = self.H_func.H_mat @
            out_C_yy = torch.zeros_like(x0_pred)
            return guided_vec, x0_pred.mean(axis=0), out_C_yy

    
    def _get_alt_guidance(
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
        """
        An alternative guidance function to use before/after reaching a certain time.

        Here we take it to be PiGDM.
        """
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        # r_t_2 as in Song et al. 2022
        r_t_2 = std_t**2 / (alpha_t**2 + std_t**2)

        # get the noise level of observation
        sigma_y = self.noiser.sigma

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
        if clamp_to is not None:
            clamp_to = flow_pred.flatten().abs().max().item()
            scaled_grad = torch.clamp(scaled_grad, -clamp_to, clamp_to)

        guided_vec = scaled_grad + flow_pred
        return guided_vec
        # return flow_pred

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
        # condition on the time
        # if num_t < 2:
        return self._get_guidance(
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
            **kwargs,
        )

        # else:
        #     # print(f"Using alternative guidance after time {num_t}")
        #     return self._get_alt_guidance(
        #         model_fn,
        #         x_t,
        #         num_t,
        #         y_obs,
        #         alpha_t,
        #         std_t,
        #         da_dt,
        #         dstd_dt,
        #         clamp_to,
        #         clamp_condition,
        #         **kwargs,
        #     )


@register_guided_sampler(name="tmpd_gmres")
class TMPD_gmres(GuidedSampler):
    def _get_guidance(
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
        """
        Using GMRES to solve the linear system for the guidance.
        """
        data_name = kwargs.get("data_name", None)
        gmres_max_iter = kwargs.get("gmres_max_iter", 50)
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

        # x_0_pred, vjp_estimate_x_0, flow_pred = torch.func.vjp(
        #     estimate_x_0, x_t, has_aux=True
        # )
        x0_hat_obs, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        coeff_C_yy = std_t**2 / (alpha_t)

        difference = y_obs - x0_hat_obs

        def cov_y_xt(v):
            if num_t < 0.8:
                return(
                    self.noiser.sigma**2 * v
                    + self.H_func.H(
                        self.H_func.Ht(v)
                    ) * coeff_C_yy * (1 - num_t)
                )
            else:
                return(
                    self.noiser.sigma**2 * v
                    + self.H_func.H(vjp_estimate_h_x_0(v)[0]) * coeff_C_yy
                )

        grad_ll, V_basis = gmres(
            A=cov_y_xt,
            b=difference,
            maxiter=gmres_max_iter,
        )

        grad_ll = vjp_estimate_h_x_0(grad_ll)[0]

        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None and clamp_condition:
            if num_t < 0.1:
                guided_vec = torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred

            else:
                guided_vec = scaled_grad + flow_pred
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        if not return_basis:
            return guided_vec
        else:
            return guided_vec, V_basis

    def _get_alt_guidance(
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
        data_name = kwargs.get("data_name", None)
        gmres_max_iter = kwargs.get("gmres_max_iter", 50)
        return_basis = kwargs.get("return_basis", False)
        s = 0.9 + 0.1 * num_t

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
            
            x0_hat = mutils.convert_m0t_to_mst(
                m_0t=x0_hat,
                x_t=x,
                sde=self.sde,
                t=num_t,
                s = s
            )

            x0_hat_obs = self.H_func.H(x0_hat)

            return (x0_hat_obs, flow_pred)

        # x_0_pred, vjp_estimate_x_0, flow_pred = torch.func.vjp(
        #     estimate_x_0, x_t, has_aux=True
        # )
        x0_hat_obs, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        coeff_C_yy = std_t**2 / (alpha_t)

        difference = y_obs - x0_hat_obs

        def cov_y_xt(v):
            return(
                self.noiser.sigma**2 * v
                + self.H_func.H(vjp_estimate_h_x_0(v)[0]) * coeff_C_yy
            )

        grad_ll, V_basis = gmres(
            A=cov_y_xt,
            b=difference,
            maxiter=gmres_max_iter,
        )

        grad_ll = vjp_estimate_h_x_0(grad_ll)[0]

        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None and clamp_condition:
            if num_t < 0.1:
                guided_vec = torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred

            else:
                guided_vec = scaled_grad + flow_pred
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        if not return_basis:
            return guided_vec
        else:
            return guided_vec, V_basis
        

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
        use_svd = kwargs.get("use_svd", False)
        ablate = kwargs.get("ablate", False)
        if use_svd:
            return self._get_alt_guidance(
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
                **kwargs,
            )
        else:
            return self._get_guidance(
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
                **kwargs,
            )
        
@register_guided_sampler(name="tmpd_gmres_ablate")
class TMPD_gmres_ablate(GuidedSampler):
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
        """
        Using GMRES to solve the linear system for the guidance.
        This is the original GMRES algorithm in Rozet et al. 2024.
        However, we observe that it has blurry reconstructions,
        with smoothed out details.
        
        It has a very similar failure mode to the original TMPD
        in Boys et al. 2024, which uses a row-sum approximation. 
        
        We conjecture that this is due to the corruption of the 
        information about forward operator HH^t.  
        """
        data_name = kwargs.get("data_name", None)
        gmres_max_iter = kwargs.get("gmres_max_iter", 50)
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

        x0_hat_obs, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        coeff_C_yy = std_t**2 / (alpha_t)

        difference = y_obs - x0_hat_obs

        def cov_y_xt(v):
            return (
                self.noiser.sigma**2 * v
                + self.H_func.H(vjp_estimate_h_x_0(v)[0]) * coeff_C_yy
            )

        grad_ll, V_basis = gmres(
            A=cov_y_xt,
            b=difference,
            maxiter=gmres_max_iter,
        )

        grad_ll = vjp_estimate_h_x_0(grad_ll)[0]

        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None and clamp_condition:
            if num_t < 0.1:
                guided_vec = torch.clamp(scaled_grad, -clamp_to, clamp_to) + flow_pred

            else:
                guided_vec = scaled_grad + flow_pred
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        if not return_basis:
            return guided_vec
        else:
            return guided_vec, V_basis
        
        

@register_guided_sampler(name="tmpd_h")
class TMPD_hutchinson(GuidedSampler):
    # TMPD does not seem to require any additional hyperparameters
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
        num_hutchinson_samples = kwargs.get("num_hutchinson_samples", 50)
        new_noise = kwargs.get("new_noise", None)
        data_name = kwargs.get("data_name", None)
        gmm_model = kwargs.get("gmm_model", None)
        assert gmm_model is not None, "GMM model must be provided."

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

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

            return x0_hat, flow_pred

        x_0_pred, vjp_estimate_x_0, flow_pred = torch.func.vjp(
            estimate_x_0, x_t, has_aux=True
        )

        coeff_C_yy = std_t**2 / (alpha_t)
        cov_0t = gmm_model.get_cov_0t_batched(num_t, x_t)
        jac_x_0 = cov_0t / coeff_C_yy

        # redefine the function to estimate the Jacobian
        def v_vjp_est(x):
            vjv = torch.einsum(
                "ij, bjk, kl -> bil", self.H_func.V_mat.T, jac_x_0, self.H_func.V_mat
            )
            return torch.einsum("bij, bj -> bi", vjv, x)
        
        diagonal_est = self.parallel_hutchinson_diag_est(
            vjp_est=v_vjp_est,
            shape=(self.shape[0], math.prod(self.shape[1:])),
            num_samples=num_hutchinson_samples,
            chunk_size=num_hutchinson_samples,
        )
        
        # def h_vjp_est(x):
        #     hsh = torch.einsum(
        #         "ij, bjk, kl -> bil",
        #         self.H_func.H_mat,
        #         jac_x_0,
        #         self.H_func.H_mat.T
        #     )
        #     return torch.einsum("bij, bj -> bi", hsh, x)

        # diagonal_est = self.parallel_hutchinson_diag_est(
        #     vjp_est=h_vjp_est,
        #     shape=(self.shape[0], math.prod(y_obs.shape[1:])),
        #     num_samples=num_hutchinson_samples,
        #     chunk_size=num_hutchinson_samples,
        # )
        
        # difference
        # add noise to the observation
        new_noise_std = 0.0
        # y_obs = y_obs + new_noise * new_noise_std
        difference = y_obs - self.H_func.H(x_0_pred)
        
        diagonal_est = (num_t) * diagonal_est + (1 - num_t) * 1.0

        vjp_product = self.H_func.HHt_inv_diag(
            vec=difference,
            diag=coeff_C_yy * diagonal_est,
            sigma_y_2=self.noiser.sigma**2 + new_noise_std**2,
        )
        
        # vjp_product = difference / (diagonal_est + self.noiser.sigma**2)

        grad_ll = vjp_estimate_x_0(self.H_func.Ht(vjp_product))[0]

        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        gamma_t = 1.0

        if clamp_to is not None:
            # guided_vec = (gamma_t * scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)
        return guided_vec


@register_guided_sampler(name="tmpd_trace")
class TMPD_trace(GuidedSampler):
    """
    Same guidance but only use isotropic covariance matrix.
    The variance is computed using the average trace of the Jacobian matrix,
    which is computed using Hutchinson's trace estimator.
    """

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
        **kwargs
    ):
        num_hutchinson_samples = kwargs.get("num_hutchinson_samples", 100)

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        x_t = x_t.clone().detach()

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

            h_x_0_hat = self.H_func.H(x0_hat)

            return h_x_0_hat

        def estimate_x_0(x):
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

            return (x0_hat, flow_pred)

        # this computes a function vjp(u) = u^t @ (∇_x x0_hat)

        _, vjp_estimate_x_0, flow_pred = torch.func.vjp(estimate_x_0, x_t, has_aux=True)

        h_x_0_hat, vjp_estimate_h_x_0 = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=False
        )

        # use Hutchinson's trace estimator to compute the average trace of the Jacobian
        hull_trace = self.parallel_hutchinson_trace_est(
            vjp_estimate_x_0, shape=self.shape, num_samples=num_hutchinson_samples
        )[:, None]

        coeff_C_yy = std_t**2 / (alpha_t)

        # difference
        difference = y_obs - h_x_0_hat

        vjp_product = self.H_func.HHt_inv_diag(
            vec=difference,
            diag=coeff_C_yy * hull_trace,
            sigma_y_2=self.noiser.sigma**2,
        )

        grad_ll = vjp_estimate_h_x_0(vjp_product)[0]

        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())

        # clamp to interval
        if clamp_to is not None:
            # guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
            guided_vec = (scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
        else:
            guided_vec = (scaled_grad) + (flow_pred)
        return guided_vec

    def hutchinson_trace_est(self, vjp_est, shape, num_samples=10):
        """
        Returns the average trace of the Jacobian using Hutchinson's trace estimator.

        Args:
          vjp_est (torch.func.vjp): Function that computes the Jacobian-vector product,
            takes input of shape (B, D)
          shape (tuple): Shape of the Jacobian matrix, (B, *D)
            e.g. (B, 3, 256, 256) for image data.
          num_samples (int): Number of samples to use for the estimator.

        Returns:
          torch.Tensor: shape (batch,), estimated average trace for each batch.
        """
        res = torch.zeros(shape[0], device=self.device)

        for i in range(num_samples):
            z = (
                2 * torch.randint(0, 2, size=(shape[0], *shape[1:]), device=self.device)
                - 1
            )
            # z= torch.randn(shape)
            z = z.float()
            vjpz = vjp_est(z)[0]
            res += (
                torch.sum(vjpz.view(shape[0], -1) * z.view(shape[0], -1), dim=-1)
            ) / math.prod(shape[1:])

        return res / num_samples

    def parallel_hutchinson_trace_est(
        self, vjp_est, shape, num_samples=10, chunk_size=10
    ):
        output = torch.zeros(shape[0], device=self.device)
        assert num_samples % chunk_size == 0
        for i in range(num_samples // chunk_size):
            z = (
                2
                * torch.randint(
                    0, 2, size=(chunk_size, shape[0], *shape[1:]), device=self.device
                )
                - 1
            )
            z = z.float()

            # map across the first dimension
            vmapped_vjp = torch.func.vmap(vjp_est, in_dims=0)(
                z.view(chunk_size, shape[0], *shape[1:])
            )[0]

            vjpz = torch.sum(
                z * vmapped_vjp.view(chunk_size, shape[0], *shape[1:]), dim=0
            )

            output += torch.sum(vjpz.view(shape[0], -1), dim=-1) / math.prod(shape[1:])

        return output / num_samples


# @register_guided_sampler(name="tmpd_ensemble")
class TMPD_ensemble(GuidedSampler):
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
        num_particles=10,
        **kwargs
    ):
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation with diagonal covariance matrix approximation
        using ensemble Kalman filter-like algorithm.
        """
        # record the batch size for reshaping
        batch_size = x_t.shape[0]

        t_batched = torch.ones(batch_size * num_particles, device=self.device) * num_t

        x_t = x_t.clone().detach()

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

        # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of shape (d_y,)
        # so equivalently (∇_x x0_hat) @ H^t @ u
        h_x_0, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        coeff_C_yy = std_t**2 / (alpha_t)

        sigma_0t = coeff_C_yy * self.H_func.H(
            vjp_estimate_h_x_0(torch.ones_like(y_obs))[0]
        )

        C_yy = (sigma_0t + self.noiser.sigma**2).clamp(min=1e-6)

        difference = y_obs - h_x_0

        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]

        gamma_t = 1.0

        scaled_grad = (
            grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t) * gamma_t
        )

        # clamp to interval
        if clamp_to is not None:
            # print(scaled_grad.mean())
            guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        if not self.return_cov:
            return guided_vec
        else:
            x0_pred = convert_flow_to_x0(
                u_t=flow_pred,
                x_t=x_t,
                alpha_t=alpha_t,
                std_t=std_t,
                da_dt=da_dt,
                dstd_dt=dstd_dt,
            )
            assert self.H_func.H_mat is not None

            return guided_vec, x0_pred.mean(axis=0), C_yy.mean(axis=0)

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        return super().guided_euler_sampler(y_obs, z, return_list, clamp_to, **kwargs)


@register_guided_sampler(name="tmpd_cd")
class TMPD_cd(GuidedSampler):
    """
    The continuous time TMPD guidance but with discrete-time modifications.

    Formulation as Equation (17) in the arxiv version of Pokle et al. 2024.
    but we update the conditional expectation E[x_1 | x_t, y] as the DDPM case
    using a Gaussian conditional mean.
    """

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
        **kwargs
    ):
        """
        Returns a tuple (guided vec, x0_hat_pred).
        """
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        x_t = x_t.clone().detach()

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

        h_x_0, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        coeff_C_yy = std_t**2 / (alpha_t)

        coeff_C_yy = math.sqrt(coeff_C_yy)

        C_yy = (
            coeff_C_yy * self.H_func.H(vjp_estimate_h_x_0(torch.ones_like(y_obs))[0])
            + self.noiser.sigma**2
        ).clamp(min=1e-6)

        # difference
        difference = y_obs - h_x_0

        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]

        # NOTE: here we follow the discrete guidance in DDPM
        scaled_grad = grad_ll.detach() * coeff_C_yy

        # convert flow back to x0
        x0_hat_pred = convert_flow_to_x0(
            u_t=flow_pred,
            x_t=x_t,
            alpha_t=alpha_t,
            std_t=std_t,
            da_dt=da_dt,
            dstd_dt=dstd_dt,
        )

        # clamp to interval
        if clamp_to is not None:
            guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to)
        else:
            guided_vec = scaled_grad

        return guided_vec, x0_hat_pred

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        """
        A few slight changes to reparametrise the vector field.
        """
        if return_list:
            samples = []

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

                guided_vec, x0_hat_pred = self.get_guidance(
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

                # forming a new vector field, parametrised by x0_hat_pred
                updated_x0 = x0_hat_pred + guided_vec

                updated_guided_vec = (
                    da_dt * updated_x0 + dstd_dt * (x - alpha_t * updated_x0) / std_t
                )

                print(updated_guided_vec.mean())
                x = (
                    x.detach().clone()
                    + updated_guided_vec * dt
                    + sigma_t
                    * math.sqrt(dt)
                    * torch.randn_like(guided_vec).to(self.device)
                )

                # if return_list and i % (self.sde.sample_N // 10) == 0:
                #     samples.append(x.detach().clone())
                # if i == self.sde.sample_N - 1 and return_list:
                #     samples.append(x.detach().clone())
                if return_list:
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


@register_guided_sampler(name="tmpd_og")
class TMPD_og(GuidedSampler):
    # TMPD does not seem to require any additional hyperparameters
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
        """
        Only difference is the square root in the C_yy calculation.
        """
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        x_t = x_t.clone().detach()

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

        # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of shape (d_y,)
        # so equivalently (∇_x x0_hat) @ H^t @ u
        h_x_0, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )
        # -----------------------------------------
        # Equation 9 in the TMPD paper, we are viewing the
        # covariance matrix of p(x_0|x_t) as fixed w.r.t. x_t (this is only an approximation)
        # so suffice to compute gradient for the scalar term (y-Hx)^t K (y-Hx)
        # which is (∇_x x0_hat) @ H^t @ (coeff * H @ (∇_x x0_hat) @ H^t + sigma_y^2 I)^{-1} @ (y - Hx)
        #
        # Even so, K is still approximated using row sums
        # namely K ≈ diag (H @ (∇_x x0_hat) @ H^t @ 1 + sigma_y^2 * 1)
        # -----------------------------------------

        # change this to see the performance change
        coeff_C_yy = std_t**2 / (alpha_t)

        # NOTE: Simply adding the square root changes a lot!
        # coeff_C_yy = math.sqrt(coeff_C_yy)
        C_yy = (
            coeff_C_yy * self.H_func.H(vjp_estimate_h_x_0(torch.ones_like(y_obs))[0])
            + self.noiser.sigma**2
        ).clamp(min=1e-6)

        # difference
        difference = y_obs - h_x_0

        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]

        # print(grad_ll.mean())

        # compute gamma_t scaling, used in Pokle et al. 2024
        # gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())

        # clamp to interval
        if clamp_to is not None and clamp_condition:
            # clamp_to = flow_pred.flatten().abs().max().item()
            guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
        else:
            guided_vec = (scaled_grad) + (flow_pred)
        return guided_vec


@register_guided_sampler(name="tmpd_fixed_cov")
class TMPD_fixed_cov(GuidedSampler):
    def _get_guidance(
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
        **kwargs
    ):
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation with Equation 9 in paper.
        Full Jacobian approximation but not exact.

        Args:
          - model_fn: model function that takes x_t and t as input and returns the flow prediction
          - x_t: current state x_t ~ p_t(x_t|z, y)
          - num_t: current time step
          - y_obs: observed data
          - alpha_t: alpha_t
          - std_t: std_t, the sigma_t in Pokle et al. 2024
          - da_dt: derivative of alpha w.r.t. t
          - dstd_dt: derivative of std w.r.t. t
          - clamp_to: gradient clipping for the guidance

        Returns:
         - guided_vec: guidance vector with flow prediction and guidance combined
        """
        assert hasattr(
            self.H_func, "H_mat"
        ), "H_func must have H_mat attribute for now."
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def get_x0(x):
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
            # summing over the batch dimension
            # return torch.sum(x0_hat, dim=0)
            return x0_hat, flow_pred

        # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of d_y dim
        # so equivalently (∇_x x0_hat) @ H^t @ u
        jac_x_0_func = torch.func.vmap(
            torch.func.jacrev(get_x0, argnums=0, has_aux=True),
            # in_dims=(0,),
        )

        jac_x_0, flow_pred = jac_x_0_func(x_t)

        # jac_x_0 = torch.autograd.functional.jacobian(get_x0, x_t, create_graph=True).reshape(
        #     x_t.shape[0], x_t.shape[1], x_t.shape[1]
        # )

        flow_pred = model_fn(x_t, t_batched * 999)

        coeff_C_yy = std_t**2 / alpha_t

        # difference
        x_0_hat = convert_flow_to_x0(flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt)
        h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)
        difference = y_obs - h_x_0

        C_yy = (
            coeff_C_yy
            * torch.einsum(
                "ij, bjk, kl -> bil",
                self.H_func.H_mat,
                jac_x_0,
                # torch.diag_embed(torch.diagonal(jac_x_0, dim1=-2, dim2=-1)),
                self.H_func.H_mat.T,
            )
            + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        )

        C_yy_diff = torch.linalg.solve(
            C_yy,
            difference,
        )  # (B, d_y)

        # (B, D, D) @ (D, d_y) @ (B, d_y) -> (B, D)
        grad_ll = torch.einsum(
            "bij, jk, bk -> bi", jac_x_0, self.H_func.H_mat.T, C_yy_diff
        )

        # compute gamma_t scaling
        # gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))
        gamma_t = 1.0

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())
        # clamp to interval
        if clamp_to is not None:
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
            # guided_vec = (gamma_t * scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)
        return guided_vec
        # return flow_pred

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
        **kwargs
    ):
        assert hasattr(
            self.H_func, "H_mat"
        ), "H_func must have H_mat attribute for now."
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        gmm_model = kwargs.get("gmm_model", None)
        assert gmm_model is not None, "GMM model must be provided."

        flow_pred = model_fn(x_t, t_batched * 999)
        coeff_C_yy = std_t**2 / alpha_t

        # difference
        x_0_hat = convert_flow_to_x0(flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt)
        
        # ablation
        s = 0.9 + 0.1 * num_t
        x_0_hat = gmm_model.convert_m0t_to_mst(num_t, x_t, s, x_0_hat)
        
        
        h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)
        difference = y_obs - h_x_0

        cov_x0 = gmm_model.get_cov_0t_batched(num_t, x_t)
        jac_x_0 = cov_x0 / coeff_C_yy

        # # log determinant correction
        # def compute_log_det(x_t):
        #     cov_x0 = gmm_model.get_cov_0t_batched(num_t, x_t)
        #     C_yy = (
        #         torch.einsum(
        #             "ij, bjk, kl -> bil",
        #             self.H_func.H_mat,
        #             cov_x0,
        #             self.H_func.H_mat.T,
        #         )
        #         + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        #     )
        #     log_det = torch.linalg.slogdet(C_yy)[1]
        #     return log_det.sum()

        # grad_log_det = torch.func.grad(compute_log_det)(x_t)
        
        # ablation
        cov_x0 = gmm_model.get_cov_st_batched(num_t, x_t, s)
        
        C_yy = (
            torch.einsum(
                "ij, bjk, kl -> bil",
                self.H_func.H_mat,
                cov_x0,
                self.H_func.H_mat.T,
            )
            + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        )

        C_yy_diff = torch.linalg.solve(
            C_yy,
            difference,
        )  # (B, d_y)

        grad_ll = (
            torch.einsum("bij, jk, bk -> bi", jac_x_0, self.H_func.H_mat.T, C_yy_diff)
            # - 0.5 * grad_log_det
        )

        gamma_t = 1.0
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        if clamp_to is not None:
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)
        return guided_vec

    

@register_guided_sampler(name="tmpd_fixed_diag")
class TMPD_fixed_diag(GuidedSampler):
    def _get_guidance(
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
        **kwargs
    ):
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation with Equation 9 in paper.
        Diagonal Jacobian approximation but not exact.

        I somehow believe the matrix solve is taking too much memory
        and introducing numerical instability.
        """
        assert hasattr(
            self.H_func, "H_mat"
        ), "H_func must have H_mat attribute for now."
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def get_x0(x):
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
            return x0_hat, flow_pred

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

        jac_x_0_func = torch.func.vmap(
            torch.func.jacrev(get_x0, argnums=0, has_aux=True),
            # in_dims=(0,),
        )

        jac_x_0, flow_pred = jac_x_0_func(x_t)

        flow_pred = model_fn(x_t, t_batched * 999)

        coeff_C_yy = std_t**2 / alpha_t

        # difference
        x_0_hat = convert_flow_to_x0(flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt)
        h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)
        difference = y_obs - h_x_0

        diag_jac = torch.diagonal(
            jac_x_0,
            dim1=-2,
            dim2=-1,
        )

        HHt_sigma = (
            torch.einsum(
                "ij, bjk, kl -> bil",
                self.H_func.H_mat,
                diag_jac * coeff_C_yy,
                self.H_func.H_mat.T,
            )
            + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        )

        C_yy_diff = torch.linalg.solve(
            HHt_sigma,
            difference,
        )

        # diag_jac = torch.diagonal(
        #     torch.einsum(
        #         "ij, bjk, kl -> bil", self.H_func.V_mat.T, jac_x_0, self.H_func.V_mat
        #     ),
        #     dim1=-2,
        #     dim2=-1,
        # )

        # C_yy_diff = self.H_func.HHt_inv_diag(
        #     vec=difference,
        #     # diag = torch.diagonal(vjvt, dim1=-2, dim2=-1) * coeff_C_yy,
        #     diag=diag_jac * coeff_C_yy,
        #     sigma_y_2=self.noiser.sigma**2,
        # )

        # compute the guidance term using vjp
        # h_x_0, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
        #     estimate_h_x_0, x_t, has_aux=True
        # )
        # grad_ll = vjp_estimate_h_x_0(C_yy_diff)[0]
        grad_ll = torch.einsum(
            "bij, jk, bk -> bi", jac_x_0, self.H_func.H_mat.T, C_yy_diff
        )

        gamma_t = (1 - num_t) ** 2 / coeff_C_yy
        # gamma_t = 1.0

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())
        # clamp to interval
        if clamp_to is not None:
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
            # guided_vec = (gamma_t * scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)
        return guided_vec

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
        **kwargs
    ):
        assert hasattr(
            self.H_func, "H_mat"
        ), "H_func must have H_mat attribute for now."
        gmm_model = kwargs.get("gmm_model", None)
        assert gmm_model is not None, "GMM model must be provided."

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def get_x0(x):
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
            return x0_hat, flow_pred

        coeff_C_yy = std_t**2 / alpha_t
        cov_0t = gmm_model.get_cov_0t_batched(num_t, x_t)
        jac_x_0 = cov_0t / coeff_C_yy

        flow_pred = model_fn(x_t, t_batched * 999)
        # difference
        x_0_hat = convert_flow_to_x0(flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt)
        h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)
        difference = y_obs - h_x_0

        diag_jac = torch.diagonal(
            torch.einsum(
                "ij, bjk, kl -> bil", self.H_func.V_mat.T, jac_x_0, self.H_func.V_mat
            ),
            dim1=-2,
            dim2=-1,
        )

        C_yy_diff = self.H_func.HHt_inv_diag(
            vec=difference,
            # diag = torch.diagonal(vjvt, dim1=-2, dim2=-1) * coeff_C_yy,
            diag=diag_jac * coeff_C_yy,
            sigma_y_2=self.noiser.sigma**2,
        )

        # diag_cov = torch.diagonal(
        #     cov_0t,
        #     dim1=-2,
        #     dim2=-1,
        # )

        # diag_cov = torch.diag_embed(diag_cov, dim1=-2, dim2=-1)

        # HHt_sigma = (
        #     torch.einsum(
        #         "ij, bjk, kl -> bil",
        #         self.H_func.H_mat,
        #         diag_cov,
        #         self.H_func.H_mat.T,
        #     )
        #     + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        # )

        # C_yy_diff = torch.linalg.solve(
        #     HHt_sigma,
        #     difference,
        # )

        grad_ll = torch.einsum(
            "bij, jk, bk -> bi", jac_x_0, self.H_func.H_mat.T, C_yy_diff
        )

        gamma_t = 1.0
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        if clamp_to is not None:
            # guided_vec = (gamma_t * scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + flow_pred
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)
        return guided_vec


@register_guided_sampler(name="tmpd_ablation")
class TMPD_ablation(GuidedSampler):
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
        **kwargs
    ):
        """
        Ablation study for TMPD guidance for OT path.
        """
        # the GMM model will give the true vector field
        gmm_model = kwargs.get("gmm_model", None)
        if gmm_model is None:
            raise ValueError("GMM model must be provided for ablation study.")

        ######## computing the TMPD guidance ########
        new_noise = kwargs.get("new_noise", None)
        # assert new_noise is not None, "New noise must be provided for ablation study."

        def get_x0(x):
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
            return x0_hat, flow_pred

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        flow_pred = model_fn(x_t, t_batched * 999)

        coeff_C_yy = std_t**2 / alpha_t

        cov_0t = gmm_model.get_cov_0t_batched(num_t, x_t)
        jac_x_0 = cov_0t / coeff_C_yy

        # difference
        x_0_hat = convert_flow_to_x0(flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt)
        h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)

        # add noise to the observation
        # new_noise_sigma = 0.1 * (1 - num_t)
        new_noise_sigma = 0.0
        # new_noise_sigma = 0.5 * num_t * (1 - num_t)
        # new_noise = torch.randn_like(y_obs) * new_noise_sigma
        # y_obs += new_noise * new_noise_sigma

        difference = y_obs - h_x_0

        diag_jac = torch.diagonal(
            torch.einsum(
                "ij, bjk, kl -> bil", self.H_func.V_mat.T, jac_x_0, self.H_func.V_mat
            ),
            dim1=-2,
            dim2=-1,
        )

        # coefficient weighting
        beta = 0.5 * num_t

        C_yy_diff = self.H_func.HHt_inv_diag(
            vec=difference,
            # diag = torch.diagonal(vjvt, dim1=-2, dim2=-1) * coeff_C_yy,
            diag=(diag_jac * coeff_C_yy),
            sigma_y_2=self.noiser.sigma**2 + new_noise_sigma**2,
        )
        grad_ll = torch.einsum(
            "bij, jk, bk -> bi", jac_x_0, self.H_func.H_mat.T, C_yy_diff
        )
        ############################

        # true vector field guidance
        # true_grad = gmm_model.grad_yt(num_t, x_t, y_obs, self.H_func.H_mat, self.noiser.sigma)
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        gamma_t = 1.0
        if clamp_to is not None:
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)

        # return gmm_model.true_vector_field(num_t, x_t, y_obs, self.H_func.H_mat, self.noiser.sigma)
        # if self.ablate:
        #     return guided_vec, scaled_grad
        # else:
        #     return guided_vec
        return guided_vec


@register_guided_sampler(name="true_vec")
class true_vector_field(GuidedSampler):
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
        **kwargs
    ):
        """
        Returns the true vector field for the ablation study.
        """
        gmm_model = kwargs.get("gmm_model", None)
        if gmm_model is None:
            raise ValueError("GMM model must be provided for ablation study.")

        return gmm_model.true_vector_field(
            num_t, x_t, y_obs, self.H_func.H_mat, self.noiser.sigma
        )


@register_guided_sampler(name="tmpd_exact")
class TMPD_exact(GuidedSampler):
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
        **kwargs
    ):
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation with exact second order approximation.

        Args:
          - model_fn: model function that takes x_t and t as input and returns the flow prediction
          - x_t: current state x_t ~ p_t(x_t|z, y)
          - num_t: current time step
          - y_obs: observed data
          - alpha_t: alpha_t
          - std_t: std_t, the sigma_t in Pokle et al. 2024
          - da_dt: derivative of alpha w.r.t. t
          - dstd_dt: derivative of std w.r.t. t
          - clamp_to: gradient clipping for the guidance

        Returns:
         - guided_vec: guidance vector with flow prediction and guidance combined
        """
        assert hasattr(
            self.H_func, "H_mat"
        ), "H_func must have H_mat attribute for now."

        gmm_model = kwargs.get("gmm_model", None)
        assert gmm_model is not None, "GMM model must be provided."

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def log_likelihood_fn(x):
            # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of d_y dim
            # so equivalently (∇_x x0_hat) @ H^t @ u
            flow_pred = model_fn(x, t_batched * 999)
            coeff_C_yy = std_t**2 / alpha_t

            cov_0t = gmm_model.get_cov_0t_batched(num_t, x)
            jac_x_0 = cov_0t / coeff_C_yy

            # difference
            x_0_hat = convert_flow_to_x0(flow_pred, x, alpha_t, std_t, da_dt, dstd_dt)
            h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)

            # coeff_C_yy = math.sqrt(coeff_C_yy)
            C_yy = (
                coeff_C_yy
                * torch.einsum(
                    "ij, bjk, kl -> bil",
                    self.H_func.H_mat,
                    jac_x_0,
                    self.H_func.H_mat.T,
                )
                + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
            )

            log_likelihood = (
                -1 * (self.H_func.H_mat.shape[0] / 2) * math.log(2 * math.pi)
                - 0.5
                * torch.einsum(
                    "bi, bi -> b",
                    y_obs - h_x_0,
                    torch.linalg.solve(C_yy, y_obs - h_x_0),
                )
                - 0.5 * torch.linalg.slogdet(C_yy)[1]
            )

            # only single sample (no sum over batch dimension)
            # log_likelihood = likelihood_distr.log_prob(y_obs)

            return log_likelihood.sum(), C_yy

        # compute gradient of log likelihood
        grad_ll, C_yy = torch.func.grad(log_likelihood_fn, has_aux=True)(x_t)

        flow_pred = model_fn(x_t, t_batched * 999)

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())
        if clamp_to is not None:
            # guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
            guided_vec = (scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
        else:
            guided_vec = (scaled_grad) + (flow_pred)

        if not self.return_cov:
            return guided_vec
        else:
            x0_pred = convert_flow_to_x0(
                u_t=flow_pred,
                x_t=x_t,
                alpha_t=alpha_t,
                std_t=std_t,
                da_dt=da_dt,
                dstd_dt=dstd_dt,
            )
            return guided_vec, x0_pred.mean(axis=0), C_yy.mean(axis=0)

    # def sample(
    #     self,
    #     y_obs,
    #     return_list=False,
    #     method="euler",
    #     clamp_to=1,
    #     starting_time=0,
    #     z=None,
    #     **kwargs
    # ):
    #     """
    #     Overrides the base class method to include the exact guidance for TMPD.
    #     """
    #     return self.chunkwise_sampling(
    #         y_obs=y_obs,
    #         return_list=return_list,
    #         method=method,
    #         clamp_to=clamp_to,
    #         starting_time=starting_time,
    #         z=z,
    #     )


@register_guided_sampler(name="tmpd_row_exact")
class TMPD_row_exact(GuidedSampler):
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
        **kwargs
    ):
        """
        TMPD guidance for OT path, but with row-sum approximation.
        """
        assert hasattr(
            self.H_func, "H_mat"
        ), "H_func must have H_mat attribute for now."

        # if len(x_t.shape) > 1:
        #     raise NotImplementedError("Only single sample supported for now.")

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def log_likelihood_fn(x):
            def estimate_h_x_0(xi):
                flow_pred = model_fn(xi, t_batched * 999)
                # pass to model to get x0_hat prediction
                x0_hat = convert_flow_to_x0(
                    u_t=flow_pred,
                    x_t=xi,
                    alpha_t=alpha_t,
                    std_t=std_t,
                    da_dt=da_dt,
                    dstd_dt=dstd_dt,
                )

                x0_hat_obs = self.H_func.H(x0_hat)

                return (x0_hat_obs, flow_pred)

            h_x_0, vjp_estimate_h_x_0, flow_pred = torch.func.vjp(
                estimate_h_x_0, x_t, has_aux=True
            )

            coeff_C_yy = std_t**2 / alpha_t

            # sigma_0t = coeff_C_yy * self.H_func.H(vjp_estimate_h_x_0(torch.ones_like(y_obs))[0])
            sigma_0t = coeff_C_yy * torch.ones_like(y_obs)

            # coeff_C_yy = math.sqrt(coeff_C_yy)
            C_yy = sigma_0t + self.noiser.sigma**2

            log_likelihood = (
                -1 * (self.H_func.H_mat.shape[0] / 2) * math.log(2 * math.pi)
                - 0.5
                * torch.einsum("bi, bi -> b", y_obs - h_x_0, (y_obs - h_x_0) / C_yy)
                - 0.5 * torch.log(torch.prod(C_yy, dim=1))
            )

            # only single sample (no sum over batch dimension)
            # log_likelihood = likelihood_distr.log_prob(y_obs)

            return log_likelihood.sum(), flow_pred

        # compute gradient of log likelihood
        grad_ll, flow_pred = torch.func.grad(log_likelihood_fn, has_aux=True)(x_t)

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())
        guided_vec = (scaled_grad) + (flow_pred)

        return guided_vec

    def sample(
        self,
        y_obs,
        return_list=False,
        method="euler",
        clamp_to=1,
        starting_time=0,
        z=None,
        **kwargs
    ):
        """
        Overrides the base class method to include the exact guidance for TMPD.
        """
        if z is None:
            start_z = torch.randn(self.shape, device=self.device)
        else:
            start_z = z

        num_samples = y_obs.shape[0]

        chunk_size = 20

        assert num_samples % chunk_size == 0

        num_chunks = num_samples // chunk_size

        # list of samples to store
        # the t^th element of the list is the samples at the time t
        tmpd_samples_dict = {i: [] for i in range(num_chunks)}

        for i in tqdm(
            range(num_chunks), total=num_chunks, desc="TMPD sampling", colour="green"
        ):
            start_z_chunk = start_z[i * chunk_size : (i + 1) * chunk_size, :]
            y_obs_chunk = y_obs[i * chunk_size : (i + 1) * chunk_size, :]

            tmpd_samples_chunk, _ = self.guided_euler_sampler(
                y_obs=y_obs_chunk, clamp_to=None, z=start_z_chunk, return_list=True
            )
            tmpd_samples_dict[i] = tmpd_samples_chunk

        # make the dictionary into a list by stacking
        tmpd_fixed_cov_samples = []

        for i in range(len(tmpd_samples_dict[0])):
            temp = torch.cat(
                [tmpd_samples_dict[chunk_no][i] for chunk_no in range(num_chunks)],
                dim=0,
            )
            tmpd_fixed_cov_samples.append(temp)

        # with open("temp/tmpd_fixed_cov_samples.pkl", "wb") as f:
        #     pickle.dump(tmpd_fixed_cov_samples, f)

        if return_list:
            return tmpd_fixed_cov_samples
        else:
            return tmpd_fixed_cov_samples[-1]


@register_guided_sampler(name="tmpd_d")
class TMPD_d(GuidedSampler):
    """The DDPM implementation of the TMPD guidance function."""

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
        **kwargs
    ):
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        x_t = x_t.clone().detach()

        # print("x_t:", x_t.mean())

        def estimate_h_x_0(x):
            """assume the model fn is a score model instead of a flow model."""
            score_pred = model_fn(x, t_batched * 999)

            # pass to model to get x0_hat prediction
            x0_hat = (1 / alpha_t) * (x + std_t**2 * score_pred)

            x0_hat_obs = self.H_func.H(x0_hat)

            return (x0_hat_obs, score_pred)

        # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of shape (d_y,)
        # so equivalently (∇_x x0_hat) @ H^t @ u
        h_x_0, vjp_estimate_h_x_0, score_pred = torch.func.vjp(
            estimate_h_x_0, x_t, has_aux=True
        )

        # print("score_pred:", score_pred.mean())
        # print("h_x_0:", h_x_0.mean())
        # print(
        #     "vjp_estimate_h_x_0:", vjp_estimate_h_x_0(torch.ones_like(y_obs))[0].mean()
        # )

        # change this to see the performance change
        coeff_C_yy = std_t**2 / (alpha_t)

        C_yy = (
            coeff_C_yy * self.H_func.H(vjp_estimate_h_x_0(torch.ones_like(y_obs))[0])
            + self.noiser.sigma**2
        )

        # print("C_yy:", C_yy.mean())

        # difference
        difference = y_obs - h_x_0
        # print("difference:", difference.mean())

        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]
        # print("grad_ll:", grad_ll.mean())

        # print(grad_ll.mean())

        scaled_grad = grad_ll.detach() * coeff_C_yy

        guided_vec = scaled_grad  # + (x0_hat)

        return guided_vec, score_pred
        # return score_pred

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        if return_list:
            samples = []

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

            eps = 1e-3  # default: 1e-3

            for timestep in range(self.sde.sample_N - 1, 0, -1):
                # sampling steps default to 1000
                num_t = 1 - (
                    timestep / self.sde.sample_N * (self.sde.T - eps) + eps
                )  # scalar time

                m = self.sde.sqrt_alphas_cumprod[timestep]  # sqrt of cumulative alpha_t

                sqrt_1m_alpha = self.sde.sqrt_1m_alphas_cumprod[timestep]
                v = sqrt_1m_alpha**2
                alpha = m**2
                m_prev = self.sde.sqrt_alphas_cumprod_prev[timestep]
                v_prev = self.sde.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
                alpha_prev = m_prev**2

                # guidance x_0t which is equivalent to the term in Bayesian update
                alpha_t = m
                std_t = sqrt_1m_alpha
                da_dt = 1.0  # TODO: the whole guidance function can be changed
                dstd_dt = -1.0  # those two are just placeholders

                # # getting guidance
                guidance_vec, score_pred = self.get_guidance(
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

                # getting from network
                # score_pred_mo = model_fn(x, torch.ones_like(x) * 999 * num_t)
                noise_pred = -math.sqrt(v) * score_pred

                # print("score_pred:", score_pred.mean())

                # x_0 prediction
                # print(m)
                x_0 = (x - sqrt_1m_alpha * noise_pred) / m

                x_0 += guidance_vec

                # uses eta=1.0 as in the paper
                coeff1 = 1.0 * np.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
                coeff2 = np.sqrt(v_prev - coeff1**2)

                x_mean = m_prev * x_0 + coeff2 * noise_pred
                # print("x_mean:", x_mean.mean())

                # print(f"After guidance: {x_mean.mean()}")

                # update x by adding noise
                std = coeff1
                x = x_mean + std * torch.randn_like(x_mean)
                # print(f"New x after noising: {x.mean()}")

                if return_list:
                    samples.append(x_mean.detach().clone())

        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i])
            nfe = self.sde.sample_N
            return samples, nfe
        else:
            assert x_mean is not None
            x_mean = self.inverse_scaler(x_mean)
            nfe = self.sde.sample_N
            return x_mean, nfe


# Below is the code provided by Ben
# -------------------------------------------  TMPD  -------------------------------------------#
# @register_conditioning_method(name="tmp")
# class TweedieMomentProjection(ConditioningMethod):
#     def __init__(self, operator, noiser, **kwargs):
#         super().__init__(operator, noiser)
#         self.num_sampling = kwargs.get("num_sampling", 5)
#         # self.scale = kwargs.get('scale', 1.0)

#     def hutchinson_estimator(
#         self, x_y, measurement, estimate_x_0, r, v, noise_std, **kwargs
#     ):
#         if self.noiser.__name__ == "gaussian":
#             # NOTE: This standing functorch way seems to be only slightly faster (163 seconds instead of 188 seconds)
#             # NOTE: In torch, usually our method is up to 2x slower than dps due to the extra vjp
#             # # h_x_0, vjp_estimate_h_x_0, x_0 = torch.func.vjp(estimate_h_x_0, x_t, has_aux=True)
#             h_x_0, vjp_estimate_h_x_0, x_0 = functorch.vjp(
#                 estimate_h_x_0, x_t, has_aux=True
#             )
#             vmap_h_x_0 = functorch.vmap(h_x_0)
#             vmap_h = functorch.vmap(self.operator.forward(z))
#             vmap_vjp_estimate_h_x_0 = functorch.vmap(vjp_estimate_h_x_0)

#             m = 20
#             z = 2 * torch.randint(low=0, high=1, size=(m,) + jnp.shape(x_0)) - 1
#             h_z = vmap_h(z)
#             hutchinson_diagonal_estimate = vmap_vjp_estimate_h_x_0(h_z)
#             print(hutchin_diagonal_estimate.shape)
#             print(z.shape)

#             # print(h_x_0.shape)
#             # print(x_0.shape)
#             # y = self.operator.forward(torch.ones_like(x_0), **kwargs)
#             # really need some kind of vmap here
#             h_z = self.operator.forward(
#                 z
#             )  # for some types of forward operator there is no point calculating the vjp across values that won't contribute to the diagonal, they will only increase the variance
#             z = vjp_estimate_h_x_0(h_z, **kwargs)  # will give something like (d_y, m)
#             hutchinson_diagonal_estimate = h_z @ z

#             C_yy = (
#                 self.operator.forward(
#                     vjp_estimate_h_x_0(
#                         self.operator.forward(torch.ones_like(x_0), **kwargs)
#                     )[0],
#                     **kwargs,
#                 )
#                 + noise_std**2 / r
#             )
#             difference = measurement - h_x_0
#             norm = torch.linalg.norm(difference)
#             ls = vjp_estimate_h_x_0(difference / C_yy)[0]

#             x_0 = (
#                 x_0 + ls
#             )  # TODO: commenting it out shows that rest of the code works okay
#         else:
#             raise NotImplementedError

#     def conditioning(self, x_t, measurement, estimate_x_0, r, v, noise_std, **kwargs):
#         def estimate_h_x_0(x_t):
#             x_0 = estimate_x_0(x_t)
#             return self.operator.forward(x_0, **kwargs), x_0
#             # return self.operator.forward(x_0, **kwargs)

#         if self.noiser.__name__ == "gaussian":
#             # Due to the structure of this code, the condition operator is not accesible unless inside from in the conditioning method. That's why the analysis is here
#             # Since functorch 1.1.1 is not compatible with this
#             # functorch 0.1.1 (unstable; works with PyTorch 1.11) does not work with autograd.Function, which is what the model is written in. It can be rewritten, or package environment needs to be solved.
#             # h_x_0, vjp = torch.autograd.functional.vjp(estimate_h_x_0, x_t, self.operator.forward(torch.ones_like(x_t), **kwargs))
#             # difference = measurement - h_x_0
#             # norm = torch.linalg.norm(difference)
#             # C_yy = self.operator.forward(vjp, **kwargs) + noise_std**2 / ratio
#             # _, ls = torch.autograd.functional.vjp(estimate_h_x_0, x_t, difference / C_yy)
#             # x_0 = estimate_x_0(x_t)

#             # NOTE: This standing functorch way seems to be only slightly faster (163 seconds instead of 188 seconds)
#             # NOTE: In torch, usually our method is up to 2x slower than dps due to the extra vjp
#             # # h_x_0, vjp_estimate_h_x_0, x_0 = torch.func.vjp(estimate_h_x_0, x_t, has_aux=True)
#             h_x_0, vjp_estimate_h_x_0, x_0 = functorch.vjp(
#                 estimate_h_x_0, x_t, has_aux=True
#             )

#             # -----------------------------------------
#             # Equation 9 in the TMPD paper, we are viewing the
#             # covariance matrix of p(x_0|x_t) as fixed w.r.t. x_t (this is only an approximation)
#             # so suffice to compute gradient for the scalar term (y-Hx)^t K (y-Hx)
#             # where K is the inverse of the covariance matrix
#             # Even so, K is still approximated
#             # -----------------------------------------

#             # print(h_x_0.shape)
#             # print(x_0.shape)
#             # y = self.operator.forward(torch.ones_like(x_0), **kwargs)
#             C_yy = (
#                 self.operator.forward(
#                     vjp_estimate_h_x_0(
#                         self.operator.forward(torch.ones_like(x_0), **kwargs)
#                     )[0],
#                     **kwargs,
#                 )
#                 + noise_std**2 / r
#             )
#             difference = measurement - h_x_0
#             norm = torch.linalg.norm(difference)
#             ls = vjp_estimate_h_x_0(difference / C_yy)[0]

#             x_0 = (
#                 x_0 + ls
#             )  # TODO: commenting it out shows that rest of the code works okay
#         else:
#             raise NotImplementedError

#         return x_0, norm
