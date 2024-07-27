"""Implements the TMPD guidance functions."""

import math

import pickle
import torch
import functorch
import numpy as np
from tqdm import tqdm

import models.utils as mutils
from models.utils import convert_flow_to_x0
from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler


@register_guided_sampler(name="tmpd")
class TMPD(GuidedSampler):
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
        **kwargs
    ):
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation with row-sum approximation.

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
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        x_t = x_t.clone().detach()
        
        print("Clamp to:", clamp_to)

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
        )
        # C_yy = (
        #     coeff_C_yy * self.H_func.H(vjp_estimate_h_x_0(
        #         self.H_func.H(torch.ones_like(flow_pred))
        #         )[0])
        #     + self.noiser.sigma**2
        # )
        
        # C_yy = (
        #     coeff_C_yy * torch.ones_like(y_obs)
        #     + self.noiser.sigma**2
        # )
        # print(C_yy.mean())

        # difference
        difference = y_obs - h_x_0

        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]

        # print(grad_ll.mean())

        # compute gamma_t scaling, used in Pokle et al. 2024
        # gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None:
            guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
        else:
            guided_vec = (scaled_grad) + (flow_pred)
            
        return guided_vec
        # return flow_pred


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
            guided_vec = (scaled_grad)

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
                
                updated_guided_vec = da_dt * updated_x0 + dstd_dt * (x - alpha_t * updated_x0) / std_t

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
        # print(coeff_C_yy)
        # coeff_C_yy = 1.0
        # coeff_C_yy = std_t / alpha_t
        # coeff_C_yy = std_t**2 / math.sqrt(alpha_t)

        # C_yy = (
        #     coeff_C_yy
        #     * self.H_func.H(
        #         vjp_estimate_h_x_0(
        #             self.H_func.H(
        #                 torch.ones_like(
        #                     flow_pred
        #                 )  # why not just torch.ones_like(H_func.H(flow_pred))?
        #             )  # answer is sparsity? (from Ben Boys)
        #         )[0]
        #     )
        #     + self.noiser.sigma**2
        # )

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
        if clamp_to is not None:
            guided_vec = (scaled_grad).clamp(-clamp_to, clamp_to) + (flow_pred)
        else:
            guided_vec = (scaled_grad) + (flow_pred)
        return guided_vec


@register_guided_sampler(name="tmpd_fixed_cov")
class TMPD_fixed_cov(GuidedSampler):
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
                self.H_func.H_mat.T,
            )
            + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        )

        C_yy_diff = torch.linalg.solve(
            C_yy,
            difference,
        )  # (B, d_y)
        
        # C_yy_diff = difference / torch.diag(C_yy).reshape(-1, 1)

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

        I only implement this for a single sample for now (batch size 1)

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

        # if len(x_t.shape) > 1:
        #     raise NotImplementedError("Only single sample supported for now.")

        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        # def log_likelihood_fn(x):
        #     def get_x0(x):
        #         flow_pred = model_fn(x, t_batched * 999)

        #         # pass to model to get x0_hat prediction
        #         x0_hat = convert_flow_to_x0(
        #             u_t=flow_pred,
        #             x_t=x,
        #             alpha_t=alpha_t,
        #             std_t=std_t,
        #             da_dt=da_dt,
        #             dstd_dt=dstd_dt,
        #         )
        #         return x0_hat, flow_pred

        #     # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of d_y dim
        #     # so equivalently (∇_x x0_hat) @ H^t @ u
        #     jac_x_0_func = torch.func.jacrev(get_x0, argnums=0, has_aux=True)

        #     jac_x_0, flow_pred = jac_x_0_func(x)

        #     coeff_C_yy = std_t**2 / alpha_t

        #     # difference
        #     x_0_hat = convert_flow_to_x0(flow_pred, x, alpha_t, std_t, da_dt, dstd_dt)
        #     h_x_0 = torch.matmul(self.H_func.H_mat, x_0_hat)

        #     C_yy = (
        #         coeff_C_yy * self.H_func.H_mat @ jac_x_0 @ self.H_func.H_mat.T
        #         + self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        #     )

        #     # create distribution instance
        #     likelihood_distr = torch.distributions.MultivariateNormal(
        #         loc=h_x_0, covariance_matrix=C_yy
        #     )

        #     # only single sample (no sum over batch dimension)
        #     likelihood = likelihood_distr.log_prob(y_obs)

        #     return likelihood.squeeze()

        # # compute gradient of log likelihood
        # grad_ll = torch.func.grad(log_likelihood_fn)(x_t)

        def log_likelihood_fn(x):
            def get_x0(xi):
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
                return x0_hat, flow_pred

            # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of d_y dim
            # so equivalently (∇_x x0_hat) @ H^t @ u
            jac_x_0_func = torch.func.vmap(
                torch.func.jacrev(get_x0, argnums=0, has_aux=True)
            )

            jac_x_0, flow_pred = jac_x_0_func(x)

            coeff_C_yy = std_t**2 / alpha_t

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

            # create distribution instance
            # likelihood_distr = torch.distributions.MultivariateNormal(
            #     loc=h_x_0, covariance_matrix=C_yy,
            #     validate_args=False
            # )
            
            log_likelihood = -1 * (self.H_func.H_mat.shape[0] / 2) * math.log(2*math.pi) - 0.5 * torch.einsum(
                    "bi, bi -> b", y_obs - h_x_0, torch.linalg.solve(C_yy, y_obs - h_x_0)) - 0.5 * torch.linalg.slogdet(C_yy)[1]

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
