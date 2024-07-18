"""Implements the TMPD guidance functions."""

import math
import torch
import functorch
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
        # coeff_C_yy = std_t**2 / alpha_t
        coeff_C_yy = 1.0
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
        
        C_yy = (
            coeff_C_yy
            * self.H_func.H(
                vjp_estimate_h_x_0(
                   torch.ones_like(y_obs)
                )[0]
            )
            + self.noiser.sigma**2
        )
        
        # difference
        difference = y_obs - h_x_0
        
        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]
        
        # print(grad_ll.mean())
        
        # compute gamma_t scaling, used in Pokle et al. 2024
        gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))
        
        # TMPD does not seem to require this
        gamma_t = 1.0

        # scale gradient for flows
        # TODO: implement this as derivatives for more generality
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # print(scaled_grad.mean())

        # clamp to interval
        if clamp_to is not None:
            guided_vec = (scaled_grad * gamma_t).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
        else:
            guided_vec = (scaled_grad * gamma_t) + (flow_pred)
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
        
        coeff_C_yy = 1.0
        
        C_yy = coeff_C_yy* torch.einsum(
                "ij, bjk, kl -> bil",
                self.H_func.H_mat,
                jac_x_0,
                self.H_func.H_mat.T,
            )+ self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]

        
        C_yy_diff = torch.linalg.solve(
            C_yy,
            difference,
        ) # (B, d_y)

        # (B, D, D) @ (D, d_y) @ (B, d_y) -> (B, D)
        grad_ll = torch.einsum(
            "bij, jk, bk -> bi", jac_x_0, self.H_func.H_mat.T, C_yy_diff
        )

        # compute gamma_t scaling
        gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))
        
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
    
# @register_guided_sampler(name="tmpd_exact")
# class TMPD_exact(GuidedSampler):
#     def get_guidance(
#         self,
#         model_fn,
#         x_t,
#         num_t,
#         y_obs,
#         alpha_t,
#         std_t,
#         da_dt,
#         dstd_dt,
#         clamp_to,
#         **kwargs
#     ):
#         """
#         TMPD guidance for OT path.
#         Returns ∇ log p(y|x_t) approximation with exact second order approximation.
        
#         Args:   
#           - model_fn: model function that takes x_t and t as input and returns the flow prediction
#           - x_t: current state x_t ~ p_t(x_t|z, y)
#           - num_t: current time step
#           - y_obs: observed data
#           - alpha_t: alpha_t
#           - std_t: std_t, the sigma_t in Pokle et al. 2024
#           - da_dt: derivative of alpha w.r.t. t
#           - dstd_dt: derivative of std w.r.t. t
#           - clamp_to: gradient clipping for the guidance
          
#         Returns:  
#          - guided_vec: guidance vector with flow prediction and guidance combined
#         """
#         assert hasattr(
#             self.H_func, "H_mat"
#         ), "H_func must have H_mat attribute for now."
#         t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

#         x_t.requires_grad_()
        
#         def get_x0(x):
#             flow_pred = model_fn(x, t_batched * 999)

#             # pass to model to get x0_hat prediction
#             x0_hat = convert_flow_to_x0(
#                 u_t=flow_pred,
#                 x_t=x,
#                 alpha_t=alpha_t,
#                 std_t=std_t,
#                 da_dt=da_dt,
#                 dstd_dt=dstd_dt,
#             )
#             return x0_hat, flow_pred

#         # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of d_y dim
#         # so equivalently (∇_x x0_hat) @ H^t @ u
#         jac_x_0_func = torch.func.vmap(
#             torch.func.jacrev(get_x0, argnums=0, has_aux=True),
#             # in_dims=(0,),
#         )

#         jac_x_0, flow_pred = jac_x_0_func(x_t)

#         coeff_C_yy = std_t**2 / alpha_t

#         # difference
#         x_0_hat = convert_flow_to_x0(flow_pred, x_t, alpha_t, std_t, da_dt, dstd_dt)
#         h_x_0 = torch.einsum("ij, bj -> bi", self.H_func.H_mat, x_0_hat)
        
#         C_yy = coeff_C_yy* torch.einsum(
#                 "ij, bjk, kl -> bil",
#                 self.H_func.H_mat,
#                 jac_x_0,
#                 self.H_func.H_mat.T,
#             )+ self.noiser.sigma**2 * torch.eye(self.H_func.H_mat.shape[0])[None]
        
#         # create distribution instance
#         likelihood_distr = torch.distributions.MultivariateNormal(
#             loc = h_x_0, # (B, d_y)
#             covariance_matrix=C_yy
#         )
        
#         likelihood = likelihood_distr.log_prob(y_obs).sum().requires_grad_()
        
#         grad_ll = torch.autograd.grad(
#             likelihood,
#             x_t,
#         )[0]
        

#         # compute gamma_t scaling
#         gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))

#         # scale gradient for flows
#         # TODO: implement this as derivatives for more generality
#         scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

#         # clamp to interval
#         if clamp_to is not None:
#             guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
#                 flow_pred
#             )
#             # guided_vec = (gamma_t * scaled_grad + flow_pred).clamp(-clamp_to, clamp_to)
#         else:
#             guided_vec = (gamma_t * scaled_grad.squeeze(-1)) + (flow_pred)
#         return guided_vec
    
    


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