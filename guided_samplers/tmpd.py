"""Implements the guidance functions."""

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
        self, model_fn, x_t, num_t, y_obs, alpha_t, std_t, clamp_to, **kwargs
    ):
        """
        TMPD guidance for OT path.
        Returns ∇ log p(y|x_t) approximation.
        """
        t_batched = torch.ones(x_t.shape[0], device=self.device) * num_t

        def estimate_h_x_0(x):
            flow_pred = model_fn(x, t_batched * 999)

            # pass to model to get x0_hat prediction
            x0_hat = convert_flow_to_x0(
                u_t=flow_pred,
                x_t=x_t,
                t=num_t,
                alpha_t=alpha_t,
                std_t=std_t,
            )

            x0_hat_obs = self.H_func.H(x0_hat)

            return x0_hat_obs, flow_pred

        # this computes a function vjp(u) = u^t @ H @ (∇_x x0_hat), u of d_y dim
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
        # namely K \approx diag (H @ (∇_x x0_hat) @ H^t @ 1 + sigma_y^2 * 1)
        # -----------------------------------------
        coeff_C_yy = std_t**2 / math.sqrt(alpha_t)
        C_yy = (
            coeff_C_yy
            * self.H_func.H(
                vjp_estimate_h_x_0(
                    self.H_func.H(
                        torch.ones_like(
                            flow_pred
                        )  # why not just torch.ones_like(H_func.H(flow_pred))?
                    )  # answer is sparsity? (from Ben Boys)
                )[0]
            )
            + self.noiser.sigma**2
        )

        # difference
        difference = y_obs - h_x_0

        # norm = torch.linalg.norm(difference)

        grad_ll = vjp_estimate_h_x_0(difference / C_yy)[0]

        # compute gamma_t scaling
        gamma_t = math.sqrt(alpha_t / (alpha_t**2 + std_t**2))

        # scale gradient for flows
        scaled_grad = grad_ll.detach() * (std_t**2) * (1 / alpha_t + 1 / std_t)

        # clamp to interval
        if clamp_to is not None:
            guided_vec = (gamma_t * scaled_grad).clamp(-clamp_to, clamp_to) + (
                flow_pred
            )
        else:
            guided_vec = (gamma_t * scaled_grad) + (flow_pred)
        return guided_vec


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
