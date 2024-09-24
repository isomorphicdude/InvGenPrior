"""
Rewritten using torch. 
Based on https://github.com/bd3dowling/diffusionlib.
"""

import math
import torch
import numpy as np

from models import utils as mutils
from guided_samplers.base_guidance import GuidedSampler
from guided_samplers.registry import register_guided_sampler


# will override the sample directly
# this uses diffusion model (discrete-time)
@register_guided_sampler(name="smcdiffopt")
class SMCDiffOpt(GuidedSampler):
    def get_guidance(self):
        raise NotImplementedError("This method is not implemented for SMC methods.")

    # def _construct_obs_sequence(self, init_obs):
    #     y_s = init_obs * self.sde.sqrt_alphas_cumprod[:, None]

    #     # NOTE: for particle filter since needs to run "backwards" (from T -> 0)
    #     return y_s[::-1]

    def _batch_mvn_logpdf(self, x, loc, isotrpoic_cov):
        """
        Returns the log probability of a batch of samples x from a multivariate normal distribution.
        Assumes isotropic covariance.
        """
        # assume x has shape (batch, dim)
        # mean has shape (dim, )
        # cov has shape (, ), scalar
        n_samples = x.shape[0]
        dim = x.shape[1]

        # norm (x - mean) @ cov^-1 @ (x - mean)
        norm = torch.sum((x - loc[None, :]) ** 2, axis=1) / isotrpoic_cov
        log_prob = (
            -0.5 * (dim * math.log(2 * math.pi))
            - 0.5 * dim * math.log(isotrpoic_cov)
            - 0.5 * norm
        )
        return log_prob

    def distr_p_T(self, x_T):
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(x_T), covariance_matrix=torch.eye(x_T.shape[-1])
        )

    def get_timestep(self, t, t0, t1, num_steps):
        return torch.round((t - t0) * (num_steps - 1) / (t1 - t0)).to(torch.int32)

    def _get_times(self, num_steps):
        """"""
        ts, dt = np.linspace(0.0, 1.0, num_steps + 1, retstep=True)
        ts = ts[1:].reshape(-1, 1)
        return ts.flatten(), dt

    def batch_mul(self, x, y):
        """
        Multiply tensor x and y element-wise, where y is shape (batch,).
        """
        y_shape = [y.shape[0]] + [1] * (x.dim() - 1)
        y = y.view(*y_shape)
        return x * y

    # from smcdiffopt diffusionlib
    def proposal_X_t(self, num_t, x_t, eps_pred):
        """
        Sample x_{t-1} from x_{t} in the diffusion model.
        Args:
            num_t (float): time step
            x_t (torch.Tensor): x_t
            eps_pred (torch.Tensor): epsilon_t

        Returns:
            x_{t-1} (torch.Tensor): x_{t-1}
            x_mean (torch.Tensor): mean of x_{t-1}
        """
        # eta = 1.0 corresponds to DDPM-VP, change later
        # TODO: change this to a parameter
        eta = 1.0
        ts, dt = self._get_times(self.sde.N)
        timestep = self.get_timestep(num_t * torch.ones(1), ts[0], ts[-1], self.sde.N)
        m = self.sde.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = (self.sde.sqrt_1m_alphas_cumprod[timestep])

        v = (sqrt_1m_alpha**2)

        alpha_cumprod = self.sde.alphas_cumprod[timestep]

        alpha_cumprod_prev = self.sde.alphas_cumprod_prev[timestep]

        m_prev = self.sde.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sde.sqrt_1m_alphas_cumprod_prev[timestep] ** 2

        x_0 = (x_t - sqrt_1m_alpha.to(x_t.device) * eps_pred) / m.to(x_t.device)

        coeff1 = (
            torch.sqrt((v_prev / v) * (1 - alpha_cumprod / alpha_cumprod_prev)) * eta
        )
        coeff2 = torch.sqrt(v_prev - coeff1**2)
        x_mean = m_prev.to(x_t.device) * x_0 + coeff2.to(x_t.device) * eps_pred
        std = coeff1.to(x_t.device)

        new_x = x_mean + std * torch.randn_like(x_mean)
        return new_x, x_mean
        # dt = -1.0 / self.sde.N
        # z = torch.randn_like(x_t)
        # drift, diffusion = self.sde.reverse(self.model, probability_flow=False).sde(
        #     x_t, timestep
        # )
        # x_mean = x_t + drift * dt

        # x = x_mean + diffusion * z * math.sqrt(-dt)
        # return x, x_mean

    def anneal_scheme(self, t):
        # for inverse problems
        return 1.0

    def log_gauss_liklihood(self, x_t, y_t, c_t, d_t):
        """
        Computes (22) in thesis:
        N(y_t; mean=Ax_t, cov = c_t^2 * sigma_y^2 I + d_t^2 A A^T)

        Args:
            x_t (torch.Tensor): shape (batch*num_particles, dim_x)
            y_t (torch.Tensor): shape (batch, dim_y)
            c_t (float): drift
            d_t (float): diffusion
        Returns:
            log_prob (torch.Tensor): shape (batch, )
        """
        sigma_y = self.noiser.sigma

        # noiseless may cause division by zero
        modified_singulars = c_t**2 * sigma_y**2 + d_t**2 * self.H_func.add_zeros(
            self.H_func.singulars() ** 2
        )

        logdet = torch.sum(torch.log(modified_singulars))

        # matrix vector product of (Cov)^-1 @ (y - Ax)
        diff = (
            y_t.unsqueeze(1)
            - self.H_func.H(x_t).reshape(y_t.shape[0], -1, y_t.shape[1])
        ).reshape(
            -1, y_t.shape[1]
        )  # (batch*num_particles, dim_y)

        cov_y_xt = self.H_func.HHt_inv(
            vec=diff,
            r_t_2=d_t**2,
            sigma_y_2=c_t**2 * sigma_y**2,
        )  # (batch*num_particles, dim_y)

        norm_diff = torch.sum(diff * cov_y_xt, dim=1)

        return -0.5 * logdet - 0.5 * norm_diff

    # this function is kept for later use (for now it is ad hoc)
    def log_potential(self, x_new, x_old, y_new, y_old, c_new, c_old, d_new, d_old):
        """
        Computes the log G(x_t, x_{t+1}) in FK model.

        Args:
            x_new (torch.Tensor): shape (batch*num_particles, dim_x)
            x_old (torch.Tensor): shape (batch*num_particles, dim_x)
            y_new (torch.Tensor): shape (batch, dim_y)
            y_old (torch.Tensor): shape (batch, dim_y)
        """

        numerator = self.log_gauss_liklihood(x_new, y_new, c_new, d_new)
        denominator = self.log_gauss_liklihood(x_old, y_old, c_old, d_old)

        return numerator - denominator
    
    def _systematic_resample(self, weights):
        """
        Perform systematic resampling of the given weights.
        
        Args:
            weights (torch.Tensor): shape (N, )
        Returns:
            indexes (torch.Tensor): shape (N, )
        """
        N = len(weights)

        positions = ((torch.rand(1) + torch.arange(N)) / N).to(weights.device)

        #indexes = np.zeros(N, 'i')
        indexes = torch.zeros(N, dtype=torch.int32, device=weights.device)
        cumulative_sum = torch.cumsum(weights,dim=0)
        
        # i, j = 0, 0
        # while i < N:
        #     if positions[i] < cumulative_sum[j]:
        #         indexes[i] = j
        #         i += 1
        #     else:
        #         j += 1
        # NOTE: here cumsum is an increasing sequence
        # the while loop above starts by finding the first element
        # in positions that is less than the min of cumsum
        # resulting in indexes such that each element of it gives the position 
        # to insert position into cumsum to maintain the increasing order
        indexes = torch.searchsorted(cumulative_sum, positions)
        return indexes
    
    def _multinomial_resample(self, weights):
        """
        Perform multinomial resampling of the given weights.
        
        Args:
            weights (torch.Tensor): shape (N, )
        Returns:
            indexes (torch.Tensor): shape (N, )
        """
        N = len(weights)
        indexes = torch.multinomial(weights, N, replacement=True)
        return indexes
    
    def resample(self, weights, method="systematic"):
        if method == "systematic":
            return self._systematic_resample(weights)
        elif method == "multinomial":
            return self._multinomial_resample(weights)
        else:
            raise ValueError("Invalid resampling method.")
    

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
        Returns both samples and weights.
        """
        samples = []

        # create y_obs sequence for filtering
        # data = self._construct_obs_sequence(y_obs)
        num_particles = kwargs.get("num_particles", 10)
        score_output = kwargs.get("score_output", False)

        ts, dt = self._get_times(self.sde.N)
        reverse_ts = ts[::-1]
        c_t_func = lambda t: self.sde.sqrt_alphas_cumprod[-(t + 1)]
        # c_t_prev_func = lambda t: self.sde.sqrt_alphas_cumprod_prev[-(t + 1)]
        c_t_prev_func = lambda t: self.sde.sqrt_alphas_cumprod[-(t)]
        d_t_func = lambda t: self.sde.sqrt_1m_alphas_cumprod[-(t + 1)]
        # d_t_prev_func = lambda t: self.sde.sqrt_1m_alphas_cumprod_prev[-(t + 1)]
        d_t_prev_func = lambda t: self.sde.sqrt_1m_alphas_cumprod[-(t)]

        model_fn = mutils.get_model_fn(self.model, train=False)
        # flattened initial x, shape (batch * num_particles, dim_x)
        # where for images dim = 3*256*256
        x_t = torch.randn(self.shape[0] * num_particles, np.prod(self.shape[1:]), device=self.device)

        with torch.no_grad():
            for i, num_t in enumerate(reverse_ts):
                print(f"Sampling at time {num_t}.")
                y_new = y_obs * c_t_func(i)  # (batch, dim_y)

                y_old = y_obs * c_t_prev_func(i)  # (batch, dim_y)

                vec_t = (torch.ones(self.shape[0]) * (reverse_ts[i-1])).to(x_t.device)

                # get model prediction
                # assume input is (N, 3, 256, 256)
                # here N = batch * num_particles
                model_input_shape = (self.shape[0] * num_particles, *self.shape[1:])

                if score_output:
                    eps_pred = (
                        model_fn(x_t.view(model_input_shape), vec_t) * (-1) * d_t_func(i)
                    )  # (batch * num_particles, 3, 256, 256)
                else:
                    eps_pred = model_fn(x_t.view(model_input_shape), vec_t)
                    if eps_pred.shape[1] == 2 * self.shape[1]:
                        eps_pred, model_var_values = torch.split(eps_pred, self.shape[1], dim=1)

                x_new, x_mean_new = self.proposal_X_t(
                    num_t, x_t.view(model_input_shape), eps_pred
                )  # (batch * num_particles, 3, 256, 256)
                
                x_new = x_new.clamp(-clamp_to, clamp_to)

                x_input_shape = (self.shape[0] * num_particles, -1)
                # log_weights = self.log_potential(
                #     x_new.view(x_input_shape),
                #     x_t.view(x_input_shape),
                #     y_new,
                #     y_old,
                #     c_new=c_t_func(i),
                #     c_old=c_t_prev_func(i),
                #     d_new=d_t_func(i),
                #     d_old=d_t_prev_func(i),
                # ).view(self.shape[0], num_particles)
                

                # # normalise weights
                # log_weights = log_weights - torch.logsumexp(
                #     log_weights, dim=1, keepdim=True
                # )

                # if i != len(reverse_ts) - 1:
                #     resample_idx = self.resample(
                #         torch.exp(log_weights).view(-1)
                #     )
                #     x_new = (x_new.view(self.shape[0], num_particles, -1))[
                #         torch.arange(self.shape[0])[:, None], resample_idx.unsqueeze(0)
                #     ]

                x_t = x_new
                log_weights = torch.tensor([0.0])
                if return_list:
                    samples.append(
                            x_t.reshape(self.shape[0] * num_particles, *self.shape[1:])
                    )
                    # samples.append(
                    #     x_t.view(num_particles, self.shape[0], *self.shape[1:])[-1]
                    # )

        # average and apply inverse scaler
        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i])
            return samples, torch.exp(log_weights)
        else:
            return (
                self.inverse_scaler(
                    x_t.view(num_particles, self.shape[0], *self.shape[1:])[0]
                ),
                torch.exp(log_weights),
            )
