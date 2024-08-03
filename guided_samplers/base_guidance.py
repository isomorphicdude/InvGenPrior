"""Implements the base class of algorithms for solving inverse problems with flows."""

from abc import ABC, abstractmethod
import math

from tqdm import tqdm
import torch
from scipy import integrate

# custom imports
from models import utils as mutils


class GuidedSampler(ABC):
    """Base class for guided samplers."""

    def __init__(
        self,
        model,
        sde,
        shape,
        sampling_eps,
        inverse_scaler,
        H_func,
        noiser,
        device,
        return_cov=False,
        **kwargs,
    ):
        self.model = model
        self.sde = sde
        self.shape = shape
        self.sampling_eps = sampling_eps
        self.return_cov = return_cov

        if inverse_scaler is None:
            # if no inverse scaler is provided, default to identity
            inverse_scaler = lambda x: x

        self.inverse_scaler = inverse_scaler
        self.H_func = H_func
        self.noiser = noiser
        self.device = device

    def guided_euler_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        """
        Computes the posterior samples with respect to y_obs using the guided Euler sampler.

        Args:
          y_obs (torch.Tensor): Observed data to condition the sampling on. (B, C*H*W)
          z (torch.Tensor, optional): Optional latent variable for sampling. Default is None. (B, C, H, W)
          return_list (bool, optional): If True, returns the samples as a list. Default is False.
          clamp_to (float, optional): If not None, clamps the scores to this value. Default is 1.
          **kwargs: Additional keyword arguments for the sampling process.

        Returns:
          torch.Tensor or list: The posterior samples. If `return_list` is True, returns intermediate samples as a list.
          int: The number of function evaluations (NFEs) used in the sampling process.

          (list, list, list): The samples, mean_0t, and cov_yt.
        """
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

                if self.return_cov:
                    guided_vec, mean_0t, cov_yt = self.get_guidance(
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

                else:
                    guided_vec = self.get_guidance(
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

                x = (
                    x.detach().clone()
                    + guided_vec * dt
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

                if self.return_cov:
                    list_mean_0t.append(mean_0t)
                    list_cov_yt.append(cov_yt)

                # print name
                if self.__class__.__name__ == "TMPD" and i % 10 == 0 and len(self.shape)>2:
                    print(f"Iteration {i} of {self.sde.sample_N} completed.")
                    
                elif self.__class__.__name__ == "TMPD_trace" and i % 10 == 0 and len(self.shape)>2:
                    print(f"Iteration {i} of {self.sde.sample_N} completed.")
        # print(self.return_cov)
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

    def guided_rk45_sampler(
        self, y_obs, z=None, return_list=False, clamp_to=1, **kwargs
    ):
        """
        Computes the posterior samples with respect to y_obs using the guided RK45 sampler.

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
        with torch.no_grad():
            rtol = atol = self.sde.ode_tol
            # METHOD = "RK45"
            eps = self.sampling_eps

            # Initial sample
            if z is None:
                z0 = self.sde.get_z0(
                    torch.zeros(self.shape, device=self.device), train=False
                ).to(self.device)
                x = z0.detach().clone()
            else:
                x = z

            model_fn = mutils.get_model_fn(self.model, train=False)

            def ode_func(t, x):
                x = (
                    mutils.from_flattened_numpy(x, self.shape)
                    .to(self.device)
                    .type(torch.float32)
                )

                # compute the coefficients required for the guided sampler
                alpha_t = self.sde.alpha_t(t)
                std_t = self.sde.std_t(t)
                da_dt = self.sde.da_dt(t)
                dstd_dt = self.sde.dstd_dt(t)

                guided_vec = self.get_guidance(
                    model_fn,
                    x,
                    t,
                    y_obs,
                    alpha_t,
                    std_t,
                    da_dt,
                    dstd_dt,
                    clamp_to=clamp_to,
                    **kwargs,
                )

                return mutils.to_flattened_numpy(guided_vec)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                fun=ode_func,
                t_span=(eps, self.sde.T),
                y0=mutils.to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method="RK45",
            )
            nfe = solution.nfev

        if return_list:
            result_list = []
            for i in range(solution.y.shape[1]):
                x = (
                    torch.tensor(solution.y[:, i])
                    .reshape(self.shape)
                    .to(self.device)
                    .type(torch.float32)
                )
                x = self.inverse_scaler(x)
                result_list.append(x)
            return result_list, nfe
        else:
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(self.shape)
                .to(self.device)
                .type(torch.float32)
            )

            x = self.inverse_scaler(x)
            return x, nfe

    def sample(
        self,
        y_obs,
        return_list=False,
        method="euler",
        clamp_to=1,
        starting_time=0,
        z=None,
        **kwargs,
    ):
        """
        Samples the solution to the inverse problem using the guided sampler.

        Args:
          - y_obs (torch.Tensor): Observed data to condition the sampling on. (B, C*H*W)
          - return_list (bool, optional): If True, returns the samples as a list. Default is False.
          - method (str, optional): The method to use for sampling. Default is "euler".
          - clamp_to (float, optional): If not None, clamps the guidance scores to this value. Default is 1.
          - starting_time (int, optional): The starting time for the sampling process. Default is 0. If nonzero,
          then the latent z is replaced with a_t y + sigma_t \epsilon.

        Returns:
          - torch.Tensor or list: The posterior samples. If `return_list` is True, returns intermediate samples as a list.
        """
        if starting_time == 0:
            z = z
        else:
            assert self.H_func.__name__ == "Inpainting"
            degraded_image = self.H_func.get_degraded_image(y_obs).detach().clone()
            z = self.sde.alpha_t(starting_time) * degraded_image + self.sde.std_t(
                starting_time
            ) * torch.randn_like(degraded_image).to(self.device)

        if self.return_cov:
            print("Using Euler sampler")
            return self.guided_euler_sampler(
                y_obs, z=z, return_list=return_list, clamp_to=clamp_to, **kwargs
            )
        if method == "euler":
            return self.guided_euler_sampler(
                y_obs, z=z, return_list=return_list, clamp_to=clamp_to, **kwargs
            )[0]
        elif method == "rk45":
            return self.guided_rk45_sampler(
                y_obs, z=z, return_list=return_list, clamp_to=clamp_to, **kwargs
            )[0]
        else:
            raise NotImplementedError(f"Method {method} not yet supported.")

    @abstractmethod
    def get_guidance(self):
        raise NotImplementedError()

    def convert_dict_to_list(self, samples_dict, num_chunks):
        """
        Convert a dictionary of chunks of samples into a list.

        Each entry in the dictionary is a list of samples of length T (timesteps)
        for a given chunk.
        
        Returns a list of samples of length T.
        """
        # make the dictionary into a list by stacking
        samples = []
        # print(samples_dict[0])
        for i in range(len(samples_dict[0])):
            temp = torch.cat(
                [samples_dict[chunk_no][i] for chunk_no in range(num_chunks)],
                dim=0,
            )
            samples.append(temp)
        return samples
    
    
    def aggregate_cov_dict(self, cov_dict, num_chunks=50, chunk_size=20):
        """
        Aggregate the covariance dictionary into a list by taking average over 
        the batch dimension.
        """
        cov_list = []
        
        for i in range(len(cov_dict[0])):
            # print(cov_dict[0][i][None].shape)
            temp = torch.cat(
                [cov_dict[chunk_no][i][None] for chunk_no in range(num_chunks)],
                dim=0,
            )
            # print(temp.shape)
            cov_list.append(torch.mean(temp, dim=0))
            
        return cov_list
        

    def chunkwise_sampling(
        self,
        y_obs,
        chunk_size=20,
        return_list=False,
        method="euler",
        clamp_to=None,
        starting_time=0,
        z=None,
        **kwargs,
    ):
        if z is None:
            start_z = torch.randn(self.shape, device=self.device)
        else:
            start_z = z

        num_samples = y_obs.shape[0]
        assert num_samples % chunk_size == 0
        
        num_chunks = num_samples // chunk_size
        
        samples_dict = {i: [] for i in range(num_chunks)}
        
        if self.return_cov:
            mean_0t_dict = {i: [] for i in range(num_chunks)}
            cov_yt_dict = {i: [] for i in range(num_chunks)}
            
        for i in tqdm(
            range(num_chunks), total=num_chunks, desc="Sampling", colour="green"
        ):
            start_z_chunk = start_z[i * chunk_size : (i + 1) * chunk_size, :]
            y_obs_chunk = y_obs[i * chunk_size : (i + 1) * chunk_size, :]

            if not self.return_cov:
                samples_chunk, _ = self.guided_euler_sampler(
                    y_obs=y_obs_chunk, clamp_to=None, z=start_z_chunk, return_list=True
                )
            else:
                samples_chunk, list_mean_0t_chunk, list_cov_yt_chunk = self.guided_euler_sampler(
                    y_obs=y_obs_chunk, clamp_to=None, z=start_z_chunk, return_list=True
                )
                mean_0t_dict[i] = list_mean_0t_chunk
                cov_yt_dict[i] = list_cov_yt_chunk
                
            samples_dict[i] = samples_chunk
                
        samples = self.convert_dict_to_list(samples_dict, num_chunks)
        
        if self.return_cov:
            mean_0t = self.aggregate_cov_dict(mean_0t_dict, num_chunks, chunk_size)
            cov_yt = self.aggregate_cov_dict(cov_yt_dict, num_chunks, chunk_size)
        
        if not self.return_cov:
            if return_list:
                return samples
            else:
                return samples[-1]
        else:
            return samples, mean_0t, cov_yt
                