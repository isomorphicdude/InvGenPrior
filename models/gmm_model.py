"""Implements the Gaussian Mixture Model (GMM) class."""

import math

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import sde_lib
from models.utils import convert_x0_to_flow, convert_flow_to_x0, convert_score_to_flow


class GMM(object):
    """
    A wrapper for the GMM model.

    Attributes:
        - dim (int): The dimension of each component of the GMM.
        - n_components (int): The number of components in the GMM.
        - weights (torch.Tensor): The weights of the GMM, shape (n_components,).
        - means (torch.Tensor): The means of the GMM.
        - sde: SDE object for flows/diffusions.
    """

    def __init__(
        self, dim=4, n_components=25, weights=torch.ones(25) / 25, means=None, sde=None
    ):
        assert n_components == len(weights)
        assert dim % 2 == 0  # for the mean specification in the paper
        self.dim = dim
        self.n_components = n_components
        self.weights = weights
        self.mixture_distr = torch.distributions.Categorical(
            probs=self.weights, validate_args=False
        )

        if means is None:
            default_set = torch.arange(-2, 3) * 8.0
            # default_set = torch.arange(-2, 3) * torch.arange(1, 6) * 8.0
            self.means = torch.cartesian_prod(default_set, default_set).repeat(
                1, dim // 2
            )
            # print(self.means)
        else:
            self.means = means

        if sde is None:
            self.sde = sde_lib.RectifiedFlow(
                init_type="gaussian",
                noise_scale=1.0,
                sample_N=100,
            )
        else:
            self.sde = sde

        self.component_distr = torch.distributions.MultivariateNormal(
            loc=self.means,
            covariance_matrix=torch.eye(dim).repeat(n_components, 1, 1),
            validate_args=False,
        )

        self.prior_distr = torch.distributions.MixtureSameFamily(
            component_distribution=self.component_distr,
            mixture_distribution=self.mixture_distr,
            validate_args=False,
        )

    def prior_distr_t(self, t):
        """
        Returns the prior distribution at time t, which is a mixture
        of Gaussians with degraded mean and unit variance.

        Args:
            - t (float): the discretized time step.

        NOTE: the a_t coeff in DDPM/IM is the square root of alpha_t while
        it is just t in rectified flows.
        """
        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)

        component_t = torch.distributions.MultivariateNormal(
            loc=a_t * self.means,
            covariance_matrix=torch.eye(self.dim).repeat(self.n_components, 1, 1)
            * (a_t**2 + std_t**2),
            validate_args=False,
        )
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=self.mixture_distr,
            component_distribution=component_t,
            validate_args=False,
        )

    def score_prior_t(self, x_t, t):
        """
        Returns the gradient of the log prior at time t wrt. x_t.

        Args:
            - x_t (torch.Tensor): the observation at time t.
            - t (float): the discretized time step.
        """

        def p_t(x):
            return self.prior_distr_t(t).log_prob(x).sum()

        score = torch.func.grad(p_t)(x_t)

        return score

        # with torch.enable_grad():
        #     x_t.requires_grad_(True)
        #     output_val = p_t(x_t)

        # return torch.autograd.grad(output_val, x_t, retain_graph=True)[0].detach()

        # a_t = self.sde.alpha_t(t)

        # def p_t(x):
        #     return self.prior_distr_t(a_t).log_prob(x)

        # # use vmap instead
        # return torch.func.vmap(torch.func.grad(p_t))(x_t)

    def x0_pred(self, x_t, t):
        """
        Returns the prediction of x_0 given x_t at time t.

        Args:
            - x_t (torch.Tensor): the observation at time t.
            - t (float): the discretized time step.
        """

        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)

        return (1 / a_t) * (x_t + std_t**2 * self.score_prior_t(x_t, t))

    def flow_pred(self, x_t, t):
        """
        Returns the flow vector field given x_t at time t.
        """
        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)
        da_dt = self.sde.da_dt(t)
        dstd_dt = self.sde.dstd_dt(t)

        # print(convert_x0_to_flow(
        #     x0_hat=self.x0_pred(x_t, t),
        #     x_t=x_t,
        #     alpha_t=a_t,
        #     std_t=std_t,
        #     da_dt=da_dt,
        #     dstd_dt=dstd_dt,
        # ).mean())
        # print(convert_score_to_flow(
        #     score_t=self.score_prior_t(x_t, t),
        #     x_t=x_t,
        #     alpha_t=a_t,
        #     std_t=std_t,
        #     da_dt=da_dt,
        #     dstd_dt=dstd_dt,
        # ).mean()
        # )
        return convert_score_to_flow(
            score_t=self.score_prior_t(x_t, t),
            x_t=x_t,
            alpha_t=a_t,
            std_t=std_t,
            da_dt=da_dt,
            dstd_dt=dstd_dt,
        )
        

    def get_posterior(self, y_obs, H_mat, sigma_y):
        """
        Returns the posterior distribution torch.distributions object.

        Args:
            - y_obs: observed data of shape (d_y, ).
            - H_mat: observation matrix of shape (d_y, D).
            - sigma_y: noise standard deviation.

        Returns:
            - posterior_distr: (torch.distributions) posterior GMM.
        """
        if sigma_y == 0:
            print("Zero noise, returning prior...")
            return self.prior_distr
        post_sigma = torch.linalg.solve(
            torch.eye(self.dim) + (1 / sigma_y**2) * H_mat.T @ H_mat,
            torch.eye(self.dim),
        )
        post_means = torch.einsum(
            "ij, kj -> ki", post_sigma, (1 / sigma_y**2) * H_mat.T @ y_obs + self.means
        )

        log_post_w_tilde = torch.log(
            self.weights
        ) + torch.distributions.MultivariateNormal(
            loc=torch.einsum("ij, kj -> ki", H_mat, self.means),
            covariance_matrix=sigma_y**2
            * torch.eye(H_mat.shape[0]).repeat(self.n_components, 1, 1)
            + (H_mat @ H_mat.T)[None],
            validate_args=False,
        ).log_prob(
            y_obs
        )
        try:
            return torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    logits=log_post_w_tilde, validate_args=False
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=post_means, covariance_matrix=post_sigma, validate_args=False
                ),
            )
        except:
            print("Positive definite error, fixing it...")
            post_sigma = (post_sigma + post_sigma.T) / 2
            return torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    logits=log_post_w_tilde, validate_args=False
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=post_means, covariance_matrix=post_sigma, validate_args=False
                ),
            )

    def get_gmm_cov(self, distr, dim):
        """
        Returns the covariance matrix of the GMM distribution.

        Args:
          distr (torch.distributions.MixtureSameFamily): the GMM distribution.
          dim (int): the dimension of the GMM.
        """
        mixture_probs = distr.mixture_distribution.probs
        list_cov_mat = distr.component_distribution.covariance_matrix
        list_means = distr.component_distribution.loc
        mu = distr.mean

        cov_mat = torch.zeros(dim, dim)
        cov_mat += torch.sum(mixture_probs[:, None, None] * list_cov_mat, axis=0)

        for i in range(distr._num_component):
            cov_mat += mixture_probs[i] * torch.outer(
                list_means[i] - mu, list_means[i] - mu
            )

        return cov_mat

    def get_distr_0t(self, t, x_t):
        """
        Returns the distribution p(x_0 | x_t) at time t as torch.distributions object.

        Args:
          t (float): the discretized time step.
          x_t (torch.Tensor): the observation at time t.

        Returns:
          (torch.distributions) distribution p(x_0 | x_t) at time t.
        """

        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)

        # post_sigma = torch.linalg.solve(
        #     torch.eye(self.dim) + (1 / std_t**2) * (a_t**2),
        #     torch.eye(self.dim),
        # )
        post_sigma = 1 / (1 + (1 / std_t**2) * a_t**2)
        post_means = ((1 / std_t**2) * a_t * x_t + self.means) * post_sigma

        # specify the posterior weights
        dim_x = x_t.shape[1] if len(x_t.shape) > 1 else x_t.shape[0]
        log_post_w_tilde = torch.log(
            self.weights
        ) + torch.distributions.MultivariateNormal(
            loc=a_t * self.means,
            covariance_matrix=(std_t**2 + a_t**2)
            * torch.eye(dim_x).repeat(self.n_components, 1, 1),
            validate_args=False,
        ).log_prob(
            x_t
        )

        post_sigma_mat = post_sigma * torch.eye(self.dim)
        try:
            return torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    logits=log_post_w_tilde, validate_args=False
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=post_means,
                    covariance_matrix=post_sigma_mat,
                    validate_args=False,
                ),
            )
        except:
            print("Positive definite error, fixing it...")
            post_sigma = (post_sigma + post_sigma.T) / 2
            return torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    logits=log_post_w_tilde, validate_args=False
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=post_means,
                    covariance_matrix=post_sigma_mat,
                    validate_args=False,
                ),
            )

    def get_cov_0t(self, t, x_t):
        """
        Returns the covariance matrix of p(x_0 | x_t) at time t.
        This is the analytical form of the covariance matrix.
        """
        distribution = self.get_distr_0t(t, x_t)

        return self.get_gmm_cov(distribution, self.dim)

    def get_mean_0t(self, t, x_t):
        """
        Returns the mean of p(x_0 | x_t) at time t.
        """
        distribution = self.get_distr_0t(t, x_t)
        # print(distribution.mixture_distribution._)
        return distribution.mean

    def get_distr_yt(self, t, x_t, H_mat, sigma_y):
        """
        Returns the distribution p(y | x_t) at time t as torch.distributions object.
        """
        # assert x_t has shape (dim, )
        if len(x_t.shape) == 1:
            x_t = x_t[None]

        distr_0t = self.get_distr_0t(t, x_t)

        new_component_cov = sigma_y**2 * torch.eye(H_mat.shape[0]).repeat(
            self.n_components, 1, 1
        ) + torch.einsum(
            "ij, kjl, lm -> kim",
            H_mat,
            distr_0t.component_distribution.covariance_matrix,
            H_mat.T,
        )

        new_component = torch.distributions.MultivariateNormal(
            loc=torch.einsum(
                "ij, kj -> ki", H_mat, distr_0t.component_distribution.loc
            ),
            covariance_matrix=new_component_cov,
            validate_args=False,
        )

        return torch.distributions.MixtureSameFamily(
            mixture_distribution=distr_0t.mixture_distribution,
            component_distribution=new_component,
            validate_args=False,
        )

    def old_grad_yt(self, t, x_t, y_obs, H_mat, sigma_y):
        """
        Returns the score of likelihood at time t, p(y | x_t)'s gradient wrt. x_t.
        """

        def log_p(x):
            distr = self.get_distr_yt(t, x, H_mat, sigma_y)
            return distr.log_prob(y_obs).sum()

        grad = torch.func.grad(log_p)(x_t)

        return grad

    def batched_mvn_log_prob_iso(self, x, mean, isotrpoic_cov):
        """
        Returns the log probability of a batch of samples x from a multivariate normal distribution.
        Assumes isotropic covariance.
        """
        # assume x has shape (n_samples, dim)
        # mean has shape (dim, )
        # cov has shape (, ), scalar
        n_samples = x.shape[0]
        dim = x.shape[1]

        # norm (x - mean) @ cov^-1 @ (x - mean)
        norm = torch.sum((x - mean[None, :]) ** 2, axis=1) / isotrpoic_cov
        log_prob = (
            -0.5 * (dim * math.log(2 * math.pi))
            - 0.5 * dim * math.log(isotrpoic_cov)
            - 0.5 * norm
        )

        return log_prob

    def batched_mvn_log_prob(self, x, mean, K):
        """
        Returns the log probability of a batch of samples x from a multivariate normal distribution.
        General case.
        """
        n_samples = x.shape[0]
        dim = x.shape[1]
        # print(x.shape, mean.shape, K.shape)
        # norm (x - mean) @ K @ (x - mean)
        diff = (x - mean[None, :]).squeeze()
        norm = torch.einsum("bi, ij, bj -> b", diff, K, diff)
        log_prob = (
            -0.5 * (dim * math.log(2 * math.pi))
            + 0.5 * torch.linalg.slogdet(K)[1]
            - 0.5 * norm
        )

        return log_prob

    def batched_mvn_log_prob_mean(self, x, mean, inv_cov):
        """
        A even more general case where mean is a matrix.

        Args:
            x(torch.Tensor): (batch_size, dim)
            mean(torch.Tensor): (batch_size, n_components, dim)
            inv_cov(torch.Tensor): (dim, dim)

        Returns:
            log_prob(torch.Tensor): (batch_size, n_components)
        """
        n_samples = x.shape[0]
        dim = x.shape[1]
        n_components = mean.shape[1]
        diff = x[:, None, :].repeat(1, n_components, 1) - mean
        norm = torch.einsum("bik, kj, bij -> bi", diff, inv_cov, diff)

        log_prob = (
            -0.5 * (dim * math.log(2 * math.pi))
            + 0.5 * torch.linalg.slogdet(inv_cov)[1]
            - 0.5 * norm
        )

        return log_prob

    def batched_mvn_log_prob_mean_iso(self, x, mean, isotropic_cov):
        """
        A even more general case where mean is a matrix.

        Args:
            x(torch.Tensor): (batch_size, dim)
            mean(torch.Tensor): (batch_size, n_components, dim)
            cov(torch.Tensor): (dim, dim)

        Returns:
            log_prob(torch.Tensor): (batch_size, n_components)
        """
        n_samples = x.shape[0]
        dim = x.shape[1]
        n_components = mean.shape[1]
        diff = x[:, None, :].repeat(1, n_components, 1) - mean
        norm = torch.sum(diff**2, axis=2) / isotropic_cov

        log_prob = (
            -0.5 * (dim * math.log(2 * math.pi))
            - 0.5 * dim * math.log(isotropic_cov)
            - 0.5 * norm
        )

        return log_prob

    def log_prob_yxt(self, t, x_t, y_obs, H_mat, sigma_y):
        """
        Efficient alternative that computes the log p(y | x_t), which returns (batch_size, ).

        Args:
            t (float): the discretized time step.
            x_t (torch.Tensor): the observation at time t, shape (batch_size, dim).
            y_obs (torch.Tensor): the observed data at time t, shape (batch_size, d_y).
            H_mat (torch.Tensor): the observation matrix, shape (d_y, dim).
            sigma_y (float): the noise standard deviation.

        Returns:
            log_prob (torch.Tensor): the log probability of p(y | x_t), shape (batch_size, ).
        """
        batch_size = x_t.shape[0]
        # component mean of p(x_0 | x_t)
        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)
        cov_0t = 1 / (1 + (1 / std_t**2) * a_t**2)

        # the mean has shape (batch_size, n_components, dim)
        mean_0t = (
            ((1 / std_t**2) * a_t * x_t)[:, None, :].repeat(1, self.n_components, 1)
            + self.means[None, ...].repeat(batch_size, 1, 1)
        ) * cov_0t

        # weights of p(x_0 | x_t), (batch_size, n_components)
        log_weights_0t = self.batched_mvn_log_prob_mean_iso(
            x_t,
            a_t * self.means[None, ...].repeat(batch_size, 1, 1),
            (std_t**2 + a_t**2),
        ) + torch.log(self.weights)[None, :].repeat(batch_size, 1)

        normalized_log_weights_0t = torch.log_softmax(log_weights_0t, dim=1)

        # component mean of p(y | x_t), (batch_size, n_components, d_y)
        mean_yt = torch.einsum("ij, bkj -> bki", H_mat, mean_0t)

        # component covariance of p(y | x_t), (batch_size, n_components, d_y, d_y)
        # this in fact shared so suffice to (d_y, d_y)
        cov_yt = sigma_y**2 * torch.eye(H_mat.shape[0]) + H_mat @ H_mat.T * cov_0t

        inv_cov_yt = torch.linalg.solve(cov_yt, torch.eye(H_mat.shape[0]))

        # log prob of p(y | x_t)
        log_components_yt = self.batched_mvn_log_prob_mean(y_obs, mean_yt, inv_cov_yt)
        log_mix_prob = torch.log_softmax(normalized_log_weights_0t, dim=1)

        return torch.logsumexp(log_components_yt + log_mix_prob, dim=1)

    def grad_yt(self, t, x_t, y_obs, H_mat, sigma_y):
        """
        Returns the gradient of log p(y | x_t) wrt. x_t.
        """

        def log_p(x):
            return self.log_prob_yxt(t, x, y_obs, H_mat, sigma_y).sum()

        grad = torch.func.grad(log_p)(x_t)

        return grad
    
    def true_vector_field(self, t, x_t, y_obs, H_mat, sigma_y):
        """
        Returns the true vector field guiding towards p(x0 | y) at time t.
        """
        grad_yt = self.grad_yt(t, x_t, y_obs, H_mat, sigma_y)
        uncond_flow = self.flow_pred(x_t, t)
        
        # coefficients
        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)
        
        guidance_coeff = (std_t**2) * (1 / a_t + 1 / std_t)
        
        return uncond_flow + guidance_coeff * grad_yt
    
    def get_cov_0t_batched(self, t, x_t):
        """
        An efficient implementation of batched covariance computation for p(x0 | x_t).
        
        Args:  
            t (float): the discretized time step.
            x_t (torch.Tensor): the observation at time t, shape (batch_size, dim).
        """
        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)
        
        batch_size = x_t.shape[0]
        
        # weights of p(x_0 | x_t), (batch_size, n_components)
        log_weights_0t = self.batched_mvn_log_prob_mean_iso(
            x_t,
            a_t * self.means[None, ...].repeat(batch_size, 1, 1),
            (std_t**2 + a_t**2),
        ) + torch.log(self.weights)[None, :].repeat(batch_size, 1)
        
        sigma_0t = 1 / (1 + (1 / std_t**2) * a_t**2) # diagonal but implemented as scalar

        normalized_log_weights_0t = torch.log_softmax(log_weights_0t, dim=1)
        
        # component means of p(x_0 | x_t), (batch_size, n_components, dim)
        mean_0t = (
            ((1 / std_t**2) * a_t * x_t)[:, None, :].repeat(1, self.n_components, 1)
            + self.means[None, ...].repeat(batch_size, 1, 1)
        ) * sigma_0t
        
        # overall mean (batch_size, dim)
        mu_0t = torch.einsum("bk, bki -> bi", torch.exp(normalized_log_weights_0t), mean_0t)
        
        # overall covariance (batch_size, dim, dim)
        # sum w_i (mu_i - mu)(mu_i - mu)^T + sum w_i cov_i
        outer_prod = torch.einsum(
            "bk, bki, bkj -> bij",
            torch.exp(normalized_log_weights_0t),
            mean_0t - mu_0t[:, None, :],
            mean_0t - mu_0t[:, None, :],
        )
        
        C_0t = torch.eye(self.dim).repeat(batch_size, 1, 1) * sigma_0t + outer_prod
        
        return C_0t
    
    
    def get_cov_st_batched(self, t, x_t, s):
        """
        Computes covariance matrix of p(x_s | x_t) for batched x_t at time t.
        """
        alpha_t = self.sde.alpha_t(t)
        alpha_s = self.sde.alpha_t(s)
        std_t = self.sde.std_t(t)
        std_s = self.sde.std_t(s)
        
        alpha_ts = alpha_t / alpha_s
        std_ts = (alpha_s * std_t - alpha_t * std_s) / alpha_s
        
        identity = torch.eye(self.dim).repeat(x_t.shape[0], 1, 1)
        
        first_term = 1 / alpha_ts * identity
        second_term = (std_ts**2 * alpha_t**2) / (alpha_ts * std_t**4) * self.get_cov_0t_batched(t, x_t)
        third_term = (-1) * std_ts**2 / (alpha_ts * std_t**2) * identity
        
        return (first_term + second_term + third_term) * (std_ts**2 / alpha_ts)
    
    
    def convert_m0t_to_mst(self, t, x_t, s, m_0t):
        """
        Converts the mean of p(x_0 | x_t) to the mean of p(x_s | x_t).
        """
        alpha_t = self.sde.alpha_t(t)
        alpha_s = self.sde.alpha_t(s)
        std_t = self.sde.std_t(t)
        std_s = self.sde.std_t(s)
        
        alpha_ts = alpha_t / alpha_s
        std_ts = (alpha_s * std_t - alpha_t * std_s) / alpha_s
        
        return (1 / alpha_ts) * (x_t + (std_ts**2 / std_t**2) * (alpha_t * m_0t - x_t))
        

    def get_mean_yt(self, t, x_t, H_mat, sigma_y):
        """
        Returns the mean of p(y | x_t) at time t.
        """
        distribution = self.get_distr_yt(t, x_t, H_mat, sigma_y)
        # print(distribution.component_distribution.covariance_matrix)
        return distribution.mean

    def get_cov_yt(self, t, x_t, H_mat, sigma_y):
        """
        Returns the covariance matrix of p(y | x_t) at time t.
        This is the analytical form of the covariance matrix.
        """
        distribution = self.get_distr_yt(t, x_t, H_mat, sigma_y)
        dim = H_mat.shape[0]

        return self.get_gmm_cov(distribution, dim)

    def get_avg_cov_yt(self, t, x_t, H_mat, sigma_y):
        """
        Given a batch of x_t at time t, compute the average covariance matrix of p(y | x_t) at time t.
        """
        avg_true_cov_yt = torch.zeros(H_mat.shape[0], H_mat.shape[0])
        for i in range(x_t.shape[0]):
            avg_true_cov_yt += self.get_cov_yt(t, x_t[i, :], H_mat, sigma_y)

        return avg_true_cov_yt / x_t.shape[0]

    def get_avg_cov_yt_list(self, x_t_list, H_mat, sigma_y):
        """
        Given a list of batches of samples x_t, compute the average covariance matrix of p(y | x_t) at time t.
        """
        # calculate the time step corresponding to each index i
        assert self.sde.sample_N == len(x_t_list)
        t_list = torch.linspace(0, 1, len(x_t_list))
        list_true_cov_yt = []

        for i in tqdm(
            range(len(x_t_list)),
            desc="Computing average covariance matrix of p(y | x_t) at time t",
            colour="green",
        ):
            try:
                temp = self.get_avg_cov_yt(t_list[i], x_t_list[i], H_mat, sigma_y)
            except:
                print("Error at index", i)
                temp = torch.zeros(H_mat.shape[0], H_mat.shape[0])
                continue
            list_true_cov_yt.append(temp)

        return list_true_cov_yt

    def flatten_list(self, list_of_tensors):
        """
        Return the entry-wise list of a list of tensors.
        e.g. if the input is [tensor1, tensor2, tensor3]
        the output will be lists of length 3, the no. of lists is equal
        to the numel of each tensor.
        """
        shape = list_of_tensors[0].shape
        if not all(tensor.shape == shape for tensor in list_of_tensors):
            raise ValueError("All tensors must have the same shape")

        lists_to_return = [[] for _ in range(list_of_tensors[0].numel())]

        for tensor in list_of_tensors:
            for i, val in enumerate(tensor.view(-1)):
                lists_to_return[i].append(val)

        return lists_to_return

    def rowsum_list(self, list_of_tensors):
        """
        Same as `flatten_list` but returns the row sums of the tensors.
        That is, if the tensor has shape (n, m), it will be turned to (n,)
        and then `flatten_list` will be applied.
        """
        temp = []

        for tensor in list_of_tensors:
            temp.append(tensor.sum(axis=1))  # sum along the rows

        return self.flatten_list(temp)

    def sample_from_prior(self, n_samples):
        return self.prior_distr.sample((n_samples,))

    def sample_from_prior_t(self, n_samples, t):
        return self.prior_distr_t(t).sample((n_samples,))

    def plot_prior(self, ax, n_samples, dims_tuple=(0, 1)):
        """
        Returns a scatter plot of the prior samples with ax.

        Args:
            - ax: matplotlib axis object.
            - n_samples: number of samples to draw from the prior.
            - dims_tuple: tuple of dimensions to plot.
        """
        samples = self.sample_from_prior(n_samples)
        ax.scatter(
            samples[:, dims_tuple[0]], samples[:, dims_tuple[1]], s=10, alpha=0.5
        )
        ax.set_title("Prior samples")

    def plot_posterior(self, ax, n_samples, y_obs, H_mat, sigma_y, dims_tuple=(0, 1)):
        """
        Returns a scatter plot of the posterior samples with ax.
        """
        samples = self.get_posterior(y_obs, H_mat, sigma_y).sample((n_samples,))
        ax.scatter(
            samples[:, dims_tuple[0]], samples[:, dims_tuple[1]], s=10, alpha=0.5
        )
        ax.set_title("Posterior samples")

    def plot_prior_t(self, ax, n_samples, t, dims_tuple=(0, 1)):
        samples = self.sample_from_prior_t(n_samples, t)
        ax.scatter(
            samples[:, dims_tuple[0]], samples[:, dims_tuple[1]], s=10, alpha=0.5
        )
        ax.set_title("Prior samples at t = {}".format(t))

    def sample_from_posterior(self, n_samples, y_obs, H_mat, sigma_y):
        samples = self.get_posterior(y_obs, H_mat, sigma_y).sample((n_samples,))
        return samples
