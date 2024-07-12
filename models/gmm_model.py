import torch
import matplotlib.pyplot as plt

from models import sde_lib
from models.utils import convert_x0_to_flow, convert_flow_to_x0


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

    def prior_distr_t(self, a_t):
        """
        Returns the prior distribution at time t, which is a mixture
        of Gaussians with degraded mean and unit variance.

        Args:
            - a_t (torch.Tensor): the degredation in N(x_t; a_t x_0, sigma_t I)
            in the forward process.

        NOTE: the a_t coeff in DDPM/IM is the square root of alpha_t while
        it is just t in rectified flows.
        """
        component_t = torch.distributions.MultivariateNormal(
            loc=a_t * self.means,
            covariance_matrix=torch.eye(self.dim).repeat(self.n_components, 1, 1),
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
        # requires grad

        a_t = self.sde.alpha_t(t)

        # print(t.shape)
        def p_t(x):
            return self.prior_distr_t(a_t).log_prob(x).sum()

        # return torch.autograd.grad(p_t(x_t), x_t)[0]
        return torch.func.grad(
            p_t
        )(x_t)

    def x0_pred(self, x_t, t):
        """
        Returns the prediction of x_0 given x_t at time t.

        Args:
            - x_t (torch.Tensor): the observation at time t.
            - t (float): the discretized time step.
        """

        a_t = self.sde.alpha_t(t)
        # std_t = self.sde.std_t(t)

        return (1 / a_t) * (x_t + (1 - a_t) * self.score_prior_t(x_t, t))

    def flow_pred(self, x_t, t):
        """
        Returns the flow vector field given x_t at time t.
        """
        a_t = self.sde.alpha_t(t)
        std_t = self.sde.std_t(t)
        da_dt = self.sde.da_dt(t)
        dstd_dt = self.sde.dstd_dt(t)

        return convert_x0_to_flow(
            x0_hat=self.x0_pred(x_t, t),
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
        post_sigma = torch.linalg.solve(
            torch.eye(self.dim) + (1 / sigma_y**2) * H_mat.T @ H_mat,
            torch.eye(self.dim),
        )
        post_means = torch.einsum(
            "ij, kj -> ki", post_sigma, (1 / sigma_y**2) * H_mat.T @ y_obs + self.means
        )

        post_w_tilde = self.weights * torch.exp(
            torch.distributions.MultivariateNormal(
                loc=torch.einsum("ij, kj -> ki", H_mat, self.means),
                covariance_matrix=sigma_y**2
                * torch.eye(H_mat.shape[0]).repeat(self.n_components, 1, 1)
                + (H_mat @ H_mat.T)[None],
                validate_args=False,
            ).log_prob(y_obs)
        )

        post_w_tilde = post_w_tilde / post_w_tilde.sum()
        try:
            return torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    probs=post_w_tilde, validate_args=False
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
                    probs=post_w_tilde, validate_args=False
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=post_means, covariance_matrix=post_sigma, validate_args=False
                ),
            )

    def sample_from_prior(self, n_samples):
        return self.prior_distr.sample((n_samples,))

    def sample_from_prior_t(self, n_samples, a_t):
        return self.prior_distr_t(a_t).sample((n_samples,))

    def plot_prior(self, n_samples, dims_tuple=(0, 1)):
        samples = self.sample_from_prior(n_samples)
        plt.scatter(
            samples[:, dims_tuple[0]], samples[:, dims_tuple[1]], s=10, alpha=0.5
        )
        plt.title("Prior samples")
        plt.show()

    def plot_posterior(self, n_samples, y_obs, H_mat, sigma_y, dims_tuple=(0, 1)):
        samples = self.get_posterior(y_obs, H_mat, sigma_y).sample((n_samples,))
        plt.scatter(
            samples[:, dims_tuple[0]], samples[:, dims_tuple[1]], s=10, alpha=0.5
        )
        plt.title("Posterior samples")
        plt.show()

    def plot_prior_t(self, n_samples, a_t, dims_tuple=(0, 1)):
        samples = self.sample_from_prior_t(n_samples, a_t)
        plt.scatter(
            samples[:, dims_tuple[0]], samples[:, dims_tuple[1]], s=10, alpha=0.5
        )
        plt.title("Prior samples at time t")
        plt.show()
