from tqdm import tqdm
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
            covariance_matrix=torch.eye(self.dim).repeat(self.n_components, 1, 1) * (a_t**2 + std_t**2),
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
            cov_mat += mixture_probs[i] * torch.outer(list_means[i] - mu, list_means[i] - mu)
        
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
    
        post_sigma = torch.linalg.solve(
            torch.eye(self.dim) + (1 / std_t**2) * (a_t**2),
            torch.eye(self.dim),
        )
        post_means = torch.einsum(
            "ij, kj -> ki", post_sigma, (1 / std_t**2) *  a_t * x_t + self.means
        )

        
        post_w_tilde = self.weights * torch.exp(
            torch.distributions.MultivariateNormal(
                loc=a_t * self.means,
                covariance_matrix=std_t**2
                * torch.eye(x_t.shape[1]).repeat(self.n_components, 1, 1)
                + a_t**2,
                validate_args=False,
            ).log_prob(x_t)
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
            
    def get_cov_0t(self, t, x_t):
        """
        Returns the covariance matrix of p(x_0 | x_t) at time t.  
        This is the analytical form of the covariance matrix.
        """
        distribution = self.get_distr_0t(t, x_t)
        
        return self.get_gmm_cov(distribution, self.dim)
        
        
    
    def get_mean_0t(self, t, x_t, x_0):
        """
        Returns the mean of p(x_0 | x_t) at time t.
        """
        distribution = self.get_distr_0t(t, x_t)
        
        return distribution.mean
    
    
    def get_distr_yt(self, t, x_t, H_mat, sigma_y):
        """
        Returns the distribution p(y | x_t) at time t as torch.distributions object.  
        """
        # assert x_t has shape (dim, )
        if len(x_t.shape) == 1:
            x_t = x_t[None]
        
        distr_0t = self.get_distr_0t(t, x_t)
        
        new_component_cov = sigma_y**2 + torch.einsum("ij, kjl, lm -> kim", H_mat, distr_0t.component_distribution.covariance_matrix, H_mat.T)
        
        new_component = torch.distributions.MultivariateNormal(
            loc = torch.einsum("ij, kj -> ki", H_mat, distr_0t.component_distribution.loc),
            covariance_matrix=new_component_cov,
            validate_args=False
        )
        
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=distr_0t.mixture_distribution,
            component_distribution=new_component,
            validate_args=False
        )
        
    def get_mean_yt(self, t, x_t, H_mat, sigma_y):
        """
        Returns the mean of p(y | x_t) at time t.
        """
        distribution = self.get_distr_yt(t, x_t, H_mat, sigma_y)
        
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
        
        for i in tqdm(range(len(x_t_list)),
                      desc="Computing average covariance matrix of p(y | x_t) at time t",
                      colour="green"):
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
            temp.append(tensor.sum(axis=1)) # sum along the rows
            
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