"""The configuration file for the Gaussian Mixture Model (GMM) experiments."""

import torch
import ml_collections


def get_H_mat(dim, obs_dim):
    """
    Returns the H_mat, its V, U, and singulars (same shape as obs_dim).

    Args:
        - dim: int, dimension of the true data
        - obs_dim: int, dimension of the observation space

    Returns:
        - H_mat: torch.Tensor, of shape (obs_dim, dim), the observation matrix
        - U: torch.Tensor, of shape (dim, dim), the left singular vectors of H_mat
        - S: torch.Tensor, of shape (dim,), the singular values of H_mat
        - Vt: torch.Tensor, of shape (dim, obs_dim), the right singular vectors of H_mat
    """
    H_mat = torch.randn(obs_dim * dim).reshape(obs_dim, dim)
    U, S, Vt = torch.linalg.svd(H_mat, full_matrices=True)
    coordinate_mask = torch.ones_like(Vt[0])
    coordinate_mask[len(S) :] = 0

    # sampling Unif[0, 1] for the singular values
    diag = torch.sort(torch.rand_like(S), descending=True).values

    H_mat = U @ (torch.diag(diag)) @ Vt[coordinate_mask == 1, :]

    return H_mat, U, diag, Vt[coordinate_mask == 1, :].T


def get_config():
    gmm_config = ml_collections.ConfigDict()
    gmm_config.dim = 80
    gmm_config.obs_dim = 1

    H_mat, U, S, V = get_H_mat(gmm_config.dim, gmm_config.obs_dim)

    # put the matrix into config
    gmm_config.H_mat = H_mat
    gmm_config.U_mat = U
    gmm_config.singulars = S
    gmm_config.V_mat = V

    gmm_config.sigma = 0.05
    gmm_config.device = 'cpu'
    gmm_config.sampling_eps = 1e-3
    gmm_config.init_type = 'gaussian'
    gmm_config.noise_scale = 1.0
    gmm_config.sample_N = 1000 # number of sampling steps
    
    return gmm_config