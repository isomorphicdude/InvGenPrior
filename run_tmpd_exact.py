import torch
import matplotlib.pyplot as plt
from guided_samplers import registry, tmpd, pgdm, dps
from models import sde_lib
from models.utils import convert_flow_to_x0, convert_x0_to_flow
from physics import noisers
from sklearn.decomposition import PCA


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


import ml_collections


def get_y_obs(gmm, H_mat, noiser=None):
    x_t = gmm.sample_from_prior(1).squeeze()
    y_obs = H_mat @ x_t
    return noiser(y_obs)


# torch.manual_seed(232)
torch.manual_seed(42)
from physics.operators import get_operator, H_func_gmm
from physics.noisers import get_noise, GaussianNoise

gmm_config = ml_collections.ConfigDict()
gmm_config.dim = 8
gmm_config.obs_dim = 1

H_mat, U, S, V = get_H_mat(gmm_config.dim, gmm_config.obs_dim)

# put the matrix into config
gmm_config.H_mat = H_mat
gmm_config.U_mat = U
gmm_config.singulars = S
gmm_config.V_mat = V

gmm_config.sigma = 0.1
gmm_config.device = "cpu"
gmm_config.sampling_eps = 1e-3
gmm_config.init_type = "gaussian"
gmm_config.noise_scale = 1.0
gmm_config.sample_N = 1000  # number of sampling steps


sde = sde_lib.RectifiedFlow(
    init_type=gmm_config.init_type,
    noise_scale=gmm_config.noise_scale,
    sample_N=gmm_config.sample_N,
    sigma_var=0.0,
)

from models import gmm_model

gmm = gmm_model.GMM(dim=gmm_config.dim, sde=sde)


class gmm_flow_model(torch.nn.Module):
    def __init__(self, gmm):
        super().__init__()
        self.gmm = gmm

    def forward(self, x, t):
        if len(t.shape) >= 1:
            # print("Batch of time steps")
            # if t is a batch of time steps
            t = t[0]
        t = t / 999
        return self.gmm.flow_pred(x, t)


H_func = get_operator(name="gmm_h", config=gmm_config)
noiser = get_noise(name="gaussian", config=gmm_config)
y_obs = get_y_obs(gmm, H_func.H_mat, noiser)


num_samples = 500
start_z = torch.randn(num_samples, gmm_config.dim)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
gmm.plot_posterior(ax, 1000, y_obs, H_mat, noiser.sigma)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
plt.savefig("gmm_posterior.png")


tmpd_fixed_cov_sampler = registry.get_guided_sampler(
    name="tmpd_exact",
    model=gmm_flow_model(gmm),
    sde=gmm.sde,
    shape=(num_samples, gmm_config.dim),
    inverse_scaler=None,
    H_func=H_func,
    noiser=noiser,
    device="cpu",
    sampling_eps=1e-3,
)

from tqdm import tqdm

y_obs_batched = y_obs.unsqueeze(0).repeat(num_samples, 1)

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
    y_obs_chunk = y_obs_batched[i * chunk_size : (i + 1) * chunk_size, :]

    tmpd_samples_chunk = tmpd_fixed_cov_sampler.sample(
        y_obs_chunk, clamp_to=None, z=start_z_chunk, return_list=True
    )
    tmpd_samples_dict[i] = tmpd_samples_chunk


# make the dictionary into a list by stacking
tmpd_fixed_cov_samples = []

for i in range(len(tmpd_samples_dict[0])):
    temp = torch.cat(
        [tmpd_samples_dict[chunk_no][i] for chunk_no in range(num_chunks)], dim=0
    )
    tmpd_fixed_cov_samples.append(temp)


import pickle

with open("tmpd_fixed_cov_samples.pkl", "wb") as f:
    pickle.dump(tmpd_fixed_cov_samples, f)

for i in range(len(tmpd_fixed_cov_samples)):
    if i == len(tmpd_fixed_cov_samples) - 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        gmm.plot_posterior(ax, 1000, y_obs, H_mat, noiser.sigma)
        ax.scatter(
            tmpd_fixed_cov_samples[i][:, 0],
            tmpd_fixed_cov_samples[i][:, 1],
            s=10,
            alpha=0.5,
            color="orange",
        )
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        plt.savefig(f"gmm_posterior_tmpd_{i}.png")
