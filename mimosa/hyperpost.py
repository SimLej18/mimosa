from typing import Tuple

# Third party
import jax.numpy as jnp
from jax import jit, Array

from kernax import AbstractKernel, AbstractMean

# Local imports
from mimosa.linalg import scatter_to_grid_1d, scatter_to_grid_2d, cho_factor, cho_solve


# General function
@jit
def hyperpost(inputs: Array, outputs: Array, mappings: Array, grid: Array, mixture_coeffs: Array,
              mean_func: AbstractMean, mean_kernel: AbstractKernel, task_kernel: AbstractKernel) -> Tuple[Array, Array]:
	"""
	Computes the posterior mean and covariance of a Magma GP given the inputs, outputs, mappings, prior mean and kernels.

	:param inputs: Input points. Shape (F*N, I) if shared_inputs_in_tasks, else (T, F*N, I)
	:param outputs: Outputs points. Shape (T, F*N, O)
	:param mappings: Indices of each input in the grid, padded with len(grid). Shape (T, F*N)
	:param grid: points on which to compute the posterior. Shape (F*G, I)
	:param mixture_coeffs: mixture coefficients for every task. Shape(T, K)
	:param mean_func: Mean function to compute the prior mean of each process. Should accept the grid as input.
	:param mean_kernel: Kernel to be used to compute the mean covariance. Should accept the grid as input.
	:param task_kernel: Kernel to be used to compute the task covariance. Should accept the inputs as input.

	:return: a 2-tuple of the posterior mean and covariance
	"""
	big_eye = jnp.eye(grid.shape[0])
	small_eye = jnp.eye(inputs.shape[-2])

	mean_process_covs = mean_kernel(grid)  # Shape (K, O, F*G, F*G) with K=1 if shared_cluster_hps and O=1 if shared_output_hps

	# Compute mean covariance and its Cholesky factor
	mean_covs_u = cho_factor(mean_process_covs)  # Same shape
	mean_covs_inv = cho_solve(mean_covs_u, jnp.broadcast_to(big_eye, mean_covs_u.shape))  # Same shape

	# Compute task covariance, its Cholesky factor and mask NaNs with identity rows/cols
	task_covs = task_kernel(inputs)  # Shape: (T, K, O, F*N, F*N) with K=1 if shared_cluster_hps and O=1 if shared_output_hps
	eyed_task_covs = jnp.where(jnp.isnan(task_covs), small_eye, task_covs)

	# --- Posterior covariance ---
	# Small task covs
	task_covs_U = cho_factor(eyed_task_covs) # Shape: (T, K, O, F*N, F*N)
	task_covs_inv = cho_solve(task_covs_U, jnp.broadcast_to(small_eye, task_covs_U.shape))  # Same shape
	task_covs_inv -= jnp.where(jnp.isnan(task_covs), small_eye, 0)  # Correction on the diagonal
	task_covs_inv *= mixture_coeffs[:, :, None, None, None]  # Apply mixture coefficients

	# Map to full grid and sum over tasks
	task_covs_inv = scatter_to_grid_2d(jnp.full((len(grid), len(grid)), jnp.nan), task_covs_inv, mappings)  # Shape (T, K, O, F*G, F*G)
	task_covs_inv = jnp.nan_to_num(task_covs_inv).sum(axis=0)  # Shape (K, O, F*G, F*G)

	# Sum mean and task covariances and compute Cholesky factor of the posterior covariance
	post_covs_inv = cho_factor(mean_covs_inv + task_covs_inv)  # Shape (K, O, F*G, F*G)
	post_covs = cho_solve(post_covs_inv, jnp.broadcast_to(big_eye, post_covs_inv.shape))  # Shape (K, O, F*G, F*G)

	# --- Posterior mean ---
	# Compute prior means
	prior_means = mean_func(grid)  # Shape (K, O, F*G) with K=1 if shared_cluster_hps and O=1 if shared_output_hps
	prior_means = cho_solve(mean_covs_u, prior_means)  # Same shape

	# Compute weighted tasks
	outputs = outputs[:, None, :, :].swapaxes(-1, -2)  # Shape (T, 1, O, F*N)
	if task_covs_U.shape[1] != 1:  # If not shared_cluster_hps, we need to broadcast outputs
		outputs = jnp.broadcast_to(outputs, (outputs.shape[0], task_covs_U.shape[1]) + outputs.shape[2:])  # Shape (T, K, O, F*N)

	task_means = cho_solve(jnp.broadcast_to(task_covs_U, outputs.shape+(task_covs_U.shape[-1],)), outputs) # Shape (T, K, O, F*N) with K=1 if shared_cluster_hps
	task_means *= mixture_coeffs[:, :, None, None]  # Shape (T, K, O, F*N)
	task_means = scatter_to_grid_1d(jnp.full((len(grid),), 0.), task_means, mappings).sum(axis=0)  # Shape (K, O, F*G)

	full_means = jnp.broadcast_to(prior_means, task_means.shape) + task_means  # Shape (K, O, F*G)
	post_means = cho_solve(jnp.broadcast_to(post_covs_inv, full_means.shape + (post_covs_inv.shape[-1],)), full_means)

	return post_means, post_covs
