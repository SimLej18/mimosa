import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from typing import Tuple

import equinox as eqx

from kernax import BatchModule, AbstractKernel, AbstractMean, sample_hps_from_uniform_priors
from mimosa.linalg import compute_mapping
from mimosa.sampling import sample_gp


# ---------------------------------------------------------------------------
# Helper functions (mirrors sandbox_2.ipynb)
# ---------------------------------------------------------------------------

def generate_grid(n_points, n_dims, bounds):
	axis = jnp.linspace(bounds[0], bounds[1], n_points)
	grids = jnp.meshgrid(*([axis] * n_dims), indexing='ij')
	return jnp.stack(grids, axis=-1).reshape(-1, n_dims)


def sample_inputs(key, grid, T, F, N, shared_inputs_in_tasks, shared_inputs_in_features):
	if shared_inputs_in_tasks:
		if shared_inputs_in_features:
			inputs = jr.choice(key, grid, (N,), replace=False)
			mappings = compute_mapping(grid, inputs)
		else:
			inputs = vmap(lambda k: jr.choice(k, grid, (N,), replace=False))(jr.split(key, F))
			mappings = vmap(lambda i: compute_mapping(grid, i))(inputs)
	else:
		if shared_inputs_in_features:
			inputs = vmap(lambda k: jr.choice(k, grid, (N,), replace=False))(jr.split(key, T))
			mappings = vmap(lambda i: compute_mapping(grid, i))(inputs)
		else:
			inputs = vmap(lambda k1: vmap(lambda k2: jr.choice(k2, grid, (N,), replace=False))(jr.split(k1, F)))(jr.split(key, T))
			mappings = vmap(vmap(lambda i: compute_mapping(grid, i)))(inputs)
	return inputs, mappings


def build_mean(
		mean: AbstractMean,
		O: int,
		K: int,
		shared_output_hps: bool = True,
		shared_cluster_hps: bool = True,):
	"""
	Build the mean for data generation based on specified hyperparameter structure.

	:param mean:
	:param O:
	:param K:
	:param shared_output_hps:
	:param shared_cluster_hps:
	:return: the batched mean for data generation
	"""
	# multi-output HPs
	if not shared_output_hps:
		mean = BatchModule(mean, batch_size=O, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean = BatchModule(mean, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if not shared_cluster_hps:
		mean = BatchModule(mean, batch_size=K, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean = BatchModule(mean, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	return mean


def build_mean_kernel(
		mean_kernel: AbstractKernel,
		O: int,
		K: int,
		shared_output_hps: bool = True,
		shared_cluster_hps: bool = True):
	"""
	Build the mean kernel for data generation based on specified hyperparameter structure.

	:param mean_kernel:
	:param O:
	:param K:
	:param shared_output_hps:
	:param shared_cluster_hps:
	:return: the batched mean kernel for data generation

	N.b: The kernel provided to this function should be the "base" kernel, i.e. the one that would be used if all HPs were shared. This function will then wrap it in the appropriate BatchModule to create the final kernel with the desired structure of HP sharing.

	That means that if F > 1, this function expects a kernel that is already wrapped in a BlockKernel to handle the multi-feature structure, and it will add additional BatchModules on top of it to handle the multi-output, cluster, and task structures. If F=1, it expects a standard kernel and will wrap it in a BlockKernel with 1 block (effectively doing nothing) before adding the BatchModules for the other structures.

	Hence, this function doesn't manage the configs relative to features.
	"""
	# multi-output HPs
	if not shared_output_hps:
		mean_kernel = BatchModule(mean_kernel, batch_size=O, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean_kernel = BatchModule(mean_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if not shared_cluster_hps:
		mean_kernel = BatchModule(mean_kernel, batch_size=K, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean_kernel = BatchModule(mean_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	return mean_kernel


def build_task_kernel(
		task_kernel: AbstractKernel,
		T: int,
		O: int,
		K: int,
		shared_output_hps: bool = True,
		shared_task_hps: bool = True,
		cluster_hps_in_tasks: bool = False,
		shared_inputs_in_tasks: bool = True):
	"""
	Build the task kernel for data generation based on specified hyperparameter structure.

	:param task_kernel:
	:param T:
	:param O:
	:param K:
	:param shared_output_hps:
	:param shared_task_hps:
	:param cluster_hps_in_tasks:
	:param shared_inputs_in_tasks:
	:return: the batched task kernel for data generation

	N.b: The kernel provided to this function should be the "base" kernel, i.e. the one that would be used if all HPs were shared. This function will then wrap it in the appropriate BatchModule to create the final kernel with the desired structure of HP sharing.

	That means that if F > 1, this function expects a kernel that is already wrapped in a BlockKernel to handle the multi-feature structure, and it will add additional BatchModules on top of it to handle the multi-output, cluster, and task structures. If F=1, it expects a standard kernel and will wrap it in a BlockKernel with 1 block (effectively doing nothing) before adding the BatchModules for the other structures.

	Hence, this function doesn't manage the configs relative to features.
	"""
	# multi-output HPs
	if not shared_output_hps:
		task_kernel = BatchModule(task_kernel, batch_size=O, batch_in_axes=0, batch_over_inputs=False)
	else:
		task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if cluster_hps_in_tasks:
		task_kernel = BatchModule(task_kernel, batch_size=K, batch_in_axes=0, batch_over_inputs=False)
	else:
		task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# task HPs
	if shared_task_hps:
		if shared_inputs_in_tasks:
			task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)
		else:
			task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=True)
	else:
		if shared_inputs_in_tasks:
			task_kernel = BatchModule(task_kernel, batch_size=T, batch_in_axes=0, batch_over_inputs=False)
		else:
			task_kernel = BatchModule(task_kernel, batch_size=T, batch_in_axes=0, batch_over_inputs=True)

	return task_kernel


def generate_data(
		key,
		T:int, K:int, F:int, N:int, I:int, O:int, grid_size: int,
		mean: eqx.Module, mean_kernel: AbstractKernel, task_kernel: AbstractKernel,
		mean_priors: dict, mean_kernel_priors: dict, task_kernel_priors: dict,
		shared_task_hps: bool = True,
		shared_cluster_hps: bool = True,
		cluster_hps_in_tasks: bool = False,
		feature_hps:bool = False,
		shared_output_hps:bool = True,
		shared_inputs_in_tasks: bool = True,
		shared_inputs_in_features: bool = True,
		input_range:Tuple[int, int] = (-50, 50),
):
	"""
	--- PRNG parameters ---
	:param key: `jax.random` PRNG key

	--- Dataset size parameters ---
	:param T: Number of tasks
	:param K: Number of clusters
	:param F: Number of features
	:param N: Number of points for each feature of each task
	:param I: Dimension of input points
	:param O: Dimension of output points

	--- Hyper-parameters modules ---
	:param mean: The mean used as a prior to sample cluster mean processes
	:param mean_kernel: The kernel used as a prior to sample cluster mean processes
	:param task_kernel: The kernel used as a prior to sample task processes
	:param mean_priors: Dictionary of priors for `mean`, aka min and max bounds for each parameter of `mean`
	:param mean_kernel_priors: Dictionary of priors for `mean_kernel`, aka min and max bounds for each parameter of `mean_kernel`
	:param task_kernel_priors: Dictionary of priors for `task_kernel`, aka min and max bounds for each parameter of `task_kernel`

	--- Hyper-parameters options ---
	:param shared_task_hps: Whether tasks share hyper-parameters
	:param shared_cluster_hps: Whether clusters share hyper-parameters
	:param cluster_hps_in_tasks: Whether task have cluster-specific hyper-parameters
	:param feature_hps: Whether features share hyper-parameters
	:param shared_output_hps: Whether outputs dimensions share hyper-parameters

	--- Sampling options ---
	:param grid_size: Grid size along each dimension. Ex: if I=2 and grid_size=10, inputs will be sampled from a regular 2D grid containing 10**2 = 100 points, from [input_range[0], input_range[0]] to [input_range[1], input_range[1]]
	:param shared_inputs_in_tasks: Whether all tasks are observed on the same sampled grid
	:param shared_inputs_in_features: Whether all features are observed on the same sampled grid

	--- Configuration parameters ---
	:param input_range: Min and max value for input points (applied to every input dimensions)

	:return: a tuple containing:
		* inputs (Shape (T, F, N**I, I)),
		* outputs (Shape (T, F, N**I, O)),
		* mixture (Shape T, K)
		* mean used to generate mean processes (eqx.Module)
		* mean kernel used to generate mean processes (AbstractKernel)
		* task kernel used to generate the data (AbstractKernel)

	TODO: Adapt for multi-feature
	"""
	# Step 0: parameters checks and conversions
	assert grid_size ** I >= N, f"Cannot sample N={N} points in a grid containing {grid_size ** I} points (= {grid_size}^{I})"
	assert O > 1 or shared_output_hps, f"No need to differentiate hyper-parameters for outputs when having only {O} output dimension"
	assert K > 1 or not cluster_hps_in_tasks, f"No need for cluster-specific hyper-parameters when having only {K} cluster"
	assert F > 1 or not feature_hps, f"No need for feature-specific hyper-parameters when having only {F} feature"

	# Step 1: generate the grid
	grid = generate_grid(grid_size, I, input_range)  # Shape (G, I) where G = grid_size**I

	# Step 2: sample the input grid
	inputs, mappings = sample_inputs(key, grid, T, F, N, shared_inputs_in_tasks, shared_inputs_in_features)  # Varying shapes

	# Step 3: batch kernels
	mean = build_mean(mean, O, K, shared_output_hps, shared_cluster_hps)
	mean_kernel = build_mean_kernel(mean_kernel, O, K, shared_output_hps, shared_cluster_hps)
	task_kernel = build_task_kernel(task_kernel, T, O, K, shared_output_hps, shared_task_hps, cluster_hps_in_tasks, shared_inputs_in_tasks)

	# Step 4: sample HPs from priors
	key, subkey1, subkey2, subkey3 = jr.split(key, 4)
	mean = sample_hps_from_uniform_priors(subkey1, mean, mean_priors)
	mean_kernel = sample_hps_from_uniform_priors(subkey2, mean_kernel, mean_kernel_priors)
	task_kernel = sample_hps_from_uniform_priors(subkey3, task_kernel, task_kernel_priors)

	# Step 5: sample mean processes for each cluster from the mean and mean kernel, evaluated on the grid
	# Adapt grid if we are in multi-feature and features don't share inputs, to create a separate grid for each feature
	if not shared_inputs_in_features:
		grid = jnp.tile(grid, (F,) + (1,) * grid.ndim)  # Shape (F*G, I)

	mean_process_means = mean(grid)  # Shape (K, O, F*G) with K=1 if shared_cluster_hps and O=1 if shared_output_hps
	mean_process_covs = mean_kernel(grid)  # Shape (K, O, F*G, F*G) with K=1 if shared_cluster_hps and O=1 if shared_output_hps

	if shared_output_hps:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m[0], c[0]), in_axes=(0, None, None))
		if shared_cluster_hps:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m[0], c[0]), in_axes=(0, None, None))
		else:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	else:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m, c), in_axes=(0, 0, 0))
		if shared_cluster_hps:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m[0], c[0]), in_axes=(0, None, None))
		else:
			sample_clusters = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	key, subkey = jr.split(key)
	subkeys = jr.split(subkey, (K, O))

	mean_processes = sample_clusters(subkeys, mean_process_means, mean_process_covs)  # Shape (K, O, F*G)

	# Step 6: assign tasks to clusters
	mixture = jnp.argmax(jnp.eye(K)[jnp.array(jnp.floor(jnp.arange(T) / T * K), dtype=int)], axis=1)  # Shape (T,)

	# Step 6: sample task processes for each task from the task kernel, evaluated on the task inputs
	task_means_on_grid = mean_processes[mixture, ...]  # Shape (T, O, F*G)
	task_means = vmap(lambda t_m, m: t_m[:, m], in_axes=(0, None if shared_inputs_in_tasks else 0))(task_means_on_grid, mappings)  # Shape (T, O, F*N)

	task_covs = task_kernel(inputs)  # Shape (T, K, O, F*N, F*N), with T=1 if shared_task_hps, K=1 if not cluster_hps_in_tasks and O=1 if shared_output_hps
	if cluster_hps_in_tasks:
		# Select covariance from the "right" cluster for each task
		task_covs = task_covs[jnp.arange(len(task_covs)), mixture]  # Shape (T, O, F*N, F*N) with T=1 if shared_task_hps and O=1 if shared_output_hps
	else:
		task_covs = task_covs[:, 0, ...]  # Shape (T, O, F*N, F*N) with T=1 if shared_task_hps and O=1 if shared_output_hps

	if shared_output_hps:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m, c[0]), in_axes=(0, 0, None))
		if shared_task_hps:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c[0]), in_axes=(0, 0, None))
		else:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	else:
		sample_outputs = vmap(lambda k, m, c: sample_gp(k, m, c), in_axes=(0, 0, 0))
		if shared_task_hps:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c[0]), in_axes=(0, 0, None))
		else:
			sample_tasks = vmap(lambda k, m, c: sample_outputs(k, m, c), in_axes=(0, 0, 0))
	key, subkey = jr.split(key)
	subkeys = jr.split(subkey, (T, O))

	outputs = sample_tasks(subkeys, task_means, task_covs).transpose(0, 2, 1)  # Shape (T, F*N, O)

	return inputs, outputs, mappings, grid, mean_process_means, mean_process_covs, mean_processes, mixture, task_means, mean, mean_kernel, task_kernel