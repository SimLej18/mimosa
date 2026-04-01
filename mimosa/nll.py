# Third party
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, Array
from equinox import filter_jit
from mimosa.linalg import cho_factor, cho_solve


@filter_jit
def mvn_nll(inputs: Array, values: Array, mean: Array, cov: Array, jitter=jnp.array(1e-5)):
	"""
	Negative log-likelihood of a multivariate normal distribution, that works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param values: outputs points corresponding to each input. Shape (F*N, O)
	:param mean: the mean of the multivariate normal distribution. Shape (O, F*N), with O=1 if shared_outputs_hps
	:param cov: the covariance of the multivariate normal distribution. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	:param jitter: when numerical stability term passed to cho_factor

	:return: the negative log-likelihood of the multivariate normal distribution. Shape (O,)
	"""
	nan_mask = jnp.isnan(inputs[:, 0])

	cov_u = jnp.where(nan_mask[None, :] | nan_mask[:, None], jnp.eye(cov.shape[-1])[None, :, :], cho_factor(cov, jitter=jitter))  # Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	diff = jnp.where(nan_mask, 0., values.T - mean)  # Shape (O, F*N)
	y = cho_solve(jnp.broadcast_to(cov_u, (diff.shape[0], cov_u.shape[1], cov_u.shape[2])), diff)  # Shape (O, F*N)

	data_fit = jnp.sum(diff * y, axis=-1)
	penalty = 2 * jnp.sum(jnp.log(jnp.diagonal(cov_u, axis1=-1, axis2=-2)), axis=-1)  # Shape (O,)
	constant = (inputs.shape[0] - jnp.sum(nan_mask)) * jnp.log(2 * jnp.pi)

	return 0.5 * (data_fit + penalty + constant)


@jit
def trace_correction(inputs: Array, cov: Array, post_cov: Array, jitter=jnp.array(1e-5)):
	"""
	Computes the trace correction term to adapt the negative log-likelihood of a MVN to the Magma algorithm. Works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param cov: the covariance of the task or mean process. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	:param post_cov: the posterior covariance of a specific mean process. Shape (O, F*G, F*G), with O=1 if shared_outputs_hps
	:param jitter: when numerical stability term passed to cho_factor

	:return: the trace correction term, defined as 0.5 * trace(post_cov @ inv(cov)). Shape (O,), with O=1 if shared_outputs_hps
	"""
	nan_mask_1d = jnp.isnan(inputs[:, 0])
	nan_mask_2d = nan_mask_1d[None, :] | nan_mask_1d[:, None]

	post_cov_u = jnp.where(nan_mask_2d, jnp.eye(cov.shape[-1])[None, :, :], cho_factor(post_cov, jitter=jitter))  # Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	cov_u = jnp.where(nan_mask_2d, jnp.eye(cov.shape[-1])[None, :, :], cho_factor(cov, jitter=jitter))  # Shape (O, F*N, F*N), with O=1 if shared_outputs_hps

	if cov_u.shape[0] > post_cov.shape[0]:
		post_cov_u = jnp.broadcast_to(post_cov_u, cov_u.shape)
	elif post_cov_u.shape[0] > cov_u.shape[0]:
		cov_u = jnp.broadcast_to(cov_u, post_cov_u.shape)

	v = jsp.linalg.solve_triangular(cov_u.mT, post_cov_u.mT, lower=True)
	return 0.5 * (jnp.sum(v**2, axis=(-2, -1)) - jnp.sum(nan_mask_1d))  # Shape (O,)


@filter_jit
def full_nll(inputs: Array, values: Array, mean: Array, cov: Array, post_cov: Array, jitter=jnp.array(1e-5)):
	"""
	Full negative log-likelihood of a mean process in the Magma algorithm, including the trace correction term. Works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param values: outputs points corresponding to each input. Shape (F*N, O)
	:param mean: the posterior mean of a specific mean process. Shape (O, F*G), with O=1 if shared_outputs_hps
	:param cov: the covariance of the task or mean process. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	:param post_cov: the posterior covariance of a specific mean process. Shape (O, F*G, F*G), with O=1 if shared_outputs_hps
	:param jitter: when numerical stability term passed to cho_factor

	:return: the full negative log-likelihood of a mean process in the Magma algorithm, including the trace correction term. Shape (O,)
	"""
	return mvn_nll(inputs, values, mean, cov, jitter=jitter) + trace_correction(inputs, cov, post_cov, jitter=jitter)


@filter_jit
def means_nlls(post_means: Array, post_covs: Array, grid: Array, cluster_means: Array, cluster_covs: Array, jitter=jnp.array(1e-5)):
	"""
	Computes the negative log-likelihood of every cluster, for each output

	:param post_means: Shape (K, O, F*G)
	:param post_covs: Shape (K, O, F*G, F*G)
	:param grid: Shape (F*G, I)
	:param cluster_means: Shape (K, O, F*G) with K=1 if shared_cluster_hps and O=1 if shared_outputs_hps
	:param cluster_covs: Shape (K, O, F*G, F*G) with K=1 if shared_cluster_hps and O=1 if shared_outputs_hps
	:param jitter: when numerical stability term passed to cho_factor

	:return: the negative log-likelihood of every cluster, for each output. Shape (K, O)
	"""
	cluster_means = jnp.broadcast_to(cluster_means, post_means.shape)
	cluster_covs = jnp.broadcast_to(cluster_covs, post_covs.shape)

	return vmap(full_nll, in_axes=(None, 0, 0, 0, 0, None))(grid, post_means.mT, cluster_means, cluster_covs, post_covs, jitter)


@filter_jit
def tasks_nlls(inputs: Array, outputs: Array, mappings: Array, task_covs: Array, post_means: Array, post_covs: Array, jitter=jnp.array(1e-5)):
	"""
	Computes the negative log-likelihood of every task, for each cluster, for each output

	:param inputs: Shape (T, F*N, I) if not shared_inputs_in_tasks else (F*N, I)
	:param outputs: Shape (T, F*N, O)
	:param mappings: Shape (T, F*N) if not shared_inputs_in_tasks else (F*N,), with values in [0, F*G-1] and padded with F*G for missing inputs, mapping each input to a point in the grid on which the post_means and post_covs are computed
	:param task_covs: Shape (T, K, O, F*N, F*N) with T=1 is shared_task_hps, K=1 if shared_cluster_hps and O=1 if shared_output_hps
	:param post_means: Shape (K, O, F*G)
	:param post_covs: Shape (K, O, F*G, F*G)
	:param jitter: when numerical stability term passed to cho_factor

	:return: the negative log-likelihood of every task, for each cluster, for each output. Shape (T, K, O)
	"""
	# A nice trick we can use in this function is that it can just be a vmap over `full_nll`, providing only the right portions of post_means and post_covs to each task according to the mappings.
	#outputs = jnp.broadcast_to(jnp.swapaxes(outputs[:, None, :, :], -1, -2), (outputs.shape[0], post_covs.shape[0], outputs.shape[-1], outputs.shape[-2]))
	task_covs = jnp.broadcast_to(task_covs, (outputs.shape[0],)+post_covs.shape[:-2]+task_covs.shape[-2:])

	if inputs.ndim == 3:
		return vmap(
			lambda i, o, m, k_t_c: vmap(
				lambda p_m, p_c, t_c:
		            full_nll(
			            i,
			            o,
			            p_m[:, m],
			            t_c,
			            p_c[:, m, :][:, :, m],
			            jitter))(post_means, post_covs, k_t_c))(inputs, outputs, mappings, task_covs)
	return vmap(
		lambda o, k_t_c: vmap(
			lambda p_m, p_c, t_c:
				full_nll(
					inputs,
					o,
					p_m[:, mappings],
					t_c,
					p_c[:, mappings, :][:, :, mappings],
					jitter))(post_means, post_covs, k_t_c))(outputs, task_covs)
