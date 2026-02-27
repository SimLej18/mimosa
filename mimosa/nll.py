# Third party
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jit, Array
from equinox import filter_jit

# Local imports
from mimosa.linalg import cho_factor, cho_solve

@filter_jit
def mvn_nll(inputs: Array, outputs: Array, mean: Array, cov: Array, optim=False):
	"""
	Negative log-likelihood of a multivariate normal distribution, that works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param outputs: outputs points corresponding to each input. Shape (F*N, O)
	:param mean: the mean of the multivariate normal distribution. Shape (O, F*N), with O=1 if shared_outputs_hps
	:param cov: the covariance of the multivariate normal distribution. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	:param optim: when optimizing mean-function/kernel hyperparameters, we can ignore the constant term in the log-likelihood, as it does not depend on the hyperparameters. Setting `optim=True` will ignore this constant term, which can help with performance and numerical stability when optimizing.

	:return: the negative log-likelihood of the multivariate normal distribution. Shape (O,)
	"""
	nan_mask = jnp.isnan(inputs[:, 0])

	cov_u = jnp.where(nan_mask[None, :] | nan_mask[:, None], jnp.eye(cov.shape[-1])[None, :, :], cho_factor(cov))  # Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	diff = jnp.where(nan_mask, 0., outputs.T - mean)  # Shape (O, F*N)
	y = cho_solve(jnp.broadcast_to(cov_u, (diff.shape[0], cov_u.shape[1], cov_u.shape[2])), diff)  # Shape (O, F*N)

	quad = jnp.sum(diff * y, axis=-1)
	log_det = 2 * jnp.sum(jnp.log(jnp.diagonal(cov_u, axis1=-1, axis2=-2)), axis=-1)  # Shape (O,)

	if optim:
		return 0.5 * (quad + log_det)
	constant = (inputs.shape[0] - jnp.sum(nan_mask)) * jnp.log(2 * jnp.pi)
	return 0.5 * (quad + log_det + constant)


@jit
def trace_correction(inputs: Array, post_cov: Array, cov: Array):
	"""
	Computes the trace correction term to adapt the negative log-likelihood of a MVN to the Magma algorithm. Works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param post_cov: the posterior covariance of a specific mean process. Shape (O, F*G, F*G), with O=1 if shared_outputs_hps
	:param cov: the covariance of the task or mean process. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps

	:return: the trace correction term, defined as 0.5 * trace(post_cov @ inv(cov)). Shape (O,), with O=1 if shared_outputs_hps
	"""
	nan_mask_1d = jnp.isnan(inputs[:, 0])
	nan_mask_2d = nan_mask_1d[None, :] | nan_mask_1d[:, None]

	post_cov_u = jnp.where(nan_mask_2d, jnp.eye(cov.shape[-1])[None, :, :], cho_factor(post_cov))  # Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	cov_u = jnp.where(nan_mask_2d, jnp.eye(cov.shape[-1])[None, :, :], cho_factor(cov))  # Shape (O, F*N, F*N), with O=1 if shared_outputs_hps

	if cov_u.shape[0] > post_cov.shape[0]:
		post_cov_u = jnp.broadcast_to(post_cov_u, cov_u.shape)
	elif post_cov_u.shape[0] > cov_u.shape[0]:
		cov_u = jnp.broadcast_to(cov_u, post_cov_u.shape)

	v = jsp.linalg.solve_triangular(cov_u, post_cov_u, lower=False)
	return 0.5 * (jnp.sum(v**2, axis=(-2, -1)) - jnp.sum(nan_mask_1d))  # Shape (O,)


@filter_jit
def full_nll(inputs: Array, outputs: Array, post_mean: Array, post_cov: Array, cov: Array, optim=False):
	"""
	Full negative log-likelihood of a mean process in the Magma algorithm, including the trace correction term. Works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param outputs: outputs points corresponding to each input. Shape (F*N, O)
	:param post_mean: the posterior mean of a specific mean process. Shape (O, F*G), with O=1 if shared_outputs_hps
	:param post_cov: the posterior covariance of a specific mean process. Shape (O, F*G, F*G), with O=1 if shared_outputs_hps
	:param cov: the covariance of the task or mean process. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps
	:param optim: when optimizing mean-function/kernel hyperparameters, we can ignore the constant term in the log-likelihood, as it does not depend on the hyperparameters. Setting `optim=True` will ignore this constant term, which can help with performance and numerical stability when optimizing.

	:return: the full negative log-likelihood of a mean process in the Magma algorithm, including the trace correction term. Shape (O,)
	"""
	return mvn_nll(inputs, outputs, post_mean, post_cov, optim) + trace_correction(inputs, post_cov, cov)


@filter_jit
def means_nlls(post_means: Array, post_covs: Array, grid: Array, cluster_means: Array, cluster_covs: Array, optim=False):
	"""
	Computes the negative log-likelihood of every cluster, for each output

	:param post_means: Shape (K, O, F*G)
	:param post_covs: Shape (K, O, F*G, F*G)
	:param grid: Shape (F*G, I)
	:param cluster_means: Shape (K, O, F*G) with K=1 if shared_cluster_hps and O=1 if shared_outputs_hps
	:param cluster_covs: Shape (K, O, F*G, F*G) with K=1 if shared_cluster_hps and O=1 if shared_outputs_hps
	:param optim: when optimizing mean-function/kernel hyperparameters, we can ignore the constant term in the log-likelihood, as it does not depend on the hyperparameters. Setting `optim=True` will ignore this constant term, which can help with performance and numerical stability when optimizing.

	:return: the negative log-likelihood of every cluster, for each output. Shape (K, O)
	"""
	cluster_means = jnp.broadcast_to(cluster_means, post_means.shape)
	cluster_covs = jnp.broadcast_to(cluster_covs, post_covs.shape)

	return vmap(full_nll, in_axes=(None, 0, 0, 0, 0, None))(grid, jnp.swapaxes(post_means, -1, -2), cluster_means, post_covs, cluster_covs, optim)


@filter_jit
def tasks_nlls(inputs: Array, outputs: Array, mappings: Array, post_means: Array, post_covs: Array, task_covs: Array, optim=False):
	"""
	Computes the negative log-likelihood of every task, for each cluster, for each output

	:param inputs: Shape (T, F*N, I) if not shared_inputs_in_tasks else (F*N, I)
	:param outputs: Shape (T, F*N, O)
	:param mappings: Shape (T, F*N) if not shared_inputs_in_tasks else (F*N,), with values in [0, F*G-1] and padded with F*G for missing inputs, mapping each input to a point in the grid on which the post_means and post_covs are computed
	:param post_means: Shape (K, O, F*G)
	:param post_covs: Shape (K, O, F*G, F*G)
	:param task_covs: Shape (T, K, O, F*N, F*N) with T=1 is shared_task_hps, K=1 if shared_cluster_hps and O=1 if shared_output_hps
	:param optim: when optimizing mean-function/kernel hyperparameters, we can ignore the constant term in the log-likelihood, as it does not depend on the hyperparameters. Setting `optim=True` will ignore this constant term, which can help with performance and numerical stability when optimizing.

	:return: the negative log-likelihood of every task, for each cluster, for each output. Shape (T, K, O)
	"""
	# A nice trick we can use in this function is that it can just be a vmap over `means_nlls`, providing only the right portions of post_means and post_covs to each task according to the mappings.
	outputs = jnp.broadcast_to(jnp.swapaxes(outputs[:, None, :, :], -1, -2), (outputs.shape[0], post_covs.shape[0], outputs.shape[-1], outputs.shape[-2]))
	task_covs = jnp.broadcast_to(task_covs, (outputs.shape[0],)+task_covs.shape[1:])
	if inputs.ndim == 3:
		return vmap(lambda o, i, m, tc: means_nlls(
			o,
			post_covs[:, :, m, :][:, :, :, m],
			i,
			post_means[:, :, m],
			tc,
			optim))(outputs, inputs, mappings, task_covs)
	return vmap(lambda o, tc: means_nlls(
			o,
			post_covs[:, :, mappings, :][:, :, :, mappings],
			inputs,
			post_means[:, :, mappings],
			tc,
			optim))(outputs, task_covs)
