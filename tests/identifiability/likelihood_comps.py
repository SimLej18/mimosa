# Third party
import jax.numpy as jnp
from jax import Array
from equinox import filter_jit
from mimosa.linalg import cho_factor, cho_solve


@filter_jit
def mvn_nll_datafit(inputs: Array, values: Array, mean: Array, cov: Array, jitter=jnp.array(1e-5)):
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
	return 0.5 * data_fit


@filter_jit
def mvn_nll_penalty(inputs: Array, values: Array, mean: Array, cov: Array, jitter=jnp.array(1e-5)):
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
	penalty = 2 * jnp.sum(jnp.log(jnp.diagonal(cov_u, axis1=-1, axis2=-2)), axis=-1)  # Shape (O,)

	return 0.5 * penalty


@filter_jit
def mvn_nll_constant(inputs: Array, values: Array, mean: Array, cov: Array):
	"""
	Negative log-likelihood of a multivariate normal distribution, that works on padded data and multi-outputs.

	:param inputs: inputs points on which the mean and covariance were computed. Used for masking NaNs. Shape (F*N, I)
	:param values: outputs points corresponding to each input. Shape (F*N, O)
	:param mean: the mean of the multivariate normal distribution. Shape (O, F*N), with O=1 if shared_outputs_hps
	:param cov: the covariance of the multivariate normal distribution. Shape (O, F*N, F*N), with O=1 if shared_outputs_hps

	:return: the negative log-likelihood of the multivariate normal distribution. Shape (O,)
	"""
	nan_mask = jnp.isnan(inputs[:, 0])

	constant = (inputs.shape[0] - jnp.sum(nan_mask)) * jnp.log(2 * jnp.pi)
	return 0.5 * constant
