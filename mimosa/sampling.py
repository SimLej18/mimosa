import jax.random as jr
import jax.numpy as jnp
from jax import jit, Array


@jit
def sample_gp(key, mean, cov, jitter=jnp.array(1e-5)) -> Array:
	"""
	Sample outputs from a gaussian process given inputs, mean(s) and kernel(s)

	:param key: JAX PRNG key
	:param mean: the mean(s) of the generative process for each output dim. Shape (..., N)
	:param cov: the covariance matrix. Should have shape (..., N, N)
	:param jitter: jitter to add to the diagonal of the covariance matrix for numerical stability. Default is 1e-5.

	:return: generated outputs. Shape (N, O)
	"""
	return jr.multivariate_normal(key, mean, cov + jitter*jnp.eye(cov.shape[-1]))
