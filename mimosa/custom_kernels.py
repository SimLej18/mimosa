from jax import numpy as jnp
from jax import Array
import equinox as eqx
from equinox import filter_jit

from kernax import StaticAbstractKernel, AbstractKernel
from kernax.transforms import to_constrained, to_unconstrained


class StaticRBFKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)
		return kern.variance * jnp.exp(-0.5 * ((x1 - x2) @ (x1 - x2)) / kern.length_scale ** 2)

class RBFKernel(AbstractKernel):
	"""
	RBF (Radial Basis Function) Kernel with constrained positive parameters.

	Both length_scale and variance are constrained to be positive.
	"""
	static_class = StaticRBFKernel

	_unconstrained_length_scale: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_variance: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, length_scale, variance, **kwargs):
		"""
		Initialize the RBF kernel.

		:param length_scale: length scale parameter (ℓ) - must be positive
		:param variance: variance parameter (σ²) - must be positive
		"""
		# Validate parameters are positive
		length_scale = jnp.array(length_scale)
		variance = jnp.array(variance)
		length_scale = eqx.error_if(length_scale, jnp.any(length_scale <= 0), "length_scale must be positive.")
		variance = eqx.error_if(variance, jnp.any(variance <= 0), "variance must be positive.")

		# Initialize parent
		super().__init__(**kwargs)

		# Store in unconstrained space
		self._unconstrained_length_scale = to_unconstrained(jnp.asarray(length_scale))
		self._unconstrained_variance = to_unconstrained(jnp.asarray(variance))

	@property
	def length_scale(self) -> Array:
		"""Get length_scale in constrained (positive) space."""
		return to_constrained(self._unconstrained_length_scale)

	@property
	def variance(self) -> Array:
		"""Get variance in constrained (positive) space."""
		return to_constrained(self._unconstrained_variance)