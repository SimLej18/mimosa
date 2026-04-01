#%% md
#  ## 2: Kernel
#%%
# Jax configuration
USE_JIT = True
USE_X64 = True
DEBUG_NANS = False
VERBOSE = False
#%%
# Standard library imports
import os
os.environ['JAX_ENABLE_X64'] = str(USE_X64).lower()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#%%
# Third party
import jax
jax.config.update("jax_disable_jit", not USE_JIT)
jax.config.update("jax_debug_nans", DEBUG_NANS)
#%%
import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
#%%
from mimosa.linalg import cho_factor
#%%
from jax import numpy as jnp
from jax import Array
import equinox as eqx
from equinox import filter_jit

from kernax import AbstractKernel, StaticAbstractKernel, VarianceKernel, SEKernel

class StaticMOKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)

		# As the formula only involves diagonal matrices, we can compute directly with vectors
		sigma_diag = jnp.exp(kern.length_scale_1) + jnp.exp(kern.length_scale_2) + jnp.exp(kern.length_scale_u)  # Σ
		sigma_det = jnp.prod(sigma_diag)  # |Σ|
		diff = x1 - x2  # x - x'

		# Compute the quadratic form: (x - x')^T Sigma^{-1} (x - x')
		# Since Sigma^{-1} is diagonal, this simplifies to sum of (diff_i^2 * sigma_inv_diag_i)
		quadratic_form = jnp.sum(diff**2 / sigma_diag)

		return jnp.exp(kern.variance_1) * jnp.exp(kern.variance_2) /(((2 * jnp.pi)**(len(x1)/2)) * jnp.sqrt(sigma_det)) * jnp.exp(-0.5 * quadratic_form)


class MOKernel(AbstractKernel):
	"""
	Squared Exponential (aka "RBF" or "Gaussian") Kernel
	"""

	length_scale_1: Array = eqx.field(converter=jnp.asarray)
	length_scale_2: Array = eqx.field(converter=jnp.asarray)
	length_scale_u: Array = eqx.field(converter=jnp.asarray)
	variance_1: Array = eqx.field(converter=jnp.asarray)
	variance_2: Array = eqx.field(converter=jnp.asarray)

	static_class = StaticMOKernel

	def __init__(self, length_scale_1, length_scale_2, length_scale_u, variance_1, variance_2):
		super().__init__()
		self.length_scale_1 = length_scale_1
		self.length_scale_2 = length_scale_2
		self.length_scale_u = length_scale_u
		self.variance_1 = variance_1
		self.variance_2 = variance_2

jitter = 1e-3

#vrs = jnp.linspace(.01, 20., 10)
#lss = jnp.linspace(-5., 1.3, 7)
#lus = jnp.linspace(-10., 1., 5)
#inputs = jnp.linspace(-3, 3, 100)[:, None]

vrs = jnp.linspace(.01, 15., 10)
lss = jnp.linspace(-5., 5., 15)
lus = jnp.linspace(-10., 10., 10)
inputs = jnp.linspace(-3, 3, 100)[:, None]

for lu in lus:
	for var1 in vrs:
		for ls1 in lss:
			for ls2 in lss:
				for var2 in vrs:
					A = MOKernel(length_scale_1=ls1, length_scale_2=ls1, length_scale_u=lu, variance_1=var1, variance_2=var1)(inputs)
					B = MOKernel(length_scale_1=ls1, length_scale_2=ls2, length_scale_u=lu, variance_1=var1, variance_2=var2)(inputs)
					C = MOKernel(length_scale_1=ls2, length_scale_2=ls2, length_scale_u=lu, variance_1=var2, variance_2=var2)(inputs)

					psd_A = not jnp.isnan(cho_factor(A + jitter*jnp.eye(A.shape[0]))).any().item()
					psd_B = not jnp.isnan(cho_factor(B + jitter*jnp.eye(B.shape[0]))).any().item()
					psd_C = not jnp.isnan(cho_factor(C + jitter*jnp.eye(C.shape[0]))).any().item()

					K = jnp.block([[A, B], [B.T, C]])

					psd_K = not jnp.isnan(cho_factor(K + jitter * jnp.eye(K.shape[0]))).any().item()

					if not psd_K:
						if psd_A and psd_B and psd_C:
							logging.error(f"Non-PSD block matrix with PSD blocks for ls1={ls1:.2f}, ls2={ls2:.2f}, lu={lu:.2f}, var1={var1:.2f}, var2={var2:.2f}")
						else:
							logging.warning(f"Non-PSD block matrix with non-PSD blocks for ls1={ls1:.2f}, ls2={ls2:.2f}, lu={lu:.2f}, var1={var1:.2f}, var2={var2:.2f}")
					else:
						logging.info(f"PSD block matrix for ls1={ls1:.2f}, ls2={ls2:.2f}, lu={lu:.2f}, var1={var1:.2f}, var2={var2:.2f}")
