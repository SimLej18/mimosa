from jax.nn import softmax
import jax.numpy as jnp
from mimosa.nll import tasks_nlls


def update_mixture(inputs, outputs, mappings, task_kern, post_mean, post_cov, mixture_proportions, jitter=jnp.array(1e-3)):
	task_llhs = tasks_nlls(inputs, outputs, mappings, task_kern(inputs), post_mean, post_cov, jitter=jitter)
	return softmax(jnp.log(mixture_proportions) - task_llhs, axis=1)
