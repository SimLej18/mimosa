import jax.numpy as jnp
from mimosa.nll import means_nlls, tasks_nlls
import optimistix as optx
from equinox import filter_jit, combine


@filter_jit
def optimise_clusters(
		clust_mean, clust_kern,
		post_mean, post_cov, grid,
		solver=optx.LBFGS(atol=1e-5, rtol=1e-5), jitter=jnp.array(1e-3),
		clust_mean_frozen=None, clust_kern_frozen=None):

	@filter_jit
	def loss_fn(params, frozen):
		mean = params[0] if frozen[0] is None else combine(params[0], frozen[0])
		kern = params[1] if frozen[1] is None else combine(params[1], frozen[1])

		return means_nlls(post_mean, post_cov, grid, mean(grid), kern(grid), jitter=jitter).sum()


	params = (clust_mean, clust_kern)
	frozen = (clust_mean_frozen, clust_kern_frozen)

	return optx.minimise(loss_fn, solver, params, frozen).value

@filter_jit
def optimise_tasks(
		task_kern,
		inputs, outputs, mappings, post_mean, post_cov, mixture_coeffs,
		solver=optx.LBFGS(atol=1e-5, rtol=1e-5), jitter=jnp.array(1e-3),
		task_kern_frozen=None):

	@filter_jit
	def loss_fn(params, frozen):
		kern = params if frozen is None else combine(params, frozen)

		return (tasks_nlls(inputs, outputs, mappings, kern(inputs), post_mean, post_cov, jitter=jitter) * mixture_coeffs).sum()

	return optx.minimise(loss_fn, solver, task_kern, task_kern_frozen).value
