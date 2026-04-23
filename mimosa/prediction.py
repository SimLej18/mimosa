# Third party
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, Array

# Local imports
from mimosa.linalg import cho_factor


def predict_task_output(output_obs, post_mean_obs, post_mean_grid, gamma_obs, gamma_crossed, gamma_grid):
	""" Predicts a single task output, within a single cluster"""
	padding_mask_1D = ~jnp.isnan(output_obs)[:, None]
	padding_mask_2D = padding_mask_1D & padding_mask_1D.T

	gamma_obs = jnp.where(padding_mask_2D, gamma_obs, jnp.eye(len(gamma_obs)))
	gamma_crossed = jnp.where(padding_mask_1D, gamma_crossed, 0.)

	gamma_pred_U = cho_factor(gamma_obs)
	z = lax.linalg.triangular_solve(gamma_pred_U, gamma_crossed.T).T
	y = lax.linalg.triangular_solve(gamma_pred_U, jnp.nan_to_num(output_obs) - post_mean_obs)

	pred_mean = post_mean_grid + (z.T @ y)
	pred_cov = gamma_grid - (z.T @ z)

	return pred_mean, pred_cov


def predict_task_in_cluster(output_obs, post_mean_obs, post_mean_grid, gamma_obs, gamma_crossed, gamma_grid):
	if gamma_obs.shape[0] == 1:
		return (vmap(
			predict_task_output,
			in_axes=(0, 0, 0, None, None, None))
		        (output_obs.T, post_mean_obs, post_mean_grid, gamma_obs[0], gamma_crossed[0], gamma_grid[0]))
	else:
		return (vmap(
			predict_task_output,
			in_axes=(0, 0, 0, 0, 0, 0))
		        (output_obs.T, post_mean_obs, post_mean_grid, gamma_obs, gamma_crossed, gamma_grid))


def predict_clusters(task_outputs, task_mappings,
                     post_mean_grid,
                     post_cov_grid,
                     task_cov_obs, task_cov_grid, task_cov_crossed):
	post_mean_obs = post_mean_grid[:, :, task_mappings]
	post_cov_obs = post_cov_grid[:, :, task_mappings, :][:, :, :, task_mappings]
	post_cov_crossed = post_cov_grid[:, :, task_mappings, :]

	gamma_obs = post_cov_obs + task_cov_obs
	gamma_grid = post_cov_grid + task_cov_grid
	gamma_crossed = post_cov_crossed + task_cov_crossed

	if gamma_obs.shape[0] == 1:
		return vmap(
			predict_task_in_cluster,
			in_axes=(None, 0, 0, None, None, None)
		)(task_outputs, post_mean_obs, post_mean_grid, gamma_obs[0], gamma_crossed[0], gamma_grid[0])
	return vmap(
		predict_task_in_cluster,
		in_axes=(None, 0, 0, 0, 0, 0)
	)(task_outputs, post_mean_obs, post_mean_grid, gamma_obs, gamma_crossed, gamma_grid)


def predict(outputs, mappings,
            post_mean_grid,
            post_cov_grid,
            tasks_cov_obs, tasks_cov_grid, tasks_cov_crossed):
	""" Predict every task for every cluster. """
	if tasks_cov_obs.shape[0] == 1:
		return vmap(
			predict_clusters,
			in_axes=(0, 0 if mappings.ndim == 2 else None,
			         None,
			         None,
			         None, None, None),
		)(outputs, mappings,
		  post_mean_grid,
		  post_cov_grid,
		  tasks_cov_obs[0], tasks_cov_grid[0], tasks_cov_crossed[0])

	return vmap(
		predict_clusters,
		in_axes=(0, 0 if mappings.ndim == 2 else None,
		         None,
		         None,
		         0, 0, 0),
	)(outputs, mappings,
	  post_mean_grid,
	  post_cov_grid,
	  tasks_cov_obs, tasks_cov_grid, tasks_cov_crossed)
