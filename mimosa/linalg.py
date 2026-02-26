"""
This module contains linear algebra functions that are used in the mimosa library.
These functions are implemented using JAX and are designed to be efficient and compatible with JAX's JIT compilation
and automatic differentiation features.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax.lax import cond, while_loop


@jit
def scatter_to_grid_1d(full, tensor, maps):
	T = tensor.shape[0]
	N = maps.shape[-1]
	G = full.shape[0]
	if maps.ndim == 1:
		maps = jnp.broadcast_to(maps, (T, N))
	middle = tensor.shape[1:-1]
	flat_tensor = tensor.reshape(T, -1, N)  # (T, M, N)

	@vmap
	def scatter(t, m):  # t: (M, N), m: (N,)
		return jnp.broadcast_to(full, (t.shape[0], G)).copy().at[:, m].set(t)

	return scatter(flat_tensor, maps).reshape(T, *middle, G)


@jit
def scatter_to_grid_2d(full, tensor, maps):
	"""
	Scatters a tensor to a full tensor using mappings.

	:param full: the matrix to scatter into, of shape (G, G)
	:param tensor: the tensor to scatter, of shape (T, ..., N, N)
	:param maps: the mappings to scatter with, of shape (T, N).
	Each row contains the indices of the full matrix where the corresponding row of the tensor should be scattered.
	:return: a tensor of shape (T, ..., G, G) where the values from the input tensor have been scattered to the full matrix according to the mappings.
	"""
	T = tensor.shape[0]
	N = maps.shape[-1]
	G = full.shape[0]
	if maps.ndim == 1:
		maps = jnp.broadcast_to(maps, (T, N))
	middle = tensor.shape[1:-2]
	flat_tensor = tensor.reshape(T, -1, N, N)

	@vmap
	def scatter(t, m):
		i, j = m[:, None], m[None, :]
		return jnp.broadcast_to(full, (t.shape[0], G, G)).copy().at[:, i, j].set(t)

	return scatter(flat_tensor, maps).reshape(T, *middle, G, G)


@jit
def cho_factor(cov, jitter=1e-5):
	"""
	Wrapper around jax.scipy.linalg.cho_factor that adds jitter to the diagonal for numerical stability and only returns the array.

	:param cov: covariance matrix to factor. Shape (..., N, N)
	:param jitter: amount of jitter to add to the diagonal. Default is 1e-5.

	:return: Cholesky upper factor of the covariance matrix. Shape (..., N, N)
	"""
	return jsp.linalg.cho_factor(cov + jitter * jnp.eye(cov.shape[-1]))[0]


@jit
def cho_solve(cov_u, res):
	"""
	Solves cov_u @ x = res for x, where cov_u is the Cholesky upper factor of a covariance matrix.
	This is a wrapper around jax.scipy.linalg.cho_solve that always uses upper factorisation.

	:param cov_u: covariance matrix to factor. Shape (..., N, N)
	:param res: result to solve for. Shape (..., N, M)
	:return: x. Shape (..., N, M)
	"""
	return jsp.linalg.cho_solve((cov_u, False), res)


@jit
def searchsorted_2D(vector: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
	"""
	Search along axis 1 for a vector in a matrix. If found, return the index of the vector.
	If not found, return len(matrix).

	For this function to work, the vectors in the matrix must be sorted lexicographically.
	ex:
	[[1, 1, 0],
	 [1, 2, 1],
	 [1, 2, 2],
	 [2, 1, 3],
	 [2, 2, 1]]

	:param vector: the vector to search for
	:param matrix: the matrix to search in
	:return: the index of the vector in the matrix, or len(matrix) if not found
	"""

	@jit
	def compare_vectors(v1, v2):
		"""Compare two vectors lexicographically. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""
		diff = v1 - v2
		# Find first non-zero element
		nonzero_mask = diff != 0
		# If all elements are zero, vectors are equal
		first_nonzero_idx = jnp.argmax(nonzero_mask)

		return cond(
			jnp.any(nonzero_mask),
			lambda: jnp.array(jnp.sign(diff[first_nonzero_idx])).astype(int),
			lambda: jnp.array(0).astype(int)
		)

	@jit
	def search_condition(state):
		start, end, found = state
		return (start < end) & (~found)

	@jit
	def search_step(state):
		start, end, found = state
		mid = (start + end) // 2

		comparison = compare_vectors(vector, matrix[mid])

		# If vectors are equal, we found it
		new_found = comparison == 0
		new_start = cond(comparison < 0, lambda: start, lambda: mid + 1)
		new_end = cond(comparison < 0, lambda: mid, lambda: end)

		# If found, return the index in start position
		final_start = cond(new_found, lambda: mid, lambda: new_start)

		return final_start, new_end, new_found

	# Initial state: (start, end, found)
	initial_state = (0, len(matrix), False)
	final_start, final_end, found = while_loop(search_condition, search_step, initial_state)

	# Return the found index or len(matrix) if not found
	return cond(found, lambda: final_start, lambda: len(matrix))


searchsorted_2D_vectorized = jit(vmap(searchsorted_2D, in_axes=(0, None)))


@jit
def lexicographic_sort(arr: jnp.ndarray) -> jnp.ndarray:
	"""
	sorts a 2D array lexicographically
	:param arr: 2D array to be sorted
	:return: sorted version along the first dimension
	"""
	return arr[jnp.lexsort(arr.T[::-1])]


@jit
def compute_mapping(grid: jnp.array, element: jnp.array) -> jnp.array:
	"""
	Returns the indices of each vector/element in the grid. The grid must be sorted lexicographically.

	:param grid: a sorted grid of shape (N,) or (N, I). If it is 2D, it must be sorted lexicographically.
	:param elements: element to search for in the grid, either scalar or of shape (I,)
	:return: indices in grid where the element can be found, as a scalar.
	"""
	if grid.shape[-1] == 1:
		# We only have 1 input dimension, and we can use the fast jnp.searchsorted function
		return jnp.searchsorted(grid.squeeze(axis=-1), element.squeeze(axis=-1))
	# Multiple input dimensions requires our custom lexicographic search
	return searchsorted_2D_vectorized(element, grid)
