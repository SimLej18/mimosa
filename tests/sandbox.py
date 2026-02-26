# Standard package imports
import math
from typing import Tuple

# Third party
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, jit, Array
from jax.tree_util import tree_map_with_path, GetAttrKey, tree_unflatten, tree_flatten

import matplotlib.pyplot as plt
import equinox as eqx
import numpy as np

from kernax import BlockKernel, BlockDiagKernel, BatchModule, WhiteNoiseKernel, BlockDiagKernel, FeatureKernel, VarianceKernel


def compute_priors(grid_size: int, dim: int, feat:int, input_range: tuple[int, int],
                   output_range: tuple[int, int]) -> dict[str, tuple[float, float]]:
	"""
	Compute reasonable prior bounds for GP hyperparameters.

	Args:
	    grid_size: Number of points per dimension
	    dim: Input dimensionality
	    input_range: (min, max) values for input components
	    output_range: (min, max) values for output components

	Returns:
	    Dictionary with prior bounds for 'length_scale', 'variance', 'value' (noise)
	"""
	# Computes spans
	input_span = input_range[1] - input_range[0]
	output_span = output_range[1] - output_range[0]

	# Smallest possible distance between two points in the grid, when only one dimension differs
	min_length_scale = (input_span / (grid_size - 1)) * 4  # Reasonable min distance

	# Average square distance between two points in an hypercube of dimension I
	avg_sq_dist = input_span ** 2 * dim / 6
	max_length_scale = math.sqrt(avg_sq_dist) / grid_size * 16  # Reasonable max distance

	if feat == 1:
		length_scale_prior = (min_length_scale*2, max_length_scale*2)
		length_scale_u_prior = (min_length_scale*2, max_length_scale*2)  # Should be useless
	else:  # Multi-feature means we have 3 distinct length scales that are summed, so we reduce their ranges
		length_scale_prior = (min_length_scale*4, max_length_scale*4)
		length_scale_u_prior = (min_length_scale*8, max_length_scale*8)

	# Variance prior based on output span
	variance_min = 0.25 * output_span
	variance_max = 0.75 * output_span

	if feat == 1:
		variance_prior = (variance_min, variance_max)
	else:  # Multi-feature means we have 2 distinct variances that are multiplied, so we reduce their ranges
		variance_prior = (jnp.sqrt(variance_min), jnp.sqrt(variance_max))

	# Noise prior based on output span
	noise_min = 0.05 * output_span * 10
	noise_max = 0.1 * output_span * 10
	noise_prior = (noise_min, noise_max)

	return {
		'length_scale': length_scale_prior,
		'length_scale_u': length_scale_u_prior,
		'_unconstrained_length_scale': length_scale_prior,
		'_unconstrained_length_scale_u': length_scale_u_prior,
		'variance': variance_prior,
		'_unconstrained_variance': variance_prior,
		'value': noise_prior,
		'_unconstrained_value': noise_prior,
		'noise': noise_prior,
		'_unconstrained_noise': noise_prior
	}


def initialize_kernel(key, kernel, priors):
    """
    Initialise les hyper-paramètres d'un kernel Equinox arbitraire.

    TODO: Move into kernax
    """

    # 1. Génération de l'arbre de clés (Key Tree)
    # On a besoin d'une sous-clé unique pour chaque feuille du kernel pour garantir
    # l'indépendance statistique.
    leaves, treedef = tree_flatten(kernel)
    keys = jr.split(key, len(leaves))
    key_tree = tree_unflatten(treedef, keys)

    # 2. Helper pour trouver les bornes dans la config
    def get_bounds(name):
        # Cherche d'abord à la racine (ex: input_range)
        if name in priors:
            return priors[name]
        return None

    # 3. Fonction de transformation
    def init_leaf(path, leaf, subkey):
        # Condition A : On ignore ce qui n'est pas un array JAX (int, str, None...)
        # eqx.is_array renvoie True pour jax.Array et numpy array, mais False pour int/float python
        if not eqx.is_array(leaf):
            return leaf

        # Condition B : On vérifie le nom de l'attribut
        # path[-1] est la clé de la feuille courante.
        # Si c'est un attribut de classe, c'est un GetAttrKey.
        if len(path) > 0 and isinstance(path[-1], GetAttrKey):
            param_name = path[-1].name
            bounds = get_bounds(param_name)

            if bounds is not None:
                min_val, max_val = bounds

                # On sample en préservant la shape de la leaf
                # JAX gère automatiquement le broadcasting du min/max si ce sont des scalaires
                return jr.uniform(
                    subkey,
                    shape=leaf.shape,
                    minval=min_val,
                    maxval=max_val
                )

        # Si aucune condition n'est remplie, on renvoie la feuille intacte
        return leaf

    # 4. Exécution du mapping sur le kernel ET l'arbre de clés
    # On passe key_tree comme 3ème argument, tree_map va itérer dessus en parallèle du kernel
    return tree_map_with_path(init_leaf, kernel, key_tree)