#!/usr/bin/env python3
"""
Wrapper script for prediction - runs a single configuration via command-line arguments.
Calls generate_data, hyperpost, then predict, and asserts output shapes.

Usage:
    python run_predictions.py --sth true --sch true --chit false --I 1 --soh true --siit true

Fixed parameters (matching the notebook test loop):
    T=6, K=3, F=1, N=25, O=2, fh=False, siif=True
    gs=100 if I==1 else 40
"""
import os
import sys
import argparse

# Ensure the project root (parent of this file's directory) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# JAX configuration — must happen before importing jax
os.environ['JAX_ENABLE_X64'] = 'true'

import jax
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_debug_nans", False)

import jax.numpy as jnp
import jax.random as jr

from kernax import WhiteNoiseKernel, VarianceKernel, SEKernel, AffineMean
from mimosa.generate_data import generate_data
from mimosa.hyperpost import hyperpost
from mimosa.prediction import predict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def str_to_bool(v: str) -> bool:
    return v.lower() in ('true', '1', 'yes')


def main():
    parser = argparse.ArgumentParser(
        description='Run prediction with a specific configuration (single subprocess call).'
    )
    parser.add_argument('--sth', type=str_to_bool, required=True, metavar='BOOL',
                        help='shared_task_hps')
    parser.add_argument('--sch', type=str_to_bool, required=True, metavar='BOOL',
                        help='shared_cluster_hps')
    parser.add_argument('--chit', type=str_to_bool, required=True, metavar='BOOL',
                        help='cluster_hps_in_tasks')
    parser.add_argument('--I', type=int, required=True, choices=[1, 2],
                        help='Input dimensionality')
    parser.add_argument('--soh', type=str_to_bool, required=True, metavar='BOOL',
                        help='shared_output_hps')
    parser.add_argument('--siit', type=str_to_bool, required=True, metavar='BOOL',
                        help='shared_inputs_in_tasks')
    args = parser.parse_args()

    # Fixed parameters (matching the notebook test loop)
    k = jr.PRNGKey(42)
    T = 6
    K = 3
    F = 1
    N = 25
    O = 2
    fh = False
    siif = True
    I = args.I
    gs = 100 if I == 1 else 40

    mean = AffineMean(slope=0., intercept=0.)
    mean_kernel = VarianceKernel(20.) * SEKernel(length_scale=10.)
    task_kernel = VarianceKernel(.2) * SEKernel(length_scale=9.) + WhiteNoiseKernel(noise=.01)

    mean_priors = {
        "slope": (-.2, .2),
        "intercept": (-2.5, 2.5)
    }

    mean_kernel_priors = {
        "variance": (5, 10.),
        "length_scale": (2.5, 10.)
    }

    task_kernel_priors = {
        "variance": (0.25, 1.),
        "length_scale": (2., 8.),
        "noise": (0.01, 0.1)
    }

    config = (f"sth={args.sth}, sch={args.sch}, chit={args.chit}, fh={fh}, "
              f"I={I}, soh={args.soh}, siit={args.siit}, siif={siif}")

    try:
        inputs, outputs, maps, grid, _, _, _, mix, _, m, m_k, t_k = generate_data(
            k, T, K, F, N, I, O, gs,
            mean, mean_kernel, task_kernel,
            mean_priors, mean_kernel_priors, task_kernel_priors,
            args.sth, args.sch, args.chit, fh, args.soh, args.siit, siif,
        )

        mix_coeffs = jnp.eye(K)[mix]

        p_m, p_c = hyperpost(inputs, outputs, maps, grid, mix_coeffs, m, m_k, t_k)

        t_k_noiseless = t_k.replace(noise=1e-12)
        if not args.siit:
            extended_grid = jnp.broadcast_to(grid, (len(inputs),) + grid.shape)
        else:
            extended_grid = grid

        pred_means, pred_covs = predict(
            outputs, maps,
            p_m, p_c,
            t_k(inputs),
            t_k_noiseless(extended_grid),
            t_k_noiseless(inputs, extended_grid),
        )

        assert pred_means.shape == (T, K, O, gs**I), \
            f"pred_means shape mismatch: got {pred_means.shape}, expected {(T, K, O, gs**I)}"
        assert pred_covs.shape == (T, K, O, gs**I, gs**I), \
            f"pred_covs shape mismatch: got {pred_covs.shape}, expected {(T, K, O, gs**I, gs**I)}"

        print(f"OK - {config}")
        sys.exit(0)
    except Exception as e:
        print(f"KO - {config}\n\t{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()