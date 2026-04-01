#!/usr/bin/env python3
"""
Wrapper script for generate_data - runs a single configuration via command-line arguments.

Usage:
	python run_generate_data.py --sth true --sch true --chit false --I 1 --soh true --siit true

Fixed parameters (matching the notebook test loop):
	T=9, K=3, F=1, N=25, O=2, fh=False, siif=True
	gs=250 if I==1 else 40
"""
import os
import sys
import argparse

# Ensure the project root (parent of this file's directory) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# JAX configuration — must happen before importing jax
os.environ['JAX_ENABLE_X64'] = 'true'

import jax
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", False)

import jax.random as jr

from kernax import WhiteNoiseKernel, VarianceKernel, SEKernel, ZeroMean
from mimosa.generate_data import generate_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def str_to_bool(v: str) -> bool:
	return v.lower() in ('true', '1', 'yes')


def main():
	parser = argparse.ArgumentParser(
		description='Run generate_data with a specific configuration (single subprocess call).'
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
	T = 9
	K = 3
	F = 1
	N = 25
	O = 2
	fh = False
	siif = True
	I = args.I
	gs = 250 if I == 1 else 40

	mean = ZeroMean()
	mean_kernel = VarianceKernel(20.) * SEKernel(length_scale=1.5)
	task_kernel = VarianceKernel(2.) * SEKernel(length_scale=1.2) + WhiteNoiseKernel(noise=2.)

	config = (f"sth={args.sth}, sch={args.sch}, chit={args.chit}, fh={fh}, "
			  f"I={I}, soh={args.soh}, siit={args.siit}, siif={siif}")

	try:
		generate_data(
			k, T, K, F, N, I, O, gs,
			mean, mean_kernel, task_kernel,
			{}, {}, {},
			args.sth, args.sch, args.chit, fh, args.soh, args.siit, siif,
		)
		print(f"OK - {config}")
		sys.exit(0)
	except Exception as e:
		print(f"KO - {config}\n\t{e}", file=sys.stderr)
		sys.exit(1)


if __name__ == '__main__':
	main()
