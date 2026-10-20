"""Sampler throughput (MC steps/s, all chains) vs number of chains.

    JAX_PLATFORMS=cuda python bench_chains.py --chains 1024 4096 16384 \\
        [--sweeps 20] [--reps 5] [--split 0|1] [--cache 0|1] [--label x]

Appends one JSON line per chain count to results/bench_chains.jsonl.
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', type=int, nargs='+', default=[1024])
    ap.add_argument('--sweeps', type=int, default=20)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--split', type=int, default=None)
    ap.add_argument('--cache', type=int, default=None)
    ap.add_argument('--cov', default='rotate')
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    import jax
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode=a.cov)
    smp = TutSampler(s, split=None if a.split is None else bool(a.split),
                     cache=None if a.cache is None else bool(a.cache))
    run = smp.make_run(a.sweeps)
    for nc in a.chains:
        st = smp.initial_state(nc, seed=3)
        t = time.perf_counter()
        st = run(st)
        jax.block_until_ready(st.energy)
        tc = time.perf_counter() - t
        # a few calls past the initial overlaps before timing
        for _ in range(5):
            st = run(st)
        rates = []
        for _ in range(a.reps):
            t = time.perf_counter()
            st = run(st)
            jax.block_until_ready(st.energy)
            rates.append(nc * a.sweeps * smp.n_movers
                         / (time.perf_counter() - t))
        res = dict(label=a.label, device=str(jax.devices()[0]), chains=nc,
                   split=smp.split, cache=smp.cache, sweeps=a.sweeps,
                   median=float(np.median(rates)), best=float(max(rates)),
                   first_call_s=tc)
        print(json.dumps(res), flush=True)
        os.makedirs('results', exist_ok=True)
        with open('results/bench_chains.jsonl', 'a') as fh:
            fh.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
