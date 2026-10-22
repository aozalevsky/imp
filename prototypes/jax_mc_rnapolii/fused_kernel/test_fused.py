"""Check the fused sweep kernel against the XLA sweep (same state, same
random numbers), then time both.

    python test_fused.py --check --chains 4096
    python test_fused.py --bench --chains 4096 16384 32768 [--no-em-cache]
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))


def build(a):
    import scoring as T
    from build_system import build as b
    from pallas_sweep import TutSamplerFused
    S = b(seed=1)
    s = T.TutSystem(S, cov_mode='rotate')
    return TutSamplerFused(s, bc=a.bc, num_warps=a.warps,
                           num_stages=a.stages, layout=a.layout,
                           em_cache=not a.no_em_cache)


def xla_run(smp, sweeps, donate=True):
    import jax
    from sampler_soa import TutSamplerSoA
    return jax.jit(lambda x: jax.lax.fori_loop(
        0, sweeps, lambda _, y: TutSamplerSoA._sweep(smp, y), x),
        donate_argnums=0 if donate else ())


def check(smp, C):
    import jax
    import jax.numpy as jnp

    def cp(x):
        return jax.tree.map(jnp.copy, x)
    st = smp.initial_state(C, seed=3)
    st = smp.resync(xla_run(smp, 20)(st))
    print(f'equilibrated {C} chains with 20 XLA sweeps', flush=True)
    rb = [i for i, p in enumerate(smp.steps) if p['kind'] == 'rb']
    bead = [i for i, p in enumerate(smp.steps) if p['kind'] == 'bead']
    for label, kT in (('all accepted (kT 1e30)', 1e30), ('kT 1', 1.0)):
        s0 = dataclasses.replace(cp(st), kT=jnp.full(C, kT, smp.dt))
        t0 = time.time()
        a = xla_run(smp, 1, donate=False)(cp(s0))
        jax.block_until_ready(a.energy)
        t1 = time.time()
        b = smp.make_run(1, donate=False)(cp(s0))
        jax.block_until_ready(b.energy)
        t2 = time.time()
        print(f'== one sweep, {label}; XLA {t1 - t0:.1f} s, fused '
              f'{t2 - t1:.1f} s (incl. compile)')
        da = np.asarray(a.accepted - s0.accepted)
        db = np.asarray(b.accepted - s0.accepted)
        for name, ix in (('bead', bead), ('rigid body', rb)):
            print(f'  {name} decisions differing: '
                  f'{(da[ix] != db[ix]).sum()} of {da[ix].size} '
                  f'(accepted XLA {da[ix].mean():.4f}, fused '
                  f'{db[ix].mean():.4f})')
        dE = np.abs(np.asarray(a.energy, np.float64)
                    - np.asarray(b.energy, np.float64))

        def mx(u, v):
            return np.abs(np.asarray(u - v)).max()
        print(f'  energy max |XLA - fused| {dE.max():.2e} (median '
              f'{np.median(dE):.1e}); em {mx(a.em, b.em):.2e}; es_body '
              f'{mx(a.es_body, b.es_body):.2e}')
        n = smp.n_dyn
        print(f'  X max diff {mx(a.X[:, :n], b.X[:, :n]):.2e}, quat '
              f'{mx(a.quat, b.quat):.2e}, trans {mx(a.trans, b.trans):.2e}')
    s1 = smp.make_run(10, donate=False)(cp(st))
    tracked = np.asarray(s1.energy, np.float64)
    s2 = smp.resync(cp(s1))
    drift = np.abs(tracked - np.asarray(s2.energy, np.float64))
    emd = np.abs(np.asarray(s1.em, np.float64) - np.asarray(s2.em, np.float64))
    print(f'== 10 fused sweeps at kT 1: tracked vs recomputed energy max '
          f'{drift.max():.2e}, median {np.median(drift):.2e} (energies '
          f'median {np.median(tracked):.1f}); EM max {emd.max():.2e}')
    s3 = xla_run(smp, 10, donate=False)(cp(st))
    t3 = np.asarray(s3.energy, np.float64)
    s4 = smp.resync(cp(s3))
    d3 = np.abs(t3 - np.asarray(s4.energy, np.float64))
    print(f'   same with 10 XLA sweeps: max {d3.max():.2e}, median '
          f'{np.median(d3):.2e}')


def bench(smp, chains, sweeps=10, reps=5, labels=('xla', 'fused')):
    import jax
    for C in chains:
        res = {}
        for label in labels:
            st = smp.initial_state(C, seed=3)
            f = xla_run(smp, sweeps) if label == 'xla' \
                else smp.make_run(sweeps)
            t0 = time.time()
            st = f(st)
            jax.block_until_ready(st.energy)
            comp = time.time() - t0
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                st = f(st)
                jax.block_until_ready(st.energy)
                ts.append(time.perf_counter() - t0)
            t = min(ts)
            res[label] = smp.n_movers * sweeps * C / t
            print(f'{C} chains {label}: {res[label] / 1e6:.1f}M MC steps/s '
                  f'({1e3 * t / sweeps:.3f} ms per sweep; first call '
                  f'{comp:.1f} s)', flush=True)
        if len(res) == 2:
            print(f'{C} chains: fused / XLA = {res["fused"] / res["xla"]:.2f}',
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', type=int, nargs='+', default=[4096])
    ap.add_argument('--bc', type=int, default=16)
    ap.add_argument('--warps', type=int, default=2)
    ap.add_argument('--stages', type=int, default=1)
    ap.add_argument('--layout', default='reg128',
                    choices=['reg128', 'reg96', 'mem96'])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--fused-only', action='store_true')
    ap.add_argument('--no-em-cache', action='store_true')
    a = ap.parse_args()
    import jax
    print('JAX', jax.__version__, jax.devices()[0].device_kind,
          f'bc {a.bc} warps {a.warps} layout {a.layout} em_cache '
          f'{not a.no_em_cache}', flush=True)
    smp = build(a)
    if a.check:
        check(smp, a.chains[0])
    if a.bench:
        bench(smp, a.chains, labels=('fused',) if a.fused_only
              else ('xla', 'fused'))


if __name__ == '__main__':
    main()
