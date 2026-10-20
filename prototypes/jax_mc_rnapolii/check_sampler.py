"""Internal consistency of the fused sampler: the incrementally tracked
energy vs a full recompute, in float64 and float32, plus a quick speed probe.

    JAX_PLATFORMS=cpu python check_sampler.py [--x64] [--chains 8]
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x64', action='store_true')
    ap.add_argument('--chains', type=int, default=8)
    ap.add_argument('--sweeps', type=int, default=10)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--cov', default='rotate')
    ap.add_argument('--split', type=int, default=None)
    ap.add_argument('--cache', type=int, default=None)
    a = ap.parse_args()
    import jax
    if a.x64:
        jax.config.update('jax_enable_x64', True)
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode=a.cov)
    smp = TutSampler(s, dtype='float64' if a.x64 else 'float32',
                     split=None if a.split is None else bool(a.split),
                     cache=None if a.cache is None else bool(a.cache))
    st = smp.initial_state(a.chains, kT=1.0, seed=3)
    e0 = T.energy_terms(s, s.quat0, s.trans0, s.bead0, np)['total']
    print('device', jax.devices()[0], 'initial energy', float(st.energy[0]),
          'numpy float64', e0)
    run = smp.make_run(a.sweeps)
    t = time.perf_counter()
    st = run(st)
    jax.block_until_ready(st.energy)
    print('compile + first run %.1f s' % (time.perf_counter() - t))
    rates = []
    for _ in range(a.reps):
        t = time.perf_counter()
        st = run(st)
        jax.block_until_ready(st.energy)
        rates.append(a.chains * a.sweeps * smp.n_movers
                     / (time.perf_counter() - t))
    tracked = np.asarray(st.energy, np.float64)
    q, tr, b = smp.dofs(st)
    nchk = min(a.chains, 8)
    full = np.array([T.energy_terms(s, q[c], tr[c], b[c], np)['total']
                     for c in range(nchk)])
    print('tracked vs numpy float64 full recompute after %d MC steps '
          '(%d chains): max |diff| %.3e (energies %s)' % (
              (a.reps + 1) * a.sweeps * smp.n_movers, nchk,
              np.abs(tracked[:nchk] - full).max(), np.round(full[:4], 4)))
    acc = np.asarray(st.accepted).sum(0) / ((a.reps + 1) * a.sweeps
                                            * a.chains)
    rb = [i for i, p in enumerate(smp.steps) if p['kind'] == 'rb']
    print('acceptance: beads %.3f, rigid bodies %s' % (
        np.delete(acc, rb).mean(), np.round(acc[rb], 3)))
    print('%d chains: median %.0f MC steps/s (best %.0f)' % (
        a.chains, np.median(rates), max(rates)))


if __name__ == '__main__':
    main()
