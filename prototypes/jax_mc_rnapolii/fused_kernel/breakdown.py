"""Where a production frame's time goes with the fused kernel (one GPU):
block of frames as in run_soa.py, without the rescore, 10 sweeps, the
kernel with fixed random numbers, without rigid-body moves, without their
EM term, and the rescore alone.

    python breakdown.py --chains 16384 [--no-em-cache]
"""
import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))


def timeit(f, *a, reps=5):
    import jax
    out = f(*a)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', type=int, default=16384)
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--no-em-cache', action='store_true')
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import scoring as T
    from build_system import build
    from pallas_sweep import TutSamplerFused
    from run_soa import make_block
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode='rotate')
    C, F = a.chains, a.frames
    n = None
    res = {}

    def fresh(prof=()):
        smp = TutSamplerFused(s, bc=16, num_warps=2,
                              em_cache=not a.no_em_cache)
        smp.prof = set(prof)
        return smp

    smp = fresh()
    n = smp.n_movers
    st0 = smp.resync(smp.initial_state(C, seed=3))

    def cp(x):
        return jax.tree.map(jnp.copy, x)
    kTs = jnp.ones(F, smp.dt)

    # production block (no host work), per frame
    blk = make_block(smp)
    t = timeit(lambda: jax.block_until_ready(blk(cp(st0), kTs)[0].energy))
    res['frame: 10 sweeps + rescore + records'] = t / F

    # without the rescore (tracked energies recorded instead)
    def frame_nr(st, kT):
        st.kT = jnp.full_like(st.kT, kT)
        st = jax.lax.fori_loop(0, 10, lambda _, x: smp._sweep(x), st)
        neg, idx = jax.lax.top_k(-st.energy, 10)
        return st, (st.energy, idx, -neg)
    blk_nr = jax.jit(lambda st, k: jax.lax.scan(frame_nr, st, k))
    t = timeit(lambda: jax.block_until_ready(blk_nr(st0, kTs)[0].energy))
    res['frame: 10 sweeps, no rescore'] = t / F

    # rescore alone
    full = jax.jit(lambda X, q: smp._full(X, q) + smp._static_ev(X))
    t = timeit(full, st0.X, st0.quat)
    res['rescore alone'] = t

    # 10 sweeps with random numbers, kernel with fixed random numbers
    sw = smp.make_run(10, donate=False)
    t = timeit(lambda: jax.block_until_ready(sw(st0).energy))
    res['10 sweeps (random numbers + kernel)'] = t
    for label, prof in (('10 kernels, fixed random numbers', ()),
                        ('10 kernels, no rigid-body moves', ('no_rb',)),
                        ('10 kernels, rigid-body moves without EM',
                         ('no_em',))):
        v = fresh(prof)
        k1, k2 = jax.random.split(jax.random.key(5))
        nrm = jax.random.normal(k1, (n, 6, C), v.dt)
        uni = jax.random.uniform(k2, (n, 3, C), v.dt)
        orig = jax.random.normal, jax.random.uniform
        jax.random.normal = lambda *a_, **k_: nrm
        jax.random.uniform = lambda *a_, **k_: uni
        r = v.make_run(10, donate=False)
        t = timeit(lambda: jax.block_until_ready(r(st0).energy))
        jax.random.normal, jax.random.uniform = orig
        res[label] = t
    print(f'{C} chains, ms (per frame = 10 sweeps; 4 bn on 4 GPUs needs '
          f'{1e3 * C * n * 10 / 1e9:.2f} ms per frame incl. output)')
    for k, v in res.items():
        print(f'  {k:45s} {1e3 * v:7.3f}')


if __name__ == '__main__':
    main()
