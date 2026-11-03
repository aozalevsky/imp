"""Speed of the engine, in MC steps/s (one step = one mover proposal).

  python -m jax_mc.bench polii --chains 1024,4096,16384 --sweeps 100
      [--float64] [--full] [--hlo]

--full times the full-rescore engine instead; --hlo counts copies of the
whole position array X in the compiled program (there should be none in
the sweep loop).
"""
import argparse
import re
import time

import numpy as np


def hlo_copies(eng, st, sweeps):
    """number of copy instructions in the compiled run whose result has
       X's shape"""
    import jax
    X = st['X']
    dt = {'float32': 'f32', 'float64': 'f64'}[str(X.dtype)]
    shape = f'{dt}[{",".join(str(n) for n in X.shape)}]'
    txt = jax.jit(eng.run, static_argnums=(1, 2)).lower(
        st, sweeps).compile().as_text()
    pat = re.compile(r'=\s*' + re.escape(shape) + r'(\{[^}]*\})?\s+copy\(')
    return sum(1 for line in txt.splitlines() if pat.search(line))


def bench(s, chains, sweeps, seed=0, incremental=True, hlo=False,
          out=print):
    import jax
    from .engine import Engine
    eng = Engine(s, incremental)
    n = len(s.movers)
    run = jax.jit(eng.run, static_argnums=(1, 2), donate_argnums=0)
    out(f'{s.name}: {n} movers in {len(eng.groups)} groups, '
        f'{len(s.terms)} terms, {eng.dtype().__name__}, '
        f'{"incremental" if incremental else "full rescore"}, '
        f'{jax.devices()[0].device_kind}')
    res = []
    for C in chains:
        st = eng.init(C, seed)
        if hlo:
            out(f'  {C} chains: {hlo_copies(eng, st, sweeps)} copies of X '
                'in the compiled run')
        t0 = time.time()
        st = run(st, sweeps)
        st['X'].block_until_ready()
        tc = time.time() - t0
        t0 = time.time()
        st = run(st, sweeps)
        st['X'].block_until_ready()
        t = time.time() - t0
        rate = C * n * sweeps / t
        acc = float(np.asarray(st['n_acc']).sum()) / (C * n * 2 * sweeps)
        out(f'  {C:6d} chains: {rate / 1e6:9.2f} M MC steps/s '
            f'({t:.2f} s per {sweeps} sweeps; first call {tc:.1f} s); '
            f'acceptance {acc:.2f}')
        res.append((C, rate))
        del st
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--chains', default='1024,4096')
    ap.add_argument('--sweeps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--float64', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--hlo', action='store_true')
    ap.add_argument('--cov', default='frozen')
    a = ap.parse_args()
    from . import xla_defaults
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', a.float64)
    from . import sources, system
    s = system.extract(sources.load(a.system), cov_mode=a.cov)
    bench(s, [int(x) for x in a.chains.split(',')], a.sweeps, a.seed,
          not a.full, a.hlo)


if __name__ == '__main__':
    main()
