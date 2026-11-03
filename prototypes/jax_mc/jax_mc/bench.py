"""Speed of the engine, in MC steps/s (one step = one mover proposal).

  python -m jax_mc.bench polii --chains 1024,4096,16384 --sweeps 100
      [--float64] [--full] [--hlo] [--sampler my.module:factory]

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
          sampler=None, kT=None, burn=0, out=print):
    """sampler: a factory s -> sampler (the protocol in DESIGN.md) to time
       instead of the reference engine. kT: one value or a list cycled
       over the chains (a ladder). burn: sweeps run untimed first."""
    import jax
    from .engine import make
    eng = make(s, incremental) if sampler is None else sampler(s)
    n = eng.steps_per_sweep                 # MC steps per sweep
    run = jax.jit(eng.run, static_argnums=(1, 2), donate_argnums=0)
    out(f'{s.name}: {len(s.movers)} movers, {type(eng).__name__}, '
        f'{len(s.terms)} terms, {eng.dtype().__name__}, '
        f'{"incremental" if eng.incremental else "full rescore"}, '
        f'{jax.devices()[0].device_kind}')
    res = []
    for C in chains:
        kTs = None if kT is None else np.resize(np.atleast_1d(kT), C)
        st = eng.init(C, seed, kTs)
        if hlo and 'X' in st:
            out(f'  {C} chains: {hlo_copies(eng, st, sweeps)} copies of X '
                'in the compiled run')
        t0 = time.time()
        st = run(st, sweeps)
        jax.block_until_ready(st)
        tc = time.time() - t0
        for _ in range(burn // sweeps):
            st = run(st, sweeps)
        jax.block_until_ready(st)
        acc0 = np.asarray(st['n_acc']).sum()
        t0 = time.time()
        st = run(st, sweeps)
        jax.block_until_ready(st)
        t = time.time() - t0
        rate = C * n * sweeps / t
        acc = float(np.asarray(st['n_acc']).sum() - acc0) / (C * n * sweeps)
        out(f'  {C:6d} chains: {rate / 1e6:9.2f} M MC steps/s '
            f'({t:.2f} s per {sweeps} sweeps; first call {tc:.1f} s); '
            f'acceptance {acc:.3f}' + (
                ''.join(f', {k} {v}' for k, v in eng.stats(st).items())
                if hasattr(eng, 'stats') else ''))
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
    ap.add_argument('--sampler', default=None,
                    help='module:factory, factory(system) -> sampler')
    ap.add_argument('--kT', default=None,
                    help='kT, or a list cycled over the chains')
    ap.add_argument('--ladder', action='store_true',
                    help="the protocol's REX temperatures, cycled")
    ap.add_argument('--burn', type=int, default=0,
                    help='sweeps run before timing')
    a = ap.parse_args()
    from . import xla_defaults
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', a.float64)
    from . import sources, system
    s = system.extract(sources.load(a.system), cov_mode=a.cov)
    from .check import _factory
    kT = s.source.ladder if a.ladder else (
        [float(x) for x in a.kT.split(',')] if a.kT else None)
    bench(s, [int(x) for x in a.chains.split(',')], a.sweeps, a.seed,
          not a.full, a.hlo, _factory(a.sampler) if a.sampler else None,
          kT, a.burn)


if __name__ == '__main__':
    main()
