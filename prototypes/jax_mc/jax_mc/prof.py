"""Where the engine's time goes: per kind of mover, and per term.

  python -m jax_mc.prof npc --chains 16384 --sweeps 20 [--float64]
      [--kinds ball,rb]

For each kind of mover (ball, rb, transform, normal) it times sweeps over
the movers of that kind only, then the same with one term's energy change
replaced by zero (XLA drops the work nothing depends on), and with every
term's. A term's share is the time its removal saves; what is left with
no terms at all is the proposals, the random numbers, the loop and the
write-back.

Read the shares with care:
- they needn't add up. On a GPU, kernels that stream different large
  arrays push each other out of the cache, so removing either of two terms
  can save more than its own work (NPC basket, 16,384 chains on an L40:
  excluded volume and EM each "cost" 160 us of a 258 us bead step);
- removing a term changes which moves are accepted, not what a step
  costs: the engine's work per step doesn't depend on the state.

Only the branches of the movers being timed are compiled.
"""
import argparse
import time

import numpy as np


class _Zero:
    """a term whose moves cost nothing and change nothing"""

    def __init__(self, t):
        self._t = t

    def __getattr__(self, k):
        return getattr(self._t, k)

    def delta(self, tab, st, cache, mv):
        import jax.numpy as jnp
        return jnp.zeros(st['X'].shape[-1], st['X'].dtype), None

    def delta_nuis(self, st, cache, nuis):
        import jax.numpy as jnp
        return jnp.zeros(st['X'].shape[-1], st['X'].dtype), None


def step_time(eng, movers, chains, sweeps, drop=(), reps=3, seed=0):
    """seconds per MC step (all chains) of sweeps over `movers` (indexes,
       in sweep order), with the deltas of the terms `drop` (indexes) left
       out. Other movers don't move; the draws are the engine's."""
    import jax
    import jax.numpy as jnp
    movers = np.asarray(movers)
    terms = eng.terms
    eng.terms = [_Zero(t) if i in drop else t for i, t in enumerate(terms)]
    try:
        used = np.unique(eng.gid[movers])
        remap = np.zeros(len(eng.groups), np.int32)
        remap[used] = np.arange(len(used))
        branches = [eng._branch(eng.groups[g]) for g in used]
        gid = jnp.asarray(remap[eng.gid[movers]])
        jid = jnp.asarray(eng.jid[movers])

        def sweep(st):
            U, Z = eng.draws(st['key'], st['sweep'], st['X'].dtype)
            kT = st['kT']
            core = {k: st[k] for k in ('X', 'quat', 'nuis', 'E', 'cache')}

            def step(core, xs):
                g, j, u, z = xs
                return jax.lax.switch(g, branches, core, j, u, z, kT)
            core, _ = jax.lax.scan(step, core, (gid, jid, U[movers],
                                                Z[movers]))
            return dict(st, **core, sweep=st['sweep'] + 1)
        run = jax.jit(lambda st, n: jax.lax.fori_loop(
            0, n, lambda _, st: sweep(st), st), static_argnums=1,
            donate_argnums=0)
        st = run(eng.init(chains, seed), sweeps)
        jax.block_until_ready(st)
        ts = []
        for _ in range(reps):
            t0 = time.time()
            st = run(st, sweeps)
            jax.block_until_ready(st)
            ts.append(time.time() - t0)
    finally:
        eng.terms = terms
    return float(np.median(ts)) / (sweeps * len(movers))


def profile(s, chains, sweeps, kinds=None, out=print):
    """the table described in the module doc, for a serial system"""
    import jax
    from .engine import Engine
    eng = Engine(s)
    kind = np.array([G.kind for G in eng.groups])[eng.gid]
    out(f'{s.name}: {len(s.movers)} movers in {len(eng.groups)} groups, '
        f'{chains} chains, {eng.dtype().__name__}, '
        f'{jax.devices()[0].device_kind}; us per MC step (all chains)')
    res = {}
    for k in dict.fromkeys(kind):
        if kinds and k not in kinds:
            continue
        mv = np.nonzero(kind == k)[0]
        base = step_time(eng, mv, chains, sweeps)
        out(f'{k} x{len(mv)}: {base * 1e6:.1f} us '
            f'({chains / base / 1e6:.2f} M MC steps/s)')
        tis = sorted({ti for g in np.unique(eng.gid[mv])
                      for ti in eng.groups[g].terms})
        by = {}
        for ti in tis:
            by.setdefault(s.terms[ti].name, []).append(ti)
        res[k] = dict(all=base)
        for name, drop in by.items():
            t = step_time(eng, mv, chains, sweeps, drop)
            res[k][name] = base - t
            out(f'    {name:18s} {(base - t) * 1e6:8.1f} us '
                f'{100 * (base - t) / base:5.1f}%')
        if tis:
            t = step_time(eng, mv, chains, sweeps, tis)
            res[k]['no terms'] = t
            out(f'    {"(no terms)":18s} {t * 1e6:8.1f} us '
                f'{100 * t / base:5.1f}%')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--chains', type=int, default=4096)
    ap.add_argument('--sweeps', type=int, default=20)
    ap.add_argument('--float64', action='store_true')
    ap.add_argument('--cov', default='frozen')
    ap.add_argument('--kinds', default=None,
                    help='mover kinds to time, e.g. ball,rb (default: all)')
    a = ap.parse_args()
    from . import xla_defaults
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', a.float64)
    from . import sources, system
    s = system.extract(sources.load(a.system), cov_mode=a.cov)
    profile(s, a.chains, a.sweeps, a.kinds.split(',') if a.kinds else None)


if __name__ == '__main__':
    main()
