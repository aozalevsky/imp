"""Exactness checks.

  python -m jax_mc.check terms polii [--states 6] [--cov frozen|rotate]
  python -m jax_mc.check decisions polii [--chains 16 --sweeps 5 --kT 1,10]
      [--ref-chains R] [--sampler my.module:factory]
  python -m jax_mc.check drift polii [--chains 256 --sweeps 100]

terms: at random states near the initial one, compare
  * positions: every row's coordinates from the degrees of freedom against
    IMP's (after writing the same state into the model);
  * every term's full() in float64 against the C++ restraints it replaces;
  * the total (terms + constants) against the C++ scoring function.
"""
import argparse
import time

import numpy as np


def terms(s, n_states=6, seed=0, scale=3.0, out=print, cpp_float=True):
    import jax
    jax.config.update('jax_enable_x64', True)
    from .terms import basic
    basic.CrossLinks.CPP_FLOAT = cpp_float
    import jax.numpy as jnp
    import IMP
    import IMP.core
    C = n_states
    dofs = s.random_dofs(C, seed=seed, scale=scale)
    jd = {k: jnp.asarray(v) for k, v in dofs.items()}
    X = np.asarray(s.positions(jd, jnp))
    st = dict(X=jnp.asarray(X), quat=jd['quat'], nuis=jd['nuis'])
    # eager on purpose: under jit XLA may contract float32 arithmetic, which
    # breaks the single-precision emulation of C++ (CrossLinks.CPP_FLOAT)
    t0 = time.time()
    vals = {}
    for t in s.terms:
        vals[t.label] = np.asarray(t.full(st), np.float64)
    t_jax = time.time() - t0
    sf = IMP.core.RestraintsScoringFunction(s.source.restraints)
    cpp = {t.label: np.zeros(C) for t in s.terms}
    cpp_total, dx = np.zeros(C), np.zeros(C)
    d0 = s.get_imp_dofs()
    masks = {}
    for c in range(C):
        s.set_imp_dofs(dofs, c)
        xyz = np.array([IMP.core.XYZ(s.m, IMP.ParticleIndex(int(p)))
                        .get_coordinates() for p in s.row_pi])
        dx[c] = np.abs(xyz - X[:, :, c].T).max()
        for t in s.terms:
            cpp[t.label][c] = t.cpp()
            if hasattr(t, 'cpp_masks'):
                masks.setdefault(t.label, []).append(t.cpp_masks())
        cpp_total[c] = sf.evaluate(False)
    s.set_imp_dofs(d0, 0)
    out(f'{s.name}: {s.n_rows} rows ({len(s.static_rows)} static), '
        f'{s.nb} bodies, {s.nbead} beads, {s.n_nuis} nuisances, '
        f'{len(s.movers)} movers ({s.mode}); cov {s.cov_mode}; '
        f'{C} states; JAX {t_jax:.1f}s')
    out(f'  positions: max |X - IMP| = {dx.max():.2e}')
    out(f'  {"term":28s} {"n":>5s} {"C++ (state 0)":>16s} '
        f'{"max |JAX-C++|":>14s} {"max rel":>9s}')
    worst = 0.
    for t in s.terms:
        a, b = vals[t.label], cpp[t.label]
        err = np.abs(a - b)
        rel = err / np.maximum(np.abs(b), 1e-300)
        worst = max(worst, rel.max())
        out(f'  {t.label:28s} {len(t.restraints):5d} {b[0]:16.8g} '
            f'{err.max():14.3e} {rel.max():9.1e}')
        if t.label in masks:
            mk = [np.stack(x, -1) for x in zip(*masks[t.label])]
            v = np.asarray(t.full(st, masks=mk), np.float64)
            e2 = np.abs(v - b)
            out(f'  {"  with C++ pair lists":28s} {"":5s} {"":16s} '
                f'{e2.max():14.3e} {(e2 / np.abs(b)).max():9.1e}  '
                f'per state {np.array2string(e2, precision=1)}')
    tot = sum(vals.values()) + s.constant
    err = np.abs(tot - cpp_total)
    out(f'  {"constant":28s} {len(s.constant_restraints):5d} '
        f'{s.constant:16.8g}')
    out(f'  {"TOTAL":28s} {"":5s} {cpp_total[0]:16.8g} {err.max():14.3e} '
        f'{(err / np.abs(cpp_total)).max():9.1e}')
    if s.skipped:
        out(f'  skipped: {len(s.skipped)} restraints with weight 0')
    return dict(vals=vals, cpp=cpp, total=tot, cpp_total=cpp_total, dx=dx)


def compare(a, b, sweeps=5, chains=16, kTs=(1.0,), seed=0, ref_chains=None,
            out=print):
    """Level D between two samplers that follow the sampler protocol
       (DESIGN.md, "From the engine to a specialized sampler"): a, under
       test, runs `chains` chains; b, the reference, runs the first
       ref_chains of them (a chain's draws don't depend on the number of
       chains). Reports differing decisions, acceptance per mover kind,
       a's tracked energy against a rescore, and the energy and positions
       of chains where no decision differs."""
    import jax
    s = a.s
    ra = jax.jit(a.run, static_argnums=(1, 2))
    rb = jax.jit(b.run, static_argnums=(1, 2))
    R = chains if ref_chains is None else min(ref_chains, chains)
    kinds = np.array([mv.kind for mv in s.movers])

    def name(x):
        return type(x).__name__ + (' (full rescore)' if getattr(
            x, 'incremental', True) is False else '')
    out(f'{s.name}: {name(a)} against {name(b)}; '
        f'{len(s.movers)} movers; {chains} chains ({R} in the reference), '
        f'{sweeps} sweeps, '
        f'{"float64" if jax.config.jax_enable_x64 else "float32"}, '
        f'{jax.devices()[0].device_kind}')
    if hasattr(a, 'groups'):
        out('  groups: ' + ', '.join(f'{G.kind} x{len(G.movers)}'
                                     for G in a.groups))
    res = []
    for kT in kTs:
        sa, sb = a.init(chains, seed, kT), b.init(R, seed, kT)
        t0 = time.time()
        sa, da = ra(sa, sweeps, True)
        da = np.asarray(da)
        t1 = time.time()
        sb, db = rb(sb, sweeps, True)
        db = np.asarray(db)
        t2 = time.time()
        diff = da[..., :R] != db
        hit = diff.any(axis=(0, 1))
        ok = ~hit
        acc = ', '.join(f'{k} {da[:, kinds == k].mean():.2f}'
                        for k in dict.fromkeys(kinds))
        out(f'  kT {kT:g}: {int(diff.sum())} of {diff.size} decisions differ'
            f' ({int(hit.sum())} chains); acceptance {acc}')
        if diff.any():
            sw, mo, ch = np.argwhere(diff)[0]
            out(f'    first: sweep {sw}, mover {mo} ({kinds[mo]}), chain {ch}')
        r = dict(kT=kT, ndiff=int(diff.sum()), n=diff.size)
        msg = []
        if hasattr(a, 'resync'):
            Ea = np.asarray(a.total(sa))
            Er = np.asarray(a.total(jax.jit(a.resync)(sa)))
            r['dE_track'] = float(np.abs(Ea - Er).max())
            msg.append(f'tracked - rescored energy: max {r["dE_track"]:.2e}')
        if ok.any():
            Xa = s.positions(a.dofs(sa))[..., :R][..., ok]
            Xb = s.positions(b.dofs(sb))[..., ok]
            r['dX'] = float(np.abs(Xa - Xb).max())
            dE = np.abs(np.asarray(a.total(sa))[:R]
                        - np.asarray(b.total(sb)))[ok].max()
            msg.append(f'vs reference: energy {dE:.2e}, positions '
                       f'{r["dX"]:.2e}')
        msg.append(f'time {t1 - t0:.1f} s / {t2 - t1:.1f} s (incl. compile)')
        out('    ' + '; '.join(msg))
        res.append(r)
    return res


def decisions(s, sweeps=5, chains=16, kTs=(1.0,), seed=0, ref_chains=None,
              sampler=None, out=print):
    """Level D. Without `sampler`: the incremental engine against the
       full-rescore engine (same draws). With sampler (a factory
       s -> sampler): that sampler against the engine."""
    from .engine import Engine
    if sampler is None:
        a, b = Engine(s), Engine(s, incremental=False)
    else:
        a, b = sampler(s), Engine(s)
    return compare(a, b, sweeps, chains, kTs, seed, ref_chains, out)


def _factory(spec):
    """'package.module:name' -> the object"""
    import importlib
    mod, name = spec.split(':')
    return getattr(importlib.import_module(mod), name)


def drift(s, sweeps=100, chains=256, seed=0, kT=None, out=print):
    """Level F: tracked energy against a rescore after `sweeps` sweeps (run
       in float32 with x64 off)"""
    import jax
    from .engine import Engine
    a = Engine(s)
    st = a.init(chains, seed, kT)
    st = jax.jit(a.run, static_argnums=(1, 2))(st, sweeps)
    Ea = np.asarray(a.total(st), np.float64)
    st2 = jax.jit(a.resync)(st)
    Er = np.asarray(a.total(st2), np.float64)
    per = np.abs(np.asarray(st['E']) - np.asarray(st2['E'])).max(axis=1)
    out(f'{s.name}: {chains} chains, {sweeps} sweeps, '
        f'{a.dtype().__name__}: |tracked - rescored| max '
        f'{np.abs(Ea - Er).max():.3e} (energy ~{np.abs(Er).mean():.4g}); '
        'per term ' + ', '.join(f'{t.label} {e:.1e}'
                                for t, e in zip(a.terms, per)))
    return np.abs(Ea - Er).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['terms', 'decisions', 'drift'])
    ap.add_argument('system')
    ap.add_argument('--states', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scale', type=float, default=3.0)
    ap.add_argument('--cov', default='frozen')
    ap.add_argument('--chains', type=int, default=16)
    ap.add_argument('--ref-chains', type=int, default=None)
    ap.add_argument('--sampler', default=None,
                    help='module:factory, factory(system) -> sampler, '
                    'checked against the engine')
    ap.add_argument('--sweeps', type=int, default=5)
    ap.add_argument('--kT', default='1')
    ap.add_argument('--float32', action='store_true')
    a = ap.parse_args()
    from . import xla_defaults
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', not a.float32)
    from . import sources, system
    src = sources.load(a.system)
    s = system.extract(src, cov_mode=a.cov)
    if a.what == 'terms':
        terms(s, a.states, a.seed, a.scale)
    elif a.what == 'decisions':
        decisions(s, a.sweeps, a.chains,
                  [float(x) for x in a.kT.split(',')], a.seed, a.ref_chains,
                  _factory(a.sampler) if a.sampler else None)
    else:
        drift(s, a.sweeps, a.chains, a.seed)


if __name__ == '__main__':
    main()
