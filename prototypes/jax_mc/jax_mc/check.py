"""Exactness checks.

  python -m jax_mc.check terms polii [--states 6] [--cov frozen|rotate]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['terms'])
    ap.add_argument('system')
    ap.add_argument('--states', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scale', type=float, default=3.0)
    ap.add_argument('--cov', default='frozen')
    a = ap.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    from . import sources, system
    src = sources.load(a.system)
    s = system.extract(src, cov_mode=a.cov)
    terms(s, a.states, a.seed, a.scale)


if __name__ == '__main__':
    main()
