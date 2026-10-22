"""Per-mover dE from fixed states (all proposals accepted) and full terms,
for comparing sampler variants against a float64 dense reference.

    python verify.py --out ref.npz --x64 --grid 0     # float64, dense
    python verify.py --out f32.npz --soa [--body-grid] [--normalize]
    python verify.py --compare ref.npz f32.npz ...

The first call equilibrates --chains chains and saves them to --states
(one file per EV resolution by default); later calls reuse them.
--normalize renormalizes the saved float32 quaternions in float64, which
--body-grid needs for a float64 check (the per-body grids invert a body's
frame with R^T). TUT_EV_RES=1 in the
environment checks the tutorial with EV at resolution 1.
"""
import argparse
import dataclasses
import os

import numpy as np


def delta_e(smp, dofs, nrm, uni):
    import jax
    import jax.numpy as jnp
    dt = smp.dt
    if getattr(smp, 'soa', False):
        return delta_e_soa(smp, dofs, nrm, uni)
    q, t, b = dofs
    st = smp.initial_state(q.shape[0], quat=q, trans=t, bead=b)
    n = q.shape[0]
    st.energy = jnp.zeros(n, dt)
    st.kT = jnp.full(n, 1e30, dt)

    def one(s_, nr, un):
        out = []
        for i, p in enumerate(smp.steps):
            f = smp._bead_step if p['kind'] == 'bead' else smp._rb_step
            out.append(f(i, dataclasses.replace(s_), nr, un).energy)
        return jnp.stack(out)
    g = jax.jit(jax.vmap(one))
    return np.asarray(g(st, jnp.asarray(nrm, dt), jnp.asarray(uni, dt)),
                      np.float64)


def delta_e_soa(smp, dofs, nrm, uni):
    import jax
    import jax.numpy as jnp
    dt = smp.dt
    q, t, b = dofs
    st = smp.initial_state(q.shape[0], quat=q, trans=t, bead=b)
    n = q.shape[0]
    st.energy = jnp.zeros(n, dt)
    st.kT = jnp.full(n, 1e30, dt)

    def all_(s_, nr, un):
        out = []
        for i, p in enumerate(smp.steps):
            f = smp._bead_step if p['kind'] == 'bead' else smp._rb_step
            out.append(f(i, dataclasses.replace(s_), nr, un).energy)
        return jnp.stack(out, -1)
    nr = jnp.asarray(np.moveaxis(nrm, 0, -1), dt)
    un = jnp.asarray(np.moveaxis(uni, 0, -1), dt)
    return np.asarray(jax.jit(all_)(st, nr, un), np.float64)


def compare(paths):
    ref = dict(np.load(paths[0]))
    kinds = ref['kinds']
    u = np.random.default_rng(9).random(ref['dE'].shape)
    acc_r = (ref['dE'] <= 0) | (u < np.exp(-np.clip(ref['dE'], 0, 700)))
    print('reference', paths[0])
    for p in paths[1:]:
        d = dict(np.load(p))
        print(p)
        for kind in ('bead', 'rb'):
            sel = kinds == kind
            err = np.abs(d['dE'][:, sel] - ref['dE'][:, sel])
            acc = (d['dE'] <= 0) | (u < np.exp(-np.clip(d['dE'], 0, 700)))
            flip = 100 * (acc != acc_r)[:, sel].mean()
            print('  %-4s dE err median %.2e max %.2e; decisions flipped '
                  '%.4f%%' % (kind, np.median(err), err.max(), flip))
        e = np.abs(d['E'] - ref['E'])
        print('  full energy err median %.2e max %.2e' % (np.median(e),
                                                          e.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--x64', action='store_true')
    ap.add_argument('--grid', type=int, default=None)
    ap.add_argument('--states', default=None,
                    help='equilibrated states, made and saved if missing '
                         '(default: verify_states.npz, or '
                         'verify_states_res<R>.npz with TUT_EV_RES=R)')
    ap.add_argument('--chains', type=int, default=512)
    ap.add_argument('--compare', nargs='+')
    ap.add_argument('--soa', action='store_true')
    ap.add_argument('--body-grid', action='store_true')
    ap.add_argument('--normalize', action='store_true',
                    help='normalize the states\' quaternions in float64')
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if a.states is None:
        res = os.environ.get('TUT_EV_RES')
        a.states = f'verify_states_res{res}.npz' if res else \
            'verify_states.npz'
    import jax
    if a.x64:
        jax.config.update('jax_enable_x64', True)
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode='rotate')
    kw = {} if a.grid is None else dict(grid=bool(a.grid))
    if a.soa:
        from sampler_soa import TutSamplerSoA
        smp = TutSamplerSoA(s, dtype='float64' if a.x64 else 'float32',
                            body_grid=a.body_grid)
        smp.soa = True
    else:
        smp = TutSampler(s, dtype='float64' if a.x64 else 'float32', **kw)
    print('variant: x64', a.x64, 'split', smp.split, 'cache', smp.cache,
          'grid', getattr(smp, 'grid', None), flush=True)
    if not os.path.exists(a.states):
        st = smp.initial_state(a.chains, seed=5)
        run = smp.make_run(10)
        for _ in range(300):
            st = smp.resync(run(st))
        q, t, b = smp.dofs(st)
        rng = np.random.default_rng(5)
        nrm = rng.standard_normal((a.chains, smp.n_movers, 6))
        uni = rng.random((a.chains, smp.n_movers, 3))
        uni[..., 1] = 0.
        np.savez(a.states, q=q, t=t, b=b, nrm=nrm, uni=uni)
        print('saved states', a.states)
    d = np.load(a.states)
    q = d['q']
    if a.normalize:
        # float32 snapshots have |q| - 1 ~ 1e-7; the per-body grids invert
        # a body's frame with R^T, exact only for a unit quaternion
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    dofs = (q, d['t'], d['b'])
    dE = delta_e(smp, dofs, d['nrm'], d['uni'])
    st = smp.initial_state(dofs[0].shape[0], quat=dofs[0], trans=dofs[1],
                           bead=dofs[2])
    st, tr = smp.resync(st, terms=True)
    np.savez(a.out, dE=dE, E=np.asarray(st.energy, np.float64),
             kinds=np.array([p['kind'] for p in smp.steps]))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
