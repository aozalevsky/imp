"""Recipe R2: Unrolled sweep with static row offsets, in-place
`dynamic_update_slice_in_dim` behind `optimization_barrier`, and flat tuple
carry.

Specialization level:
  Level 3 (Sweep override `UnrolledEngine(Engine)` and in-place slice
  primitives `barrier_slice`, `barrier_update` used in `few_bodies_beads`).

Preconditions (asserted):
  1. `s.mode == 'serial'` (SerialMover order).
  2. Number of movers is small enough to unroll (`len(s.movers) <= max_movers`,
     default 128; compile time grows with the number of movers, see
     `templates/many_movers`).

A coordinate mover whose rows `mv.rows` form a contiguous slice
`[row0, row0 + len(mv.rows))` gets the in-place update. Any other one (a
TransformMover over several bodies and beads) takes the engine's own step
for its group, with its tables as constants.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jax_mc import engine as E


def check_preconditions(s, max_movers=128):
    """Assert R2 preconditions on System s."""
    assert s.mode == 'serial', (
        f'R2 requires SerialMover mode, got "{s.mode}"')
    assert 0 < len(s.movers) <= max_movers, (
        f'R2 requires 1..{max_movers} movers for unrolling, got '
        f'{len(s.movers)}')


def contiguous(mv):
    """do a coordinate mover's rows form one slice of X?"""
    r = np.asarray(mv.rows, np.int64)
    return len(r) > 0 and np.array_equal(r, np.arange(r[0], r[0] + len(r)))


def barrier_slice(X, start, length, axis=1):
    """Slice `length` rows starting at Python int `start` along `axis`, hiding
       `start` behind `jax.lax.optimization_barrier` so XLA's HloSimplifier
       cannot turn `dynamic_slice` into a static `kSlice`."""
    od = jax.lax.optimization_barrier(jnp.int32(start))
    return jax.lax.dynamic_slice_in_dim(X, od, int(length), axis=axis)


def barrier_update(X, new_slice, start, axis=1):
    """In-place `dynamic_update_slice_in_dim` at Python int `start` behind
       `jax.lax.optimization_barrier`, satisfying XLA GPU's
       `CanEmitFusedDynamicUpdateSliceInPlace` check inside unrolled loops."""
    od = jax.lax.optimization_barrier(jnp.int32(start))
    return jax.lax.dynamic_update_slice_in_dim(X, new_slice, od, axis=axis)


def barrier_select_update(X, acc, Xn, start, length, axis=1):
    """Conditionally update `X[:, start:start+length]` with `Xn` where `acc`
       is True, reading the old slice via `barrier_slice` so XLA aliases the
       output buffer to `X` in-place with zero full-buffer `kCopy`."""
    od = jax.lax.optimization_barrier(jnp.int32(start))
    Xo = jax.lax.dynamic_slice_in_dim(X, od, int(length), axis=axis)
    return jax.lax.dynamic_update_slice_in_dim(
        X, jnp.where(acc, Xn, Xo), od, axis=axis)


class UnrolledEngine(E.Engine):
    """Level-3 specialization of Engine: unrolls the SerialMover sweep in
       Python and writes accepted coordinate moves back via in-place
       `dynamic_update_slice_in_dim` behind `optimization_barrier`."""
    MAX_MOVERS = 128

    def __init__(self, s, incremental=True):
        check_preconditions(s, self.MAX_MOVERS)
        super().__init__(s, incremental=incremental)
        self._step_fns = [self._make_unrolled_step(i)
                          for i in range(len(s.movers))]

    def _make_unrolled_step(self, i):
        s = self.s
        mv_spec = s.movers[i]
        g = int(self.gid[i])
        j = int(self.jid[i])
        G = self.groups[g]
        ttabs = {ti: {k: jnp.asarray(v[j]) for k, v in tt.items()}
                 for ti, tt in G.term_tabs.items()}
        if G.kind == 'normal':
            k = int(G.tab['nuis'][j])
            sigma = float(G.tab['sigma'][j])
            lo = float(s.nuis_lower[k])
            hi = float(s.nuis_upper[k])

            def step(core, u, z, kT):
                x = core['nuis'][k]
                xn = jnp.clip(x + sigma * z[0], lo, hi)
                E_new, cache, acc = self._decide(
                    G, {ti: {tk: tv[None] for tk, tv in tt.items()}
                        for ti, tt in ttabs.items()},
                    0, core, u, kT, nuis=core['nuis'].at[k].set(xn))
                nuis = barrier_update(
                    core['nuis'], jnp.where(acc, xn, x)[None], k, axis=0)
                return dict(core, nuis=nuis, E=E_new, cache=cache), acc
            return step

        if not contiguous(mv_spec):
            branch = self._branches[g]      # the engine's step, constant j
            return lambda core, u, z, kT: branch(core, j, u, z, kT)

        rows = np.asarray(mv_spec.rows, np.int64)
        r0, nr = int(rows[0]), len(rows)
        M = G.tab['rows'].shape[1]
        rows_pad = jnp.asarray(G.tab['rows'][j])
        loc = jnp.asarray(G.tab['loc'][j, :, :nr])[..., None]
        step_sz = float(G.tab['step'][j])
        rot_sz = float(G.tab['rot'][j])
        bodies = jnp.asarray(G.tab['bodies'][j])
        cslot = jnp.asarray(G.tab['cslot'][j])
        bslot = jnp.asarray(G.tab['bslot'][j, :nr])
        cw = jnp.asarray(G.tab['cw'][j, :nr])

        def step(core, u, z, kT):
            X, quat = core['X'], core['quat']
            Xo_c = barrier_slice(X, r0, nr, axis=1)             # (3, nr, C)
            v = E._ball(z, u, step_sz)
            b_arg = qo = qn = None
            if G.kind == 'ball':
                Xn_c = Xo_c + v[:, None]
            else:
                b_arg = bodies
                qo = jnp.take(quat, bodies, axis=1, mode='clip')
                qr = E._axis_angle(z, u, rot_sz)
                qn = E._normalize(E._qmul(qr[:, None], qo))
                tb = Xo_c[:, cslot]
                if G.kind == 'rb':
                    tn = tb + v[:, None]
                else:
                    c = jnp.sum(cw[None, :, None] * Xo_c, axis=1) / jnp.sum(cw)
                    Rr = E.quat_to_rot(qr, jnp)

                    def tt(y):
                        return (E._rotate(Rr, y - c[:, None]) + c[:, None]
                                + v[:, None])
                    tn = tt(tb)
                R = E.quat_to_rot(qn, jnp)
                if G.kind == 'rb':
                    tm = tn
                else:
                    b = jnp.maximum(bslot, 0)
                    R, tm = R[:, :, b], tn[:, b]
                Xn_c = (R[:, 0] * loc[0] + R[:, 1] * loc[1] + R[:, 2] * loc[2]
                        + tm)
                if G.kind == 'transform':
                    Xn_c = jnp.where((bslot >= 0)[None, :, None], Xn_c,
                                     tt(Xo_c))
            if nr < M:
                pad_w = ((0, 0), (0, M - nr), (0, 0))
                Xo_p = jnp.pad(Xo_c, pad_w)
                Xn_p = jnp.pad(Xn_c, pad_w)
            else:
                Xo_p, Xn_p = Xo_c, Xn_c
            mv = E.Move(core, rows_pad, Xo_p, Xn_p, b_arg, qn)
            E_new, cache, acc = self._decide(
                G, {ti: {tk: tv[None] for tk, tv in tt.items()}
                    for ti, tt in ttabs.items()},
                0, core, u, kT, mv=mv)
            X = barrier_update(X, jnp.where(acc, Xn_c, Xo_c), r0, axis=1)
            if b_arg is not None:
                if G.kind == 'rb':
                    b0 = int(mv_spec.bodies[0])
                    quat = barrier_update(
                        quat, jnp.where(acc, qn, qo), b0, axis=1)
                else:
                    quat = quat.at[:, bodies].set(
                        jnp.where(acc, qn, qo), mode='drop')
            return dict(core, X=X, quat=quat, E=E_new, cache=cache), acc
        return step

    def sweep(self, st):
        U, Z = self.draws(st['key'], st['sweep'], st['X'].dtype)
        kT = st['kT']
        core = {k: st[k] for k in ('X', 'quat', 'nuis', 'E', 'cache')}
        accs = []
        for i, fn in enumerate(self._step_fns):
            core, acc = fn(core, U[i], Z[i], kT)
            accs.append(acc)
        acc_all = jnp.stack(accs, axis=0)
        return dict(st, **core, sweep=st['sweep'] + 1,
                    n_acc=st['n_acc'] + acc_all), acc_all


def factory(s):
    return UnrolledEngine(s)
