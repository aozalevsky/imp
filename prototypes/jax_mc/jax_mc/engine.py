"""The reference sampler: incremental, chain-last, any mover order.

One MC step proposes one mover, in IMP's SerialMover order, as IMP's
MonteCarlo does with a SerialMover; a sweep is one step for every mover.
All chains do the same mover at the same time.

The sweep is a lax.scan over the movers. Movers are grouped by kind and
size (rows, padded to a power of 2), each group has its tables (one row
per mover) and its branch of a lax.switch. A branch proposes, asks the
terms its mover can change for their energy change (Term.delta), applies
Metropolis, and writes the accepted rows back by a masked select.

Engine(s, incremental=False) is the same sampler with full rescores: the
same proposals and draws, but every term rescored from scratch with the
move applied. It is the reference for check.decisions.

State (a dict):
  X      (3, rows, C)   positions; beads' degrees of freedom live here and a
                        body's translation is its center row (local 0)
  quat   (4, nb, C)     body orientations
  nuis   (N, C)         nuisances
  E      (T, C)         energy per term (total: sum + system constant)
  cache  tuple          per-term caches (Term.cache)
  kT     (C,)           temperature per chain
  key    (C,)           per-chain random keys
  sweep  ()             sweeps done: the random numbers' sweep index
  n_acc  (movers, C)    accepted moves per mover

Random numbers: the contract every sampler built on the core follows, so
that check.decisions can compare it with this one:
  chain c      key_c = fold_in(key(seed), c)
  sweep s      k = fold_in(key_c, s); U = uniform(k, (n_movers, 3)),
               Z = normal(fold_in(k, 1), (n_movers, 6)), in the run's dtype
  mover i      ball or translation: Z[i, 0:3] / |Z[i, 0:3]| * step
                   * cbrt(U[i, 0]), in global coordinates (also for balls on
                   local coordinates: the distribution is the same)
               rotation: axis Z[i, 3:6] / |Z[i, 3:6]|, angle max_rot
                   * (2 U[i, 1] - 1); q' = normalize(q_axis_angle * q)
               nuisance: clip(x + sigma * Z[i, 0], lower, upper)
               Metropolis: accept if U[i, 2] < exp(-dE / kT)

Moves, as IMP makes them:
  ball       a bead: x + ball step
  rb         a body: q' = q_r q, t' = t + ball step
  transform  bodies and beads together (IMP.pmi.TransformMover): about the
             center c = (sum of beads + sum of body centers weighted by
             their number of members) / (count), y -> R_r (y - c) + c + v
  normal     a nuisance
"""
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from .system import BEAD, MEMBER, Unsupported, quat_to_rot
from .terms import pad


def _bucket(n):
    return 1 << max(int(n) - 1, 0).bit_length()


def _unit(v):
    return v / jnp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _ball(z, u, step):
    """(3, C) uniform in a ball of radius step"""
    return _unit(z[0:3]) * (step * jnp.cbrt(u[0]))


def _axis_angle(z, u, max_rot):
    """(4, C) rotation by up to max_rot about a random axis"""
    half = 0.5 * max_rot * (2. * u[1] - 1.)
    return jnp.concatenate([jnp.cos(half)[None],
                            _unit(z[3:6]) * jnp.sin(half)])


def _qmul(a, b):
    return jnp.stack([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def _normalize(q):
    return q / jnp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def _rotate(R, v):
    """R (3, 3, C) applied to v (3, k, C)"""
    return (R[:, 0][:, None] * v[0][None] + R[:, 1][:, None] * v[1][None]
            + R[:, 2][:, None] * v[2][None])


class Move:
    """A proposed coordinate move, as Term.delta sees it.
       rows (M,) padded with n_rows; Xo, Xn (3, M, C) old and new positions
       of those rows; bodies (kb,) padded with nb and their new
       orientations q_new (4, kb, C), or None for a bead move."""

    def __init__(self, st, rows, Xo, Xn, bodies=None, q_new=None):
        self.st, self.rows, self.Xo, self.Xn = st, rows, Xo, Xn
        self.bodies, self.q_new = bodies, q_new

    def pos(self, rows, slot):
        """positions (3, *rows.shape, C) of rows after the move; slot: the
           row's slot in the mover, -1 if the move doesn't change it"""
        new = jnp.take(self.Xn, jnp.maximum(slot, 0), axis=1)
        return jnp.where((slot >= 0)[None, ..., None], new,
                         self.st['X'][:, rows])

    def apply(self, st):
        """st with the move applied (copies X: for full rescores)"""
        X = st['X'].at[:, self.rows].set(self.Xn, mode='drop')
        quat = st['quat']
        if self.bodies is not None:
            quat = quat.at[:, self.bodies].set(self.q_new, mode='drop')
        return dict(st, X=X, quat=quat)


@dataclass
class Group:
    kind: str
    movers: list                    # mover indexes, in SerialMover order
    tab: dict                       # host tables, one row per mover
    terms: list                     # terms the group's moves can change
    term_tabs: dict = field(default_factory=dict)   # term index -> tables


class Engine:
    def __init__(self, s, incremental=True):
        if s.mode != 'serial':
            raise Unsupported(f'mover mode "{s.mode}" (collective template)')
        self.s, self.incremental = s, incremental
        self.terms = list(s.terms)
        kind, ref = s.row_kind, s.row_ref
        self.center_row = np.array(
            [np.nonzero((kind == MEMBER) & (ref == b))[0][0]
             for b in range(s.nb)], np.int64)
        assert np.all(s.row_is_center[self.center_row])
        self.bead_row = np.nonzero(kind == BEAD)[0]
        import IMP
        import IMP.core
        self.nmem = np.array([IMP.core.RigidBody(
            s.m, IMP.ParticleIndex(int(b))).get_number_of_members()
            for b in s.body_pi], np.float64)
        keys = {}
        for i, mv in enumerate(s.movers):
            if mv.kind == 'normal':
                k = ('normal', 0, 0)
            elif mv.kind == 'ball':
                k = ('ball', 1, 0)
            else:
                k = (mv.kind, _bucket(len(mv.rows)), _bucket(len(mv.bodies)))
            keys.setdefault(k, []).append(i)
        self.groups = [self._group(k, v) for k, v in
                       sorted(keys.items(), key=lambda kv: kv[1][0])]
        n = len(s.movers)
        self.gid = np.zeros(n, np.int32)
        self.jid = np.zeros(n, np.int32)
        for g, G in enumerate(self.groups):
            for j, i in enumerate(G.movers):
                self.gid[i], self.jid[i] = g, j
        self._branches = [self._branch(G) for G in self.groups]

    # ---------------- host tables ---------------------------------------
    def _group(self, key, movers):
        s = self.s
        kind = key[0]
        mvs = [s.movers[i] for i in movers]
        if kind == 'normal':
            nus = [s.nuis_of[int(s._moved_nuis_order[mv.nuis])] for mv in mvs]
            tab = dict(nuis=np.array(nus, np.int64),
                       sigma=np.array([mv.sigma for mv in mvs]))
            terms = [ti for ti, t in enumerate(self.terms)
                     if t.nuis_read is None
                     or np.isin(t.nuis_read, nus).any()]
            return Group(kind, movers, tab, terms)
        M, KB = key[1], max(key[2], 1)
        rk, rr = s.row_kind, s.row_ref
        rows, loc, bslot, cw, bodies, cslot = [], [], [], [], [], []
        for mv in mvs:
            r = np.array(mv.rows, np.int64)
            bl = [int(b) for b in mv.bodies]
            bpos = {b: k for k, b in enumerate(bl)}
            rows.append(r)
            loc.append(s.row_local[r].T)
            bslot.append([bpos[int(rr[x])] if rk[x] == MEMBER else -1
                          for x in r])
            cw.append([self.nmem[rr[x]] if s.row_is_center[x] else
                       float(rk[x] == BEAD) for x in r])
            bodies.append(bl)
            cslot.append([int(np.nonzero(r == self.center_row[b])[0][0])
                          for b in bl])
            if kind == 'ball':
                assert len(r) == 1 and rk[r[0]] == BEAD
            if kind == 'rb':
                assert len(bl) == 1 and cslot[-1] == [0]
        L = np.zeros((len(mvs), 3, M))
        for i, x in enumerate(loc):
            L[i, :, :x.shape[1]] = x
        tab = dict(rows=pad(rows, s.n_rows, M), loc=L,
                   bslot=pad(bslot, -1, M),
                   cw=pad(cw, 0., M, np.float64),
                   bodies=pad(bodies, s.nb, KB), cslot=pad(cslot, 0, KB),
                   step=np.array([mv.radius if kind == 'ball' else
                                  mv.max_trans for mv in mvs]),
                   rot=np.array([mv.max_rot for mv in mvs]))
        moves = [(np.array(mv.rows, np.int64), [int(b) for b in mv.bodies])
                 for mv in mvs]
        terms = [ti for ti, t in enumerate(self.terms)
                 if t.rows_read is None
                 or any(np.isin(t.rows_read, r).any() for r, _ in moves)]
        G = Group(kind, movers, tab, terms)
        if self.incremental:
            G.term_tabs = {ti: self.terms[ti].move_tables(s, moves)
                           for ti in terms}
        return G

    # ---------------- device side ---------------------------------------
    def _decide(self, G, ttabs, j, core, u, kT, mv=None, nuis=None):
        """energy change of the terms, Metropolis; returns E, cache, acc"""
        E, cache = core['E'], list(core['cache'])
        st = dict(X=core['X'], quat=core['quat'], nuis=core['nuis'])
        if self.incremental:
            dEs, new = [], {}
            for ti in G.terms:
                t, c = self.terms[ti], cache[ti]
                if mv is not None:
                    tab = {k: v[j] for k, v in ttabs[ti].items()}
                    d, new[ti] = t.delta(tab, st, c, mv)
                else:
                    d, new[ti] = t.delta_nuis(st, c, nuis)
                dEs.append((ti, d))
            dE = sum(d for _, d in dEs)
            acc = u[2] < jnp.exp(-dE / kT)
            upd = [jnp.zeros_like(dE)] * len(self.terms)
            for ti, d in dEs:
                upd[ti] = d
            E = E + jnp.where(acc, jnp.stack(upd), 0.)
            for ti, d in dEs:
                if callable(new[ti]):
                    cache[ti] = new[ti](acc)
                elif new[ti] is not None:
                    cache[ti] = jax.tree.map(
                        lambda a, b: jnp.where(acc, a, b), new[ti], cache[ti])
        else:
            st2 = mv.apply(st) if mv is not None else dict(st, nuis=nuis)
            En = jnp.stack([t.full(st2) for t in self.terms])
            dE = jnp.sum(En - E, axis=0)
            acc = u[2] < jnp.exp(-dE / kT)
            E = jnp.where(acc, En, E)
        return E, tuple(cache), acc

    def _branch(self, G):
        tab = {k: jnp.asarray(v) for k, v in G.tab.items()}
        ttabs = {ti: {k: jnp.asarray(v) for k, v in tt.items()}
                 for ti, tt in G.term_tabs.items()}
        s = self.s

        if G.kind == 'normal':
            lo, hi = jnp.asarray(s.nuis_lower), jnp.asarray(s.nuis_upper)

            def branch(core, j, u, z, kT):
                k = tab['nuis'][j]
                x = core['nuis'][k]
                xn = jnp.clip(x + tab['sigma'][j] * z[0], lo[k], hi[k])
                E, cache, acc = self._decide(
                    G, ttabs, j, core, u, kT, nuis=core['nuis'].at[k].set(xn))
                return dict(core, nuis=core['nuis'].at[k].set(
                    jnp.where(acc, xn, x)), E=E, cache=cache), acc
            return branch

        def branch(core, j, u, z, kT):
            t = {k: v[j] for k, v in tab.items()}
            X, quat = core['X'], core['quat']
            rows = t['rows']
            Xo = jnp.take(X, rows, axis=1, mode='clip')        # (3, M, C)
            v = _ball(z, u, t['step'])
            bodies = qo = qn = None
            if G.kind == 'ball':
                Xn = Xo + v[:, None]
            else:
                bodies = t['bodies']
                qo = jnp.take(quat, bodies, axis=1, mode='clip')  # (4, kb, C)
                qr = _axis_angle(z, u, t['rot'])
                qn = _normalize(_qmul(qr[:, None], qo))
                tb = Xo[:, t['cslot']]                         # (3, kb, C)
                if G.kind == 'rb':
                    tn = tb + v[:, None]
                else:
                    cw = t['cw']
                    c = jnp.sum(cw[None, :, None] * Xo, axis=1) / jnp.sum(cw)
                    Rr = quat_to_rot(qr, jnp)

                    def tt(y):
                        return _rotate(Rr, y - c[:, None]) + c[:, None] \
                            + v[:, None]
                    tn = tt(tb)
                R = quat_to_rot(qn, jnp)                       # (3,3,kb,C)
                if G.kind == 'rb':      # one body: broadcast, no gather
                    tm = tn
                else:
                    b = jnp.maximum(t['bslot'], 0)
                    R, tm = R[:, :, b], tn[:, b]               # (3,3,M,C)
                loc = t['loc'][..., None]                      # (3, M, 1)
                Xn = R[:, 0] * loc[0] + R[:, 1] * loc[1] + R[:, 2] * loc[2] \
                    + tm
                if G.kind == 'transform':
                    Xn = jnp.where((t['bslot'] >= 0)[None, :, None], Xn,
                                   tt(Xo))
            mv = Move(core, rows, Xo, Xn, bodies, qn)
            E, cache, acc = self._decide(G, ttabs, j, core, u, kT, mv=mv)
            X = X.at[:, rows].set(jnp.where(acc, Xn, Xo), mode='drop')
            if bodies is not None:
                quat = quat.at[:, bodies].set(jnp.where(acc, qn, qo),
                                              mode='drop')
            return dict(core, X=X, quat=quat, E=E, cache=cache), acc
        return branch

    def draws(self, key, sweep, dtype):
        """U (n_movers, 3, C), Z (n_movers, 6, C) for one sweep"""
        n = len(self.s.movers)

        def one(k):
            k = jax.random.fold_in(k, sweep)
            return (jax.random.uniform(k, (n, 3), dtype),
                    jax.random.normal(jax.random.fold_in(k, 1), (n, 6), dtype))
        U, Z = jax.vmap(one)(key)
        return jnp.moveaxis(U, 0, -1), jnp.moveaxis(Z, 0, -1)

    def sweep(self, st):
        """one sweep; returns the state and the decisions (n_movers, C)"""
        U, Z = self.draws(st['key'], st['sweep'], st['X'].dtype)
        kT = st['kT']
        core = {k: st[k] for k in ('X', 'quat', 'nuis', 'E', 'cache')}
        branches = self._branches

        def step(core, xs):
            g, j, u, z = xs
            return jax.lax.switch(g, branches, core, j, u, z, kT)
        core, acc = jax.lax.scan(step, core, (jnp.asarray(self.gid),
                                              jnp.asarray(self.jid), U, Z))
        return dict(st, **core, sweep=st['sweep'] + 1,
                    n_acc=st['n_acc'] + acc), acc

    def run(self, st, n_sweeps, record=False):
        """n_sweeps sweeps; with record, also the decisions
           (n_sweeps, n_movers, C)"""
        if record:
            return jax.lax.scan(lambda st, _: self.sweep(st), st, None,
                                length=n_sweeps)
        return jax.lax.fori_loop(0, n_sweeps, lambda _, st: self.sweep(st)[0],
                                 st)

    # ---------------- state ----------------------------------------------
    @staticmethod
    def dtype():
        return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    def full(self, st):
        """energies (T, C) and caches from scratch"""
        core = dict(X=st['X'], quat=st['quat'], nuis=st['nuis'])
        return (jnp.stack([t.full(core) for t in self.terms]),
                tuple(t.cache(core) for t in self.terms))

    def init(self, C, seed=0, kT=None, dofs=None):
        s, dt = self.s, self.dtype()
        d = s.initial_dofs(C) if dofs is None else dofs
        d = {k: jnp.asarray(v, dt) for k, v in d.items()}
        st = dict(X=s.positions(d, jnp), quat=d['quat'], nuis=d['nuis'])
        E, cache = jax.jit(self.full)(st)
        key = jax.vmap(lambda c: jax.random.fold_in(jax.random.key(seed), c))(
            jnp.arange(C))
        kT = s.kT if kT is None else kT
        return dict(st, E=E, cache=cache,
                    kT=jnp.broadcast_to(jnp.asarray(kT, dt), (C,)), key=key,
                    sweep=jnp.zeros((), jnp.int32),
                    n_acc=jnp.zeros((len(s.movers), C), jnp.int32))

    def resync(self, st):
        """recompute energies and caches (tracked ones drift)"""
        E, cache = self.full(st)
        return dict(st, E=E, cache=cache)

    def total(self, st):
        return jnp.sum(st['E'], axis=0) + self.s.constant

    def dofs(self, st):
        """chain-last degrees of freedom (numpy), as System.initial_dofs"""
        X = np.asarray(st['X'])
        return dict(quat=np.asarray(st['quat']), trans=X[:, self.center_row],
                    bead=X[:, self.bead_row], nuis=np.asarray(st['nuis']))
