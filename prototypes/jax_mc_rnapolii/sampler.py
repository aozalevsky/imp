"""Fused JAX Monte Carlo for the imp_tutorial system, many chains at once.

Same sampling as the tutorial's C++ run (PMI ReplicaExchange macro ->
IMP.pmi.samplers.MonteCarlo): a SerialMover over dof.get_movers() (47
BallMovers on flexible beads, radius 4, and RigidBodyMovers on Rpb4 and
Rpb7, max_trans 4, max_rot 0.3), one mover per MC step in a fixed order,
Metropolis acceptance at the chain's kT. One "frame" is
monte_carlo_steps (10) sweeps over the 49 movers = 490 MC steps.

Energy changes are computed incrementally (only terms that involve the moved
particles):
* bead step: the bead vs all 453 excluded-volume particles (its energy vs
  the 369 static ones is cached per bead, so only the new position is
  scored against them), its connectivity bonds and crosslinks;
* rigid-body step: the body's 18-19 EV particles vs all EV particles of
  other bodies/beads (static part cached per body), its crosslinks, and
  the whole EM restraint
  (10 model x 50 data Gaussians).
Each step of a sweep is specialized for its mover (static indices), and
chains are vectorized with vmap.

Positions of every particle a term reads ("rows"), component-first (3, n):
* dynamic rows, per chain:
      [body 0 EV | body 1 EV | beads | body 0 other | body 1 other]
  (each body's "other" block starts with its EM Gaussians), so every mover
  writes contiguous blocks in place;
* static rows (never move, identical in every chain):
      [static EV | static other]
  as one shared constant (split=True, the GPU default: chains read the same
  cached array) or appended to every chain's X (split=False, the CPU
  default, where XLA handles the broadcast constant less well).
Distances are computed per component (XLA CPU handles (..., 3) reductions
poorly).
"""
from dataclasses import dataclass

import numpy as np

import scoring as T

_STATE = None


def _state_cls():
    global _STATE
    if _STATE is None:
        import jax

        @jax.tree_util.register_dataclass
        @dataclass
        class TutState:
            X: jax.Array          # (3, n_dyn) dynamic row positions
            quat: jax.Array       # (nb, 4)
            trans: jax.Array      # (nb, 3)
            energy: jax.Array     # total score
            em: jax.Array         # current EM term (weighted)
            es_bead: jax.Array    # (nbead,) bead vs static EV energy
            es_body: jax.Array    # (nb,) body vs static EV energy
            kT: jax.Array
            rkey: jax.Array
            accepted: jax.Array   # (n_movers,) accepted counts
        _STATE = TutState
    return _STATE


def quat_mul(a, b, xp):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return xp.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def _dist(a, b, xp):
    """|a - b| for (3, ...) component-first arrays"""
    d0, d1, d2 = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return xp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)


class TutSampler:
    def __init__(self, s, dtype='float32', split=None, cache=None,
                 grid=None, grid_h=6.0):
        """split: keep the static rows in one shared constant and score EV in
           a dynamic and a static pass (True, the GPU default), or keep
           every row in each chain's X with the 453 EV rows contiguous and
           score EV in one pass (False, the CPU default: XLA CPU pays per
           fused kernel). cache (split only): keep each bead's/body's EV
           energy vs the static rows in the state, so a step scores only
           the new position against them."""
        import jax
        import jax.numpy as jnp
        self.s = s
        self.dt = jnp.dtype(dtype)
        gpu = jax.default_backend() != 'cpu'
        self.split = gpu if split is None else split
        self.cache = self.split and (True if cache is None else cache)
        # static EV rows through a precomputed spatial grid (split + cache
        # only): each query scores only the static rows that can touch it
        self.grid = self.cache and (True if grid is None else grid)
        kind, ref = s.row_kind, s.row_ref
        ev = s.ev_rows
        is_ev = np.zeros(s.n, bool)
        is_ev[ev] = True
        nb = s.nb
        em_set = set(s.em_rows.tolist())
        # ---- row layout ----------------------------------------------------
        if not all(is_ev[(kind == 2)]):
            raise NotImplementedError('bead without excluded volume')
        bead_rows = [int(np.nonzero((kind == 2) & (ref == k))[0][0])
                     for k in range(s.nbead)]
        blocks_ev, blocks_ot = [], []
        for b in range(nb):
            blocks_ev.append([i for i in ev if kind[i] == 1 and ref[i] == b])
            mem = [i for i in range(s.n) if kind[i] == 1 and ref[i] == b
                   and not is_ev[i]]
            blocks_ot.append([i for i in mem if i in em_set]
                             + [i for i in mem if i not in em_set])
        static_ev = [i for i in ev if kind[i] == 0]
        static_ot = [i for i in range(s.n) if kind[i] == 0 and not is_ev[i]]
        dyn_ev = sum(blocks_ev, []) + bead_rows
        dyn_ot = sum(blocks_ot, [])
        self.n_de = len(dyn_ev)              # dynamic EV rows
        self.n_se = len(static_ev)           # static EV rows
        self.n_ev = self.n_de + self.n_se
        if self.split:
            order = dyn_ev + dyn_ot + static_ev + static_ot
            ot_base = self.n_de
            self.n_dyn = self.n_de + len(dyn_ot)   # rows held per chain
            self.se_off = self.n_dyn               # first static EV row
        else:
            order = dyn_ev + static_ev + dyn_ot + static_ot
            ot_base = self.n_ev
            self.n_dyn = s.n
            self.se_off = self.n_de
        n_dyn = self.n_dyn
        order = np.array(order)
        assert len(order) == s.n and self.n_ev == len(ev)
        new = np.empty(s.n, np.int64)
        new[order] = np.arange(s.n)
        self.order, self.new = order, new
        off = np.cumsum([0] + [len(x) for x in blocks_ev])
        self.body_ev = [(int(off[b]), len(blocks_ev[b])) for b in range(nb)]
        self.bead_off = int(off[-1])
        off2 = ot_base + np.cumsum([0] + [len(x) for x in blocks_ot])
        self.body_ot = [(int(off2[b]), len(blocks_ot[b])) for b in range(nb)]
        self.r = s.row_r[order]
        self.local = s.row_local[order]
        self.static_xyz = s.row_static[order[n_dyn:]].T.copy()
        # EV rows in "full EV" order: dynamic EV then static EV
        self.ev_full = np.r_[0:self.n_de, self.se_off:self.se_off
                             + self.n_se]
        lab_of_row = dict(zip(s.ev_rows.tolist(), s.ev_label.tolist()))
        self.ev_label = np.array([lab_of_row[i]
                                  for i in order[self.ev_full]])
        # EM Gaussians: contiguous at the start of each body's other block
        em_new = new[s.em_rows]
        self.em_slices = []
        for b in range(nb):
            o, _ = self.body_ot[b]
            sel = np.nonzero(s.em_owner == b)[0]
            assert sorted(em_new[sel]) == list(range(o, o + len(sel)))
            self.em_slices.append((o, len(sel)))
        perm = np.argsort(em_new)
        self.em_cov_local = s.em_cov_local[perm]
        self.em_cov_frozen = s.em_cov_frozen[perm]
        self.em_mass = s.em_mass[perm]
        self.em_rad = s.em_rad[perm]
        self.em_owner = s.em_owner[perm]

        # ---- per-mover static tables --------------------------------------
        cpairs = new[s.conn_pairs]
        xrows = new[s.xl_rows]
        nde = self.n_de
        steps = []
        for kindname, idx, a, b_ in s.movers:
            if kindname == 'bead':
                rk = self.bead_off + idx
                ci = np.nonzero((cpairs == rk).any(1))[0]
                co = np.where(cpairs[ci, 0] == rk, cpairs[ci, 1],
                              cpairs[ci, 0])
                if (co == rk).any():
                    raise NotImplementedError('bond to itself')
                xi = np.nonzero((xrows == rk).any(1)
                                & (xrows[:, 0] != xrows[:, 1]))[0]
                xo = np.where(xrows[xi, 0] == rk, xrows[xi, 1],
                              xrows[xi, 0])
                steps.append(dict(kind='bead', row=int(rk), radius=float(a),
                                  ci=ci, co=co, xi=xi, xo=xo,
                                  evmask_dyn=np.arange(nde) != rk,
                                  evmask_all=np.arange(self.n_ev) != rk))
            else:
                (o1, n1), (o2, n2) = self.body_ev[idx], self.body_ot[idx]
                rows_b = np.r_[o1:o1 + n1, o2:o2 + n2]
                lab_b = self.ev_label[o1:o1 + n1]
                inb = np.isin(xrows, rows_b)
                xi = np.nonzero(inb.any(1) & ~inb.all(1))[0]
                swap = ~inb[xi, 0]
                xa = np.where(swap, xrows[xi, 1], xrows[xi, 0])
                xb = np.where(swap, xrows[xi, 0], xrows[xi, 1])
                assert (lab_b[:, None]
                        != self.ev_label[None, nde:]).all()
                steps.append(dict(
                    kind='rb', body=int(idx), max_trans=float(a),
                    max_rot=float(b_), xi=xi, xa=xa, xb=xb,
                    evmask_dyn=lab_b[:, None] != self.ev_label[None, :nde],
                    evmask_all=lab_b[:, None] != self.ev_label[None, :],
                    local_ev=self.local[o1:o1 + n1].T,
                    local_ot=self.local[o2:o2 + n2].T))
        self.steps = steps
        self.n_movers = len(steps)
        self.cpairs, self.xrows = cpairs, xrows
        # EV pairs with at least one dynamic row (full rescore); the
        # static-static part is a constant
        lab = self.ev_label
        self.ev_mask_dd = (np.triu(np.ones((nde, nde), bool), 1)
                           & (lab[:nde, None] != lab[None, :nde]))
        self.ev_mask_ds = lab[:nde, None] != lab[None, nde:]
        Xs = s.row_static[order[self.ev_full[nde:]]]
        rs = self.r[self.ev_full[nde:]]
        d = np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1))
        x = np.minimum(d - rs[:, None] - rs[None], 0.)
        m_ss = (np.triu(np.ones((len(rs), len(rs)), bool), 1)
                & (lab[nde:, None] != lab[None, nde:]))
        self.ev_static_const = 0.5 * s.ev_k * float((x * x)[m_ss].sum())
        if self.grid:
            self._build_grid(grid_h)
        # rows of each rigid body's cross-links, in body-local coordinates
        for p in self.steps:
            if p['kind'] == 'rb':
                p['local_xa'] = self.local[p['xa']].T
                o1, n1 = self.body_ev[p['body']]
                # pairs inside the body cancel exactly (rigid motion)
                p['evmask_dyn'] = p['evmask_dyn'].copy()
                p['evmask_dyn'][:, o1:o1 + n1] = False

    def _build_grid(self, h):
        """Cell -> static EV rows within reach of any dynamic EV row in
           the cell (padded with a dummy row far away, radius 0)"""
        nse = self.n_se
        S = self.static_xyz[:, :nse].T
        R = self.r[self.se_off:self.se_off + nse]
        rq = float(self.r[:self.n_de].max())
        pad = rq + R.max() + h
        lo = S.min(0) - pad
        n = np.ceil((S.max(0) + pad - lo) / h).astype(np.int64)
        ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n],
                                   indexing='ij'), -1).reshape(-1, 3)
        cen = lo + h * (ijk + 0.5)
        dd = np.maximum(np.abs(cen[:, None, :] - S[None]) - h / 2, 0.)
        # 0.01 A margin: a float32 cell index may be off by one at a face
        cand = np.sqrt((dd ** 2).sum(-1)) < rq + R[None] + 0.01
        K = int(cand.sum(1).max())
        table = np.full((len(cen) + 1, K), nse, np.int32)   # last: outside
        for c in np.nonzero(cand.any(1))[0]:
            j = np.nonzero(cand[c])[0]
            table[c, :len(j)] = j
        self.g_lo, self.g_h, self.g_n = lo, float(h), n
        self.g_table = table
        self.g_S = np.vstack([S, [1e4, 1e4, 1e4]]).T          # (3, nse+1)
        self.g_R = np.r_[R, 0.]
        self.g_K = K

    def _static_ev_grid(self, xq, rq):
        """sum over static EV rows of soft(|xq - S|, rq + R) for query
           rows xq (3, m) with radii rq (m,) -> (m,)"""
        import jax.numpy as jnp
        dt = xq.dtype
        n = self.g_n
        idx = jnp.floor((xq - jnp.asarray(self.g_lo, dt)[:, None])
                        / self.g_h)
        idx = jnp.clip(idx, -1, jnp.asarray(n, dt)[:, None]).astype(
            jnp.int32)
        inside = jnp.all((idx >= 0) & (idx < jnp.asarray(n)[:, None]
                                       .astype(jnp.int32)), axis=0)
        flat = (idx[0] * int(n[1]) + idx[1]) * int(n[2]) + idx[2]
        flat = jnp.where(inside, flat, len(self.g_table) - 1)
        cand = jnp.asarray(self.g_table)[flat]                  # (m, K)
        Sc = jnp.asarray(self.g_S, dt)[:, cand]                 # (3, m, K)
        Rc = jnp.asarray(self.g_R, dt)[cand]
        d = _dist(Sc, xq[:, :, None], jnp)
        return jnp.sum(self._soft(d, rq[:, None] + Rc), axis=1)

    # ------------------------------------------------------------------
    def _static(self, X):
        """Positions of the static rows (split mode: the shared constant)"""
        import jax.numpy as jnp
        assert self.split
        return jnp.asarray(self.static_xyz, self.dt)

    def take(self, X, idx):
        """Positions (3, len(idx)) of rows idx (static numpy array of row
           ids in the new order) from the dynamic X and the static rows"""
        import jax.numpy as jnp
        idx = np.asarray(idx)
        d = idx < self.n_dyn
        if d.all() or not self.split:
            return X[:, idx]
        S = self._static(X)
        if not d.any():
            return S[:, idx - self.n_dyn]
        parts = jnp.concatenate([X[:, idx[d]], S[:, idx[~d] - self.n_dyn]],
                                axis=1)
        pos = np.concatenate([np.nonzero(d)[0], np.nonzero(~d)[0]])
        return parts[:, np.argsort(pos)]

    def initial_state(self, n_chains, kT=1.0, seed=0, quat=None, trans=None,
                      bead=None):
        import jax
        import jax.numpy as jnp
        s, dt = self.s, self.dt
        q = np.broadcast_to(s.quat0 if quat is None else quat,
                            (n_chains, s.nb, 4))
        t = np.broadcast_to(s.trans0 if trans is None else trans,
                            (n_chains, s.nb, 3))
        b = np.broadcast_to(s.bead0 if bead is None else bead,
                            (n_chains, s.nbead, 3))
        nx = self.n_dyn if self.split else s.n
        if quat is None and trans is None and bead is None:
            X = np.broadcast_to(s.positions(q[0], t[0], b[0])
                                [self.order[:nx]].T, (n_chains, 3, nx))
        else:
            X = np.stack([s.positions(q[c], t[c], b[c])[self.order[:nx]].T
                          for c in range(n_chains)])
        St = _state_cls()
        st = St(X=jnp.asarray(X, dt), quat=jnp.asarray(q, dt),
                trans=jnp.asarray(t, dt),
                energy=jnp.zeros(n_chains, dt), em=jnp.zeros(n_chains, dt),
                es_bead=jnp.zeros((n_chains, s.nbead), dt),
                es_body=jnp.zeros((n_chains, s.nb), dt),
                kT=jnp.broadcast_to(jnp.asarray(kT, dt), (n_chains,)),
                rkey=jax.random.split(jax.random.key(seed), n_chains),
                accepted=jnp.zeros((n_chains, self.n_movers), jnp.int32))
        return self.resync(st)

    # ------------------------------------------------------------------
    def _full(self, X, quat):
        """All terms from the dynamic rows (one chain)"""
        import jax.numpy as jnp
        s = self.s
        dt = X.dtype
        Xf = (jnp.concatenate([X, self._static(X)], axis=1) if self.split
              else X)
        r = jnp.asarray(self.r, dt)
        cp = self.cpairs
        d = _dist(Xf[:, cp[:, 0]], Xf[:, cp[:, 1]], jnp) - r[cp[:, 0]] \
            - r[cp[:, 1]]
        x = jnp.maximum(d - jnp.asarray(s.conn_mean, dt), 0.)
        conn = jnp.sum(0.5 * jnp.asarray(s.conn_k, dt) * x * x)
        ef = self.ev_full
        nde = self.n_de
        Xe = Xf[:, ef]
        re = r[ef]
        dd = _dist(Xe[:, :nde, None], Xe[:, None, :nde], jnp)
        ds = _dist(Xe[:, :nde, None], Xe[:, None, nde:], jnp)
        ev = 0.5 * s.ev_k * (
            jnp.sum(jnp.where(jnp.asarray(self.ev_mask_dd), self._soft(
                dd, re[:nde, None] + re[None, :nde]), 0.))
            + jnp.sum(jnp.where(jnp.asarray(self.ev_mask_ds), self._soft(
                ds, re[:nde, None] + re[None, nde:]), 0.))) \
            + self.ev_static_const
        xr = self.xrows
        dist = _dist(Xf[:, xr[:, 0]], Xf[:, xr[:, 1]], jnp)
        if s.xl_same.any():
            dist = jnp.where(jnp.asarray(s.xl_same), 36.0 / 35.0
                             * r[xr[:, 0]], dist)
        xl = jnp.sum(self._xl_terms(dist, np.arange(len(xr)))) + s.xl_const
        em = self._em(X, quat)
        return conn + ev + xl + em, em, jnp.stack([conn, ev, xl, em])

    def _xl_terms(self, dist, xi):
        import jax.numpy as jnp
        s = self.s
        dt = dist.dtype
        return T.xl_neglogp(dist, jnp.asarray(s.xl_si[xi], dt),
                            jnp.asarray(s.xl_sj[xi], dt),
                            jnp.asarray(s.xl_psi[xi], dt), s.xl_length,
                            s.xl_slope, jnp)

    def _em(self, X, quat, moved=None):
        """EM term from the model Gaussians in the bodies' blocks of X;
           moved=(b, Xb2): body b's rows from its new 'other' block"""
        import jax.numpy as jnp
        s = self.s
        dt = X.dtype
        xm = jnp.concatenate(
            [moved[1][:, :n] if moved is not None and b == moved[0]
             else X[:, o:o + n] for b, (o, n) in enumerate(self.em_slices)],
            axis=1)
        if s.cov_mode == 'frozen':
            cm = jnp.asarray(self.em_cov_frozen, dt)
        else:
            R = T.quat_to_rot(quat, jnp)[self.em_owner]
            cm = jnp.einsum('nij,njk,nlk->nil', R,
                            jnp.asarray(self.em_cov_local, dt), R)
        dx = jnp.asarray(s.d_xyz.T, dt)
        mass_m = jnp.asarray(self.em_mass, dt)
        rad_m = jnp.asarray(self.em_rad, dt)
        md = self._overlap(xm[:, :, None], cm[:, None], mass_m[:, None],
                           dx[:, None, :], jnp.asarray(s.d_cov, dt)[None],
                           jnp.asarray(s.d_mass, dt)[None])
        dist = _dist(xm[:, :, None], dx[:, None, :], jnp)
        sph = dist - rad_m[:, None] - jnp.asarray(s.d_rad, dt)[None]
        mds = jnp.sum(jnp.where(sph < s.em_md_cut, md, 0.))
        mm = self._overlap(xm[:, :, None], cm[:, None], mass_m[:, None],
                           xm[:, None, :], cm[None], mass_m[None])
        dmm = _dist(xm[:, :, None], xm[:, None, :], jnp)
        sphm = dmm - rad_m[:, None] - rad_m[None]
        n = xm.shape[1]
        iu = jnp.asarray(np.triu(np.ones((n, n), bool), 1))
        mmx = jnp.sum(jnp.where(iu & (sphm < s.em_mm_cut), mm, 0.))
        return s.em_w * (-jnp.log(2. * mds / (s.em_self_mm + 2. * mmx
                                              + s.em_dd))
                         + s.em_slope * jnp.sum(dist))

    @staticmethod
    def _overlap(xa, ca, ma, xb, cb, mb):
        """isd::score_gaussian_overlap; x component-first (3, ...),
           c (..., 3, 3) broadcastable"""
        import jax.numpy as jnp
        s = ca + cb
        a, b, c = s[..., 0, 0], s[..., 0, 1], s[..., 0, 2]
        d, e, f = s[..., 1, 1], s[..., 1, 2], s[..., 2, 2]
        c00 = d * f - e * e
        c01 = c * e - b * f
        c02 = b * e - c * d
        det = a * c00 + b * c01 + c * c02
        c11 = a * f - c * c
        c12 = b * c - a * e
        c22 = a * d - b * b
        vx, vy, vz = xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]
        quad = (c00 * vx * vx + c11 * vy * vy + c22 * vz * vz
                + 2. * (c01 * vx * vy + c02 * vx * vz + c12 * vy * vz)) / det
        return ma * mb * T._GAUSS_NORM / jnp.sqrt(det) \
            * jnp.exp(-0.5 * quad)

    def _static_ev(self, X):
        """Per bead and per body: EV energy (without 0.5 k) vs the static
           EV rows (one chain)"""
        import jax.numpy as jnp
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        nse = self.n_se
        Ss = self._static(X)[:, None, :nse]
        rs = r[self.se_off:self.se_off + nse]
        nbd = self.s.nbead
        o = self.bead_off
        eb = jnp.sum(self._soft(_dist(X[:, o:o + nbd, None], Ss, jnp),
                                r[o:o + nbd, None] + rs[None]), axis=1)
        ebody = jnp.stack([jnp.sum(self._soft(
            _dist(X[:, o1:o1 + n1, None], Ss, jnp),
            r[o1:o1 + n1, None] + rs[None])) for o1, n1 in self.body_ev])
        return eb, ebody

    def resync(self, st, terms=False):
        """Recompute every chain's energy from scratch (removes float32
           drift of the incremental energy) and the cached static EV
           energies; optionally also return the (chains, 4) terms:
           connectivity, EV, crosslinks, EM"""
        import jax
        import jax.numpy as jnp
        if not hasattr(self, '_full_v'):
            def f(xq):
                X, quat = xq
                if not self.cache:
                    z = jnp.zeros((), X.dtype)
                    return self._full(X, quat) + (z, z)
                return self._full(X, quat) + self._static_ev(X)
            # batches of chains bound the memory of the dense terms
            self._full_v = jax.jit(lambda X, q: jax.lax.map(
                f, (X, q), batch_size=min(256, X.shape[0])))
        e, em, tr, eb, ebody = self._full_v(st.X, st.quat)
        st.energy, st.em = e, em
        if self.cache:
            st.es_bead, st.es_body = eb, ebody
        return (st, tr) if terms else st

    # ------------------------------------------------------------------
    @staticmethod
    def _soft(d, rr):
        import jax.numpy as jnp
        x = jnp.minimum(d - rr, 0.)
        return x * x

    def _bead_step(self, i, st, nrm, uni):
        import jax
        import jax.numpy as jnp
        s, p = self.s, self.steps[i]
        X = st.X
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        rk = p['row']
        xo = X[:, rk]
        d = nrm[i, :3]
        xn = xo + d / jnp.sqrt(jnp.sum(d * d)) * (p['radius']
                                                  * uni[i, 0] ** (1. / 3.))
        nde, nse, k = self.n_de, self.n_se, rk - self.bead_off
        if not self.split:
            # EV: one pass over the 453 contiguous EV rows
            ne = self.n_ev
            Xe = X[:, :ne]
            rr = r[rk] + r[:ne]
            e = jnp.where(jnp.asarray(p['evmask_all']),
                          self._soft(_dist(Xe, xn[:, None], jnp), rr)
                          - self._soft(_dist(Xe, xo[:, None], jnp), rr), 0.)
            de = 0.5 * s.ev_k * jnp.sum(e)
        else:
            # EV vs the dynamic EV rows (per chain) and the static ones
            # (shared)
            Xd = X[:, :nde]
            rr = r[rk] + r[:nde]
            e = jnp.where(jnp.asarray(p['evmask_dyn']),
                          self._soft(_dist(Xd, xn[:, None], jnp), rr)
                          - self._soft(_dist(Xd, xo[:, None], jnp), rr), 0.)
            if self.grid:
                es_n = self._static_ev_grid(xn[:, None], r[rk][None])[0]
            else:
                Ss = self._static(X)[:, :nse]
                rs = r[rk] + r[self.se_off:self.se_off + nse]
                es_n = jnp.sum(self._soft(_dist(Ss, xn[:, None], jnp), rs))
            es_o = (st.es_bead[k] if self.cache else
                    jnp.sum(self._soft(_dist(Ss, xo[:, None], jnp), rs)))
            de = 0.5 * s.ev_k * (jnp.sum(e) + es_n - es_o)
        if len(p['ci']):
            ci, co = p['ci'], p['co']
            xc = self.take(X, co)
            rc = r[rk] + r[co] + jnp.asarray(s.conn_mean[ci], dt)
            kk = jnp.asarray(s.conn_k[ci], dt)
            bo = jnp.maximum(_dist(xc, xo[:, None], jnp) - rc, 0.)
            bn = jnp.maximum(_dist(xc, xn[:, None], jnp) - rc, 0.)
            de += jnp.sum(0.5 * kk * (bn * bn - bo * bo))
        if len(p['xi']):
            xc = self.take(X, p['xo'])
            de += jnp.sum(self._xl_terms(_dist(xc, xn[:, None], jnp), p['xi'])
                          - self._xl_terms(_dist(xc, xo[:, None], jnp),
                                           p['xi']))
        acc = (de <= 0) | (uni[i, 1] < jnp.exp(-de / st.kT))
        st.X = jax.lax.dynamic_update_slice(
            X, jnp.where(acc, xn, xo)[:, None], (0, rk))
        if self.cache:
            st.es_bead = st.es_bead.at[k].set(jnp.where(acc, es_n,
                                                        st.es_bead[k]))
        st.energy = st.energy + jnp.where(acc, de, 0.)
        st.accepted = st.accepted.at[i].add(acc.astype(jnp.int32))
        return st

    def _rb_step(self, i, st, nrm, uni):
        import jax
        import jax.numpy as jnp
        s, p = self.s, self.steps[i]
        X = st.X
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        b = p['body']
        qo, to = st.quat[b], st.trans[b]
        d = nrm[i, :3]
        tn = to + d / jnp.sqrt(jnp.sum(d * d)) * (p['max_trans']
                                                  * uni[i, 0] ** (1. / 3.))
        ax = nrm[i, 3:6]
        ax = ax / jnp.sqrt(jnp.sum(ax * ax))
        ang = (2. * uni[i, 2] - 1.) * p['max_rot']
        qr = jnp.concatenate([jnp.cos(ang / 2)[None], jnp.sin(ang / 2) * ax])
        qn = quat_mul(qr, qo, jnp)
        qn = qn / jnp.sqrt(jnp.sum(qn * qn))
        Rn = T.quat_to_rot(qn, jnp)
        (o1, n1), (o2, n2) = self.body_ev[b], self.body_ot[b]
        Xb1 = Rn @ jnp.asarray(p['local_ev'], dt) + tn[:, None]
        Xb2 = Rn @ jnp.asarray(p['local_ot'], dt) + tn[:, None]
        nde, nse = self.n_de, self.n_se
        Xo1 = X[:, o1:o1 + n1]
        rm = r[o1:o1 + n1, None]
        if not self.split:
            # EV: the body's EV rows vs all 453 (other labels), one pass
            Xn = jax.lax.dynamic_update_slice(X, Xb1, (0, o1))
            Xn = jax.lax.dynamic_update_slice(Xn, Xb2, (0, o2))
            ne = self.n_ev
            rr = rm + r[None, :ne]
            e = jnp.where(jnp.asarray(p['evmask_all']),
                          self._soft(_dist(Xb1[:, :, None],
                                           Xn[:, None, :ne], jnp), rr)
                          - self._soft(_dist(Xo1[:, :, None],
                                             X[:, None, :ne], jnp), rr), 0.)
            de = 0.5 * s.ev_k * jnp.sum(e)
        else:
            # EV vs the dynamic EV rows of other bodies and beads (the
            # body's own rows are masked, so X serves for old and new) ...
            rr = rm + r[None, :nde]
            Xd = X[:, None, :nde]
            e = jnp.where(jnp.asarray(p['evmask_dyn']),
                          self._soft(_dist(Xb1[:, :, None], Xd, jnp), rr)
                          - self._soft(_dist(Xo1[:, :, None], Xd, jnp), rr),
                          0.)
            # ... and the static ones (all other labels)
            if self.grid:
                es_n = jnp.sum(self._static_ev_grid(Xb1, r[o1:o1 + n1]))
            else:
                Ss = self._static(X)[:, None, :nse]
                rs = rm + r[None, self.se_off:self.se_off + nse]
                es_n = jnp.sum(self._soft(_dist(Xb1[:, :, None], Ss, jnp),
                                          rs))
            es_o = (st.es_body[b] if self.cache else
                    jnp.sum(self._soft(_dist(Xo1[:, :, None], Ss, jnp),
                                       rs)))
            de = 0.5 * s.ev_k * (jnp.sum(e) + es_n - es_o)
        if len(p['xi']):
            xa, xb = p['xa'], p['xb']
            xother = self.take(X, xb)
            xa_new = Rn @ jnp.asarray(p['local_xa'], dt) + tn[:, None]
            de += jnp.sum(
                self._xl_terms(_dist(xa_new, xother, jnp), p['xi'])
                - self._xl_terms(_dist(X[:, xa], xother, jnp), p['xi']))
        quat_n = st.quat.at[b].set(qn)
        em_n = self._em(X, quat_n, moved=(b, Xb2))
        de += em_n - st.em
        acc = (de <= 0) | (uni[i, 1] < jnp.exp(-de / st.kT))
        # write back only the body's two blocks (in place)
        st.X = jax.lax.dynamic_update_slice(
            jax.lax.dynamic_update_slice(
                X, jnp.where(acc, Xb1, Xo1), (0, o1)),
            jnp.where(acc, Xb2, X[:, o2:o2 + n2]), (0, o2))
        st.quat = jnp.where(acc, quat_n, st.quat)
        st.trans = st.trans.at[b].set(jnp.where(acc, tn, to))
        st.em = jnp.where(acc, em_n, st.em)
        if self.cache:
            st.es_body = st.es_body.at[b].set(jnp.where(acc, es_n,
                                                        st.es_body[b]))
        st.energy = st.energy + jnp.where(acc, de, 0.)
        st.accepted = st.accepted.at[i].add(acc.astype(jnp.int32))
        return st

    def _sweep(self, st):
        import jax
        key, k1, k2 = jax.random.split(st.rkey, 3)
        n = self.n_movers
        nrm = jax.random.normal(k1, (n, 6), st.X.dtype)
        uni = jax.random.uniform(k2, (n, 3), st.X.dtype)
        st.rkey = key
        for i, p in enumerate(self.steps):
            if p['kind'] == 'bead':
                st = self._bead_step(i, st, nrm, uni)
            else:
                st = self._rb_step(i, st, nrm, uni)
        return st

    def make_run(self, sweeps):
        """jitted f(state) -> state after `sweeps` sweeps over the movers
           (sweeps * n_movers MC steps) for every chain"""
        import jax

        def one(st):
            return jax.lax.fori_loop(0, sweeps, lambda _, x: self._sweep(x),
                                     st)
        return jax.jit(jax.vmap(one))

    # ------------------------------------------------------------------
    def dofs(self, st):
        """(quat, trans, bead) host arrays for all chains"""
        s = self.s
        X = np.asarray(st.X, np.float64)
        bead = X[:, :, self.bead_off:self.bead_off + s.nbead] \
            .transpose(0, 2, 1)
        return (np.asarray(st.quat, np.float64),
                np.asarray(st.trans, np.float64), bead)
