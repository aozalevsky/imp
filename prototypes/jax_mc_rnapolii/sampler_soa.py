"""Struct-of-arrays version of the fused sampler, for the GPU.

Chains are the last (fastest-varying) axis of every state array, so the
values of consecutive chains are consecutive in memory: the many small
per-step loads and stores (a bead's position, its cached energy, the
energy, the random numbers, the acceptance counters) coalesce, instead of
touching one memory sector per chain as with jax.vmap over chain-major
arrays. Same tables, moves and energies as sampler.TutSampler with
split=True, cache=True and the static EV grid. Differences:

* one random key for all chains (state.rkey), normal and uniform numbers
  drawn as (n_movers, k, chains) per sweep;
* rotations (rigid-body members, EM covariances) are elementwise float32
  products, never dot_general (which can run as TF32 on the GPU,
  salilab/imp#1129);
* make_run donates the state, so it is updated in place.

    smp = TutSamplerSoA(s)
    st = smp.initial_state(32768, seed=3)
    run = smp.make_run(10)
    st = run(st)
"""
import copy
from dataclasses import dataclass

import numpy as np

import scoring as T
from sampler import TutSampler, _dist

_STATE = None


def _state_cls():
    global _STATE
    if _STATE is None:
        import jax

        @jax.tree_util.register_dataclass
        @dataclass
        class SoAState:
            X: jax.Array          # (3, n_dyn, C) dynamic row positions
            quat: jax.Array       # (nb, 4, C)
            trans: jax.Array      # (nb, 3, C)
            energy: jax.Array     # (C,) total score
            em: jax.Array         # (C,) current EM term (weighted)
            es_bead: jax.Array    # (nbead, C) bead vs static EV energy
            es_body: jax.Array    # (nb, C) body vs static EV energy
            kT: jax.Array         # (C,)
            rkey: jax.Array       # one key for all chains
            accepted: jax.Array   # (n_movers, C) accepted counts
        _STATE = SoAState
    return _STATE


def quat_mul_cf(a, b, xp):
    """quaternion product, component-first (4, ...) arrays"""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return xp.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], 0)


def quat_to_rot_cf(q, xp):
    """T.quat_to_rot for component-first (4, ...) -> (3, 3, ...)"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    tx, ty, tz = 2. * x, 2. * y, 2. * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return xp.stack([
        xp.stack([1. - (tyy + tzz), txy - twz, txz + twy], 0),
        xp.stack([txy + twz, 1. - (txx + tzz), tyz - twx], 0),
        xp.stack([txz - twy, tyz + twx, 1. - (txx + tyy)], 0)], 0)


def rotate_cf(R, local, t):
    """R (3, 3, C) applied to local points (3, m), plus t (3, C), as
       elementwise products: (3, m, C)"""
    return (R[:, 0, None] * local[0][None, :, None]
            + R[:, 1, None] * local[1][None, :, None]
            + R[:, 2, None] * local[2][None, :, None]) + t[:, None]


def build_grid(P, R, rq, h):
    """Cell -> points within reach of any query of radius <= rq in the
       cell, padded with a dummy point far away (radius 0). P (n, 3)."""
    pad = rq + R.max() + h
    lo = P.min(0) - pad
    n = np.ceil((P.max(0) + pad - lo) / h).astype(np.int64)
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'),
                   -1).reshape(-1, 3)
    cen = lo + h * (ijk + 0.5)
    dd = np.maximum(np.abs(cen[:, None, :] - P[None]) - h / 2, 0.)
    # 0.01 A margin: a float32 cell index may be off by one at a face
    cand = np.sqrt((dd ** 2).sum(-1)) < rq + R[None] + 0.01
    K = int(cand.sum(1).max())
    table = np.full((len(cen) + 1, K), len(P), np.int32)   # last: outside
    for c in np.nonzero(cand.any(1))[0]:
        j = np.nonzero(cand[c])[0]
        table[c, :len(j)] = j
    return dict(lo=lo, h=float(h), n=n, table=table, K=K,
                P=np.vstack([P, [1e4, 1e4, 1e4]]).T, R=np.r_[R, 0.])


def grid_ev(g, xq, rq, soft, dt):
    """sum over the grid's points of soft(|xq - P|, rq + R) for query rows
       xq (3, m, C) with radii rq (m,) -> (m, C)"""
    import jax.numpy as jnp
    n = g['n']
    idx = jnp.floor((xq - jnp.asarray(g['lo'], dt)[:, None, None])
                    / g['h'])
    idx = jnp.clip(idx, -1, jnp.asarray(n, dt)[:, None, None]) \
        .astype(jnp.int32)
    inside = jnp.all((idx >= 0) & (idx < jnp.asarray(n, jnp.int32)
                                   [:, None, None]), axis=0)
    flat = (idx[0] * int(n[1]) + idx[1]) * int(n[2]) + idx[2]
    flat = jnp.where(inside, flat, len(g['table']) - 1)
    cand = jnp.asarray(g['table'])[flat]                      # (m, C, K)
    Pc = jnp.asarray(g['P'], dt)[:, cand]                     # (3, m, C, K)
    Rc = jnp.asarray(g['R'], dt)[cand]
    d = _dist(Pc, xq[..., None], jnp)
    return jnp.sum(soft(d, jnp.asarray(rq, dt)[:, None, None] + Rc),
                   axis=-1)


class TutSamplerSoA(TutSampler):
    def __init__(self, s, dtype='float32', grid_h=6.0, full_dense=False,
                 body_grid=False, body_grid_h=4.0):
        """full_dense: score the dynamic-static EV pairs of the full
           rescore densely (a check of the grid) instead of with the grid.
           body_grid: score EV against each moving rigid body through a
           grid fixed in the body's own frame (queries are transformed into
           it, so rotations never invalidate it); beads vs beads stay
           dense. Exact; slower than dense for small bodies (the
           tutorial's 18-19 EV rows), faster for large ones (README.md).
           variant() gives the other choice without rebuilding the
           tables; ev_select.py picks one for a run."""
        super().__init__(s, dtype=dtype, split=True, cache=True, grid=True,
                         grid_h=grid_h)
        self.full_dense = full_dense
        self.body_grid = body_grid
        self.body_grid_h = body_grid_h
        if body_grid:
            self._build_body_grids()
        # the static EV rows are scored without labels in the steps and the
        # cache; that is only right if no dynamic row shares their label
        assert self.ev_mask_ds.all()

    def _build_body_grids(self):
        s = self.s
        rq = float(self.r[:self.n_de].max())
        self.body_g = [build_grid(self.local[o1:o1 + n1],
                                  self.r[o1:o1 + n1], rq, self.body_grid_h)
                       for o1, n1 in self.body_ev]
        # labels: rigid members carry their body, beads are unique, so a
        # body's rows meet every bead and every other body's rows
        lab = self.ev_label
        for b, (o1, n1) in enumerate(self.body_ev):
            assert len(set(lab[o1:o1 + n1])) == 1
        o, nb = self.bead_off, s.nbead
        assert o + nb == self.n_de
        assert len(set(lab[o:o + nb])) == nb
        assert not set(lab[o:o + nb]) & set(lab[:o])

    def variant(self, body_grid):
        """This sampler with the per-body grids on or off; both share all
           tables (the grids are built on first use)"""
        if body_grid and not hasattr(self, 'body_g'):
            self._build_body_grids()
        v = copy.copy(self)
        v.body_grid = body_grid
        return v

    # ------------------------------------------------------------------
    def take(self, X, idx):
        """Positions (3, len(idx), C or 1) of rows idx from the dynamic X
           (3, n_dyn, C) and the static rows"""
        import jax.numpy as jnp
        idx = np.asarray(idx)
        d = idx < self.n_dyn
        if d.all():
            return X[:, idx]
        S = self._static(X)
        if not d.any():
            return S[:, idx - self.n_dyn][:, :, None]
        st_ = jnp.broadcast_to(S[:, idx[~d] - self.n_dyn][:, :, None],
                               (3, int((~d).sum()), X.shape[-1]))
        parts = jnp.concatenate([X[:, idx[d]], st_], axis=1)
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
        nx = self.n_dyn
        if quat is None and trans is None and bead is None:
            X = np.broadcast_to(s.positions(q[0], t[0], b[0])
                                [self.order[:nx]].T[:, :, None],
                                (3, nx, n_chains))
        else:
            X = np.stack([s.positions(q[c], t[c], b[c])[self.order[:nx]].T
                          for c in range(n_chains)], -1)
        St = _state_cls()
        st = St(X=jnp.asarray(X, dt),
                quat=jnp.asarray(np.moveaxis(q, 0, -1), dt),
                trans=jnp.asarray(np.moveaxis(t, 0, -1), dt),
                energy=jnp.zeros(n_chains, dt), em=jnp.zeros(n_chains, dt),
                es_bead=jnp.zeros((s.nbead, n_chains), dt),
                es_body=jnp.zeros((s.nb, n_chains), dt),
                kT=jnp.broadcast_to(jnp.asarray(kT, dt), (n_chains,)),
                rkey=jax.random.key(seed),
                accepted=jnp.zeros((self.n_movers, n_chains), jnp.int32))
        return self.resync(st)

    def dofs(self, st):
        """(quat, trans, bead) host arrays, chain-major as in TutSampler"""
        s = self.s
        X = np.asarray(st.X, np.float64)
        bead = X[:, self.bead_off:self.bead_off + s.nbead].transpose(2, 1, 0)
        return (np.moveaxis(np.asarray(st.quat, np.float64), -1, 0),
                np.moveaxis(np.asarray(st.trans, np.float64), -1, 0), bead)

    # ------------------------------------------------------------------
    def _overlap_cf(self, xa, ca, ma, xb, cb, mb):
        """isd::score_gaussian_overlap; x (3, ...), c (3, 3, ...)"""
        import jax.numpy as jnp
        s = ca + cb
        a, b, c = s[0, 0], s[0, 1], s[0, 2]
        d, e, f = s[1, 1], s[1, 2], s[2, 2]
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

    def _em(self, X, quat, moved=None):
        """EM term (C,); quat (nb, 4, C); moved=(b, Xb2) as in TutSampler"""
        import jax.numpy as jnp
        s = self.s
        dt = X.dtype
        xm = jnp.concatenate(
            [moved[1][:, :n] if moved is not None and b == moved[0]
             else X[:, o:o + n] for b, (o, n) in enumerate(self.em_slices)],
            axis=1)                                          # (3, 10, C)
        if s.cov_mode == 'frozen':
            cm = jnp.asarray(np.moveaxis(self.em_cov_frozen, 0, -1),
                             dt)[..., None]                  # (3, 3, 10, 1)
        else:
            R = quat_to_rot_cf(jnp.moveaxis(quat, 1, 0), jnp)  # (3,3,nb,C)
            R = R[:, :, self.em_owner]                       # (3,3,10,C)
            cl = jnp.asarray(np.moveaxis(self.em_cov_local, 0, -1),
                             dt)[..., None]                  # (3,3,10,1)
            # R C R^T elementwise: tmp[i,k] = sum_j R[i,j] C[j,k]
            tmp = jnp.sum(R[:, :, None] * cl[None], axis=1)  # (3,3,10,C)
            cm = jnp.sum(tmp[:, None] * R[None], axis=2)     # (3,3,10,C)
        dx = jnp.asarray(s.d_xyz.T, dt)                      # (3, 50)
        dc = jnp.asarray(np.moveaxis(s.d_cov, 0, -1), dt)    # (3, 3, 50)
        mass_m = jnp.asarray(self.em_mass, dt)
        rad_m = jnp.asarray(self.em_rad, dt)
        md = self._overlap_cf(
            xm[:, :, None], cm[:, :, :, None], mass_m[:, None, None],
            dx[:, None, :, None], dc[:, :, None, :, None],
            jnp.asarray(s.d_mass, dt)[None, :, None])        # (10, 50, C)
        dist = _dist(xm[:, :, None], dx[:, None, :, None], jnp)
        sph = dist - rad_m[:, None, None] \
            - jnp.asarray(s.d_rad, dt)[None, :, None]
        mds = jnp.sum(jnp.where(sph < s.em_md_cut, md, 0.), axis=(0, 1))
        mm = self._overlap_cf(
            xm[:, :, None], cm[:, :, :, None], mass_m[:, None, None],
            xm[:, None, :], cm[:, :, None, :], mass_m[None, :, None])
        dmm = _dist(xm[:, :, None], xm[:, None, :], jnp)
        sphm = dmm - rad_m[:, None, None] - rad_m[None, :, None]
        n = xm.shape[1]
        iu = jnp.asarray(np.triu(np.ones((n, n), bool), 1))[..., None]
        mmx = jnp.sum(jnp.where(iu & (sphm < s.em_mm_cut), mm, 0.),
                      axis=(0, 1))
        return s.em_w * (-jnp.log(2. * mds / (s.em_self_mm + 2. * mmx
                                              + s.em_dd))
                         + s.em_slope * jnp.sum(dist, axis=(0, 1)))

    def _xl_terms(self, dist, xi):
        """-log p of cross-links xi at distances (len(xi), C)"""
        import jax.numpy as jnp
        s = self.s
        dt = dist.dtype
        return T.xl_neglogp(dist, jnp.asarray(s.xl_si[xi], dt)[:, None],
                            jnp.asarray(s.xl_sj[xi], dt)[:, None],
                            jnp.asarray(s.xl_psi[xi], dt)[:, None],
                            s.xl_length, s.xl_slope, jnp)

    def _static_ev_grid(self, xq, rq):
        """sum over static EV rows of soft(|xq - S|, rq + R) for query
           rows xq (3, m, C) with radii rq (m,) -> (m, C)"""
        import jax.numpy as jnp
        dt = xq.dtype
        n = self.g_n
        idx = jnp.floor((xq - jnp.asarray(self.g_lo, dt)[:, None, None])
                        / self.g_h)
        idx = jnp.clip(idx, -1, jnp.asarray(n, dt)[:, None, None]) \
            .astype(jnp.int32)
        inside = jnp.all((idx >= 0) & (idx < jnp.asarray(n, jnp.int32)
                                       [:, None, None]), axis=0)
        flat = (idx[0] * int(n[1]) + idx[1]) * int(n[2]) + idx[2]
        flat = jnp.where(inside, flat, len(self.g_table) - 1)
        cand = jnp.asarray(self.g_table)[flat]                # (m, C, K)
        Sc = jnp.asarray(self.g_S, dt)[:, cand]               # (3, m, C, K)
        Rc = jnp.asarray(self.g_R, dt)[cand]
        d = _dist(Sc, xq[..., None], jnp)
        return jnp.sum(self._soft(d, jnp.asarray(rq, dt)[:, None, None]
                                  + Rc), axis=-1)

    # ------------------------------------------------------------------
    def _full(self, X, quat):
        """All terms for all chains: total (C,), em (C,), terms (4, C)"""
        import jax.numpy as jnp
        s = self.s
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        cp = self.cpairs
        d = _dist(self.take(X, cp[:, 0]), self.take(X, cp[:, 1]), jnp) \
            - (r[cp[:, 0]] + r[cp[:, 1]])[:, None]
        x = jnp.maximum(d - jnp.asarray(s.conn_mean, dt)[:, None], 0.)
        conn = jnp.sum(0.5 * jnp.asarray(s.conn_k, dt)[:, None] * x * x, 0)
        nde, nse = self.n_de, self.n_se
        Xd = X[:, :nde]
        rd = r[:nde]
        dd = _dist(Xd[:, :, None], Xd[:, None, :], jnp)
        if self.full_dense:
            Ss = self._static(X)[:, :nse]
            rs = r[self.se_off:self.se_off + nse]
            ds = _dist(Xd[:, :, None], Ss[:, None, :, None], jnp)
            e_ds = jnp.sum(self._soft(ds, (rd[:, None] + rs[None])
                                      [..., None]), axis=(0, 1))
        else:
            e_ds = jnp.sum(self._static_ev_grid(Xd, self.r[:nde]), 0)
        ev = 0.5 * s.ev_k * (
            jnp.sum(jnp.where(jnp.asarray(self.ev_mask_dd)[..., None],
                              self._soft(dd, (rd[:, None] + rd[None])
                                         [..., None]), 0.), axis=(0, 1))
            + e_ds) + self.ev_static_const
        xr = self.xrows
        dist = _dist(self.take(X, xr[:, 0]), self.take(X, xr[:, 1]), jnp)
        if s.xl_same.any():
            dist = jnp.where(jnp.asarray(s.xl_same)[:, None],
                             (36.0 / 35.0 * r[xr[:, 0]])[:, None], dist)
        xl = jnp.sum(self._xl_terms(dist, np.arange(len(xr))), 0) \
            + s.xl_const
        em = self._em(X, quat)
        return conn + ev + xl + em, em, jnp.stack([conn, ev, xl, em], -1)

    def _static_ev(self, X):
        """(nbead, C) and (nb, C): EV energy (without 0.5 k) vs the static
           EV rows, through the grid as in the steps"""
        import jax.numpy as jnp
        nbd = self.s.nbead
        o = self.bead_off
        eb = self._static_ev_grid(X[:, o:o + nbd], self.r[o:o + nbd])
        ebody = jnp.stack([jnp.sum(self._static_ev_grid(
            X[:, o1:o1 + n1], self.r[o1:o1 + n1]), 0)
            for o1, n1 in self.body_ev])
        return eb, ebody

    def resync(self, st, terms=False):
        """Recompute every chain's energy and the static EV caches from
           scratch; optionally also return the (C, 4) terms"""
        import jax
        if not hasattr(self, '_full_j'):
            self._full_j = jax.jit(lambda X, q: self._full(X, q)
                                   + self._static_ev(X))
        e, em, tr, eb, ebody = self._full_j(st.X, st.quat)
        st.energy, st.em, st.es_bead, st.es_body = e, em, eb, ebody
        return (st, tr) if terms else st

    def _body_ev(self, b, xq, rq, R, t):
        """EV of global query rows xq (3, m, C) (radii rq (m,)) against
           body b's EV rows, the body at rotation R (3, 3, C) and
           translation t (3, C): (m, C)"""
        d = xq - t[:, None]
        # R^T d: local[i] = sum_j R[j, i] d[j]
        xl = (R[0][:, None] * d[0][None] + R[1][:, None] * d[1][None]
              + R[2][:, None] * d[2][None])
        return grid_ev(self.body_g[b], xl, rq, self._soft, xq.dtype)

    # ------------------------------------------------------------------
    def _bead_step(self, i, st, nrm, uni):
        import jax
        import jax.numpy as jnp
        s, p = self.s, self.steps[i]
        X = st.X
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        rk = p['row']
        xo = X[:, rk]                                          # (3, C)
        d = nrm[i, :3]
        xn = xo + d / jnp.sqrt(jnp.sum(d * d, 0)) * (p['radius']
                                                     * uni[i, 0] ** (1. / 3.))
        nde, k = self.n_de, rk - self.bead_off
        if self.body_grid:
            # beads vs beads densely ...
            o, nb = self.bead_off, s.nbead
            Xd = X[:, o:o + nb]
            rr = (r[rk] + r[o:o + nb])[:, None]
            mask = np.arange(nb) != k
            e = jnp.sum(jnp.where(
                jnp.asarray(mask)[:, None],
                self._soft(_dist(Xd, xn[:, None], jnp), rr)
                - self._soft(_dist(Xd, xo[:, None], jnp), rr), 0.), 0)
            # ... and each body through its own grid (new, old position)
            xq = jnp.stack([xn, xo], 1)
            rq = np.full(2, self.r[rk])
            for b in range(s.nb):
                eg = self._body_ev(b, xq, rq,
                                   quat_to_rot_cf(st.quat[b], jnp),
                                   st.trans[b])
                e = e + eg[0] - eg[1]
        else:
            Xd = X[:, :nde]
            rr = (r[rk] + r[:nde])[:, None]
            e = jnp.sum(jnp.where(
                jnp.asarray(p['evmask_dyn'])[:, None],
                self._soft(_dist(Xd, xn[:, None], jnp), rr)
                - self._soft(_dist(Xd, xo[:, None], jnp), rr), 0.), 0)
        es_n = self._static_ev_grid(xn[:, None], r[rk][None])[0]
        es_o = st.es_bead[k]
        de = 0.5 * s.ev_k * (e + es_n - es_o)
        if len(p['ci']):
            ci, co = p['ci'], p['co']
            xc = self.take(X, co)
            rc = (r[rk] + r[co] + jnp.asarray(s.conn_mean[ci], dt))[:, None]
            kk = jnp.asarray(s.conn_k[ci], dt)[:, None]
            bo = jnp.maximum(_dist(xc, xo[:, None], jnp) - rc, 0.)
            bn = jnp.maximum(_dist(xc, xn[:, None], jnp) - rc, 0.)
            de += jnp.sum(0.5 * kk * (bn * bn - bo * bo), 0)
        if len(p['xi']):
            xc = self.take(X, p['xo'])
            de += jnp.sum(self._xl_terms(_dist(xc, xn[:, None], jnp), p['xi'])
                          - self._xl_terms(_dist(xc, xo[:, None], jnp),
                                           p['xi']), 0)
        acc = (de <= 0) | (uni[i, 1] < jnp.exp(-de / st.kT))
        st.X = jax.lax.dynamic_update_slice(
            X, jnp.where(acc, xn, xo)[:, None], (0, rk, 0))
        st.es_bead = jax.lax.dynamic_update_slice(
            st.es_bead, jnp.where(acc, es_n, es_o)[None], (k, 0))
        st.energy = st.energy + jnp.where(acc, de, 0.)
        st.accepted = jax.lax.dynamic_update_slice(
            st.accepted, (st.accepted[i] + acc.astype(jnp.int32))[None],
            (i, 0))
        return st

    def _rb_step(self, i, st, nrm, uni):
        import jax
        import jax.numpy as jnp
        s, p = self.s, self.steps[i]
        X = st.X
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        b = p['body']
        qo, to = st.quat[b], st.trans[b]                       # (4|3, C)
        d = nrm[i, :3]
        tn = to + d / jnp.sqrt(jnp.sum(d * d, 0)) * (
            p['max_trans'] * uni[i, 0] ** (1. / 3.))
        ax = nrm[i, 3:6]
        ax = ax / jnp.sqrt(jnp.sum(ax * ax, 0))
        ang = (2. * uni[i, 2] - 1.) * p['max_rot']
        qr = jnp.concatenate([jnp.cos(ang / 2)[None],
                              jnp.sin(ang / 2)[None] * ax], 0)
        qn = quat_mul_cf(qr, qo, jnp)
        qn = qn / jnp.sqrt(jnp.sum(qn * qn, 0))
        Rn = quat_to_rot_cf(qn, jnp)                           # (3, 3, C)
        (o1, n1), (o2, n2) = self.body_ev[b], self.body_ot[b]
        Xb1 = rotate_cf(Rn, jnp.asarray(p['local_ev'], dt), tn)
        Xb2 = rotate_cf(Rn, jnp.asarray(p['local_ot'], dt), tn)
        nde = self.n_de
        Xo1 = X[:, o1:o1 + n1]
        rm = r[o1:o1 + n1]
        # EV vs the dynamic EV rows of other bodies and beads (the body's
        # own rows are masked, so X serves for old and new) ...
        if self.body_grid:
            # beads through this body's grid, in its new and old frame ...
            o, nb = self.bead_off, s.nbead
            Xd = X[:, o:o + nb]
            rb_ = self.r[o:o + nb]
            e = jnp.sum(
                self._body_ev(b, Xd, rb_, Rn, tn)
                - self._body_ev(b, Xd, rb_, quat_to_rot_cf(qo, jnp), to), 0)
            # ... and this body's rows through the other bodies' grids
            for b2 in range(s.nb):
                if b2 == b:
                    continue
                R2, t2 = quat_to_rot_cf(st.quat[b2], jnp), st.trans[b2]
                e = e + jnp.sum(self._body_ev(b2, Xb1, self.r[o1:o1 + n1],
                                              R2, t2)
                                - self._body_ev(b2, Xo1, self.r[o1:o1 + n1],
                                                R2, t2), 0)
        else:
            rr = (rm[:, None] + r[None, :nde])[..., None]
            Xd = X[:, None, :nde]
            pe = jnp.where(
                jnp.asarray(p['evmask_dyn'])[..., None],
                self._soft(_dist(Xb1[:, :, None], Xd, jnp), rr)
                - self._soft(_dist(Xo1[:, :, None], Xd, jnp), rr), 0.)
            if n1 * nde > 8192:
                # one reduction over a large body's rows x all dynamic rows
                # runs one long serial sum per chain; summing over the
                # dynamic rows first, kept apart by the barrier (XLA would
                # merge the sums), made runs at 177 x 395 rows 2.1-2.7x
                # faster at 4-16k chains (about 4% slower at the
                # tutorial's 19 x 84 rows, hence the size test)
                e = jnp.sum(jax.lax.optimization_barrier(jnp.sum(pe, 1)), 0)
            else:
                e = jnp.sum(pe, (0, 1))
        # ... and the static ones
        es_n = jnp.sum(self._static_ev_grid(Xb1, rm), 0)
        es_o = st.es_body[b]
        de = 0.5 * s.ev_k * (e + es_n - es_o)
        if len(p['xi']):
            xa, xb = p['xa'], p['xb']
            xother = self.take(X, xb)
            xa_new = rotate_cf(Rn, jnp.asarray(p['local_xa'], dt), tn)
            de += jnp.sum(
                self._xl_terms(_dist(xa_new, xother, jnp), p['xi'])
                - self._xl_terms(_dist(X[:, xa], xother, jnp), p['xi']), 0)
        quat_n = jax.lax.dynamic_update_slice(st.quat, qn[None], (b, 0, 0))
        em_n = self._em(X, quat_n, moved=(b, Xb2))
        de += em_n - st.em
        acc = (de <= 0) | (uni[i, 1] < jnp.exp(-de / st.kT))
        # write back only the body's two blocks (in place)
        st.X = jax.lax.dynamic_update_slice(
            jax.lax.dynamic_update_slice(
                X, jnp.where(acc, Xb1, Xo1), (0, o1, 0)),
            jnp.where(acc, Xb2, X[:, o2:o2 + n2]), (0, o2, 0))
        st.quat = jax.lax.dynamic_update_slice(
            st.quat, jnp.where(acc, qn, qo)[None], (b, 0, 0))
        st.trans = jax.lax.dynamic_update_slice(
            st.trans, jnp.where(acc, tn, to)[None], (b, 0, 0))
        st.em = jnp.where(acc, em_n, st.em)
        st.es_body = jax.lax.dynamic_update_slice(
            st.es_body, jnp.where(acc, es_n, es_o)[None], (b, 0))
        st.energy = st.energy + jnp.where(acc, de, 0.)
        st.accepted = jax.lax.dynamic_update_slice(
            st.accepted, (st.accepted[i] + acc.astype(jnp.int32))[None],
            (i, 0))
        return st

    def _sweep(self, st):
        import jax
        key, k1, k2 = jax.random.split(st.rkey, 3)
        n, C = self.n_movers, st.energy.shape[0]
        nrm = jax.random.normal(k1, (n, 6, C), st.X.dtype)
        uni = jax.random.uniform(k2, (n, 3, C), st.X.dtype)
        st.rkey = key
        for i, p in enumerate(self.steps):
            if p['kind'] == 'bead':
                st = self._bead_step(i, st, nrm, uni)
            else:
                st = self._rb_step(i, st, nrm, uni)
        return st

    def make_run(self, sweeps, donate=True):
        """jitted f(state) -> state after `sweeps` sweeps over the movers
           for every chain (the input state is donated: don't reuse it)"""
        import jax

        def run(st):
            return jax.lax.fori_loop(0, sweeps, lambda _, x: self._sweep(x),
                                     st)
        return jax.jit(run, donate_argnums=0 if donate else ())
