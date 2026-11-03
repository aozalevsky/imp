"""Few-bodies-and-beads template: SerialMover systems with few rigid bodies
and many flexible beads (RNA Polymerase II).

Composes recipes R1-R7 (and works with R10-R11 in `jax_mc.run`) on top of
`jax_mc.system`, `jax_mc.terms`, and the `jax_mc.engine` random-number
contract:

  R1  Static/movable split; static EV scored through a precomputed spatial
      grid (`build_grid`, `grid_ev`) with a `uint16` cell table and packed
      `(x, y, z, r)` point table; static-vs-dynamic EV cached per bead
      (`es_bead`) and per rigid body (`es_body`).
  R2  Unrolled sweep in `SerialMover` order with static row offsets, in-place
      `dynamic_update_slice_in_dim` behind `optimization_barrier`, and a
      flat tuple loop carry (`0` copies of `X` in the compiled loop).
  R3  Blocked bead steps (`_bead_block_batched` for `C <= 4,096`,
      `_bead_block_chained` for `C > 4,096`) with exact intra-block EV,
      connectivity, and cross-link corrections.
  R4  Per-body EM cache (`_em_parts`, `_em_rb`): caches per-body model-data
      overlap and distance sums; when `cov_mode == 'rotate'`, intra-body
      model-model overlap is invariant under rigid-body moves and is computed
      once outside the sweep loop.
  R5  Closed-form 6-component covariances (`rot_cov6`, `overlap_cov6`) and
      elementwise 3x3 rotations (`rotate_cf`), avoiding `dot_general` / TF32.
  R6  Sweep-wide precomputation of bead proposals' static EV grid lookups
      (`es_n_all`) from the sweep-entry positions `X0` when no bead moves
      twice per sweep.
  R7  Contract-preserving sliced RNG draws (`sliced_draws`): generates
      per-chain draws matching `engine.draws` bit-for-bit, slicing out unused
      rotation columns before the chain-last transpose.

Check exactness (Level D) against `jax_mc.engine.Engine`:

  python -m jax_mc.check decisions polii \\
      --sampler templates.few_bodies_beads.sampler:factory \\
      --chains 1024 --ref-chains 256 --kT 1,10
  python -m jax_mc.check decisions polii \\
      --sampler templates.few_bodies_beads.sampler:factory \\
      --chains 8192 --ref-chains 128 --kT 1,10
"""
import copy

import numpy as np
import jax
import jax.numpy as jnp

from jax_mc import engine as E
from jax_mc.system import BEAD, MEMBER, STATIC, Unsupported
from jax_mc.terms import ImpJax, basic, em, geom
from recipes.r1_static_ev_grid import build_grid, dist3, grid_ev, soft_pair
from recipes.r3_blocked_beads import build_exec_segs
from recipes.r5_cov6_elementwise import (
    overlap_cov6, quat_to_rot_eigen9, quat_to_rot_imp, rot_cov6, rotate_cf,
)
from recipes.r7_sliced_rng import sliced_draws


def quat_mul_cf(a, b, xp=jnp):
    """Quaternion product for component-first (4, ...) arrays."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return xp.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], 0)


def factory(s):
    return FewBodiesBeads(s)


class FewBodiesBeads:
    """Specialized chain-last (SoA) sampler for systems with few rigid bodies
       and many flexible beads (R1-R7)."""
    incremental = True
    dtype = staticmethod(E.Engine.dtype)
    _es_thresh = 32768

    def __init__(self, s, grid_h=6.0, full_dense=False, body_grid=False,
                 body_grid_h=4.0, blocked=None, precompute=True):
        if s.mode != 'serial':
            raise Unsupported(f'few_bodies_beads template: mover mode '
                              f'"{s.mode}" (requires "serial")')
        assert s.mode == 'serial', (
            f'few_bodies_beads requires s.mode == "serial", got {s.mode}')
        assert len(s.moved_nuis) == 0, (
            'few_bodies_beads requires no NormalMovers (no moved nuisances)')
        self.s = s
        self.steps_per_sweep = len(s.movers)
        self.n_movers = len(s.movers)
        self.full_dense = full_dense
        self.body_grid = body_grid
        self.body_grid_h = body_grid_h
        self.blocked = (jax.default_backend() != 'cpu'
                        if blocked is None else bool(blocked))
        self.precompute = bool(precompute)

        # ---- classify terms ------------------------------------------------
        ev_ts = [t for t in s.terms if isinstance(t, basic.SoftSphereEV)]
        pair_ts = [t for t in s.terms if isinstance(t, basic.Pairs)]
        xl_ts = [t for t in s.terms if isinstance(t, basic.CrossLinks)]
        em_ts = [t for t in s.terms if isinstance(t, em.EMIsd)]
        prior_ts = [t for t in s.terms if isinstance(t, ImpJax)]
        other_ts = [t for t in s.terms
                    if not isinstance(t, (basic.SoftSphereEV, basic.Pairs,
                                          basic.CrossLinks, em.EMIsd, ImpJax))]
        if other_ts or len(ev_ts) != 1 or len(pair_ts) != 1 or len(em_ts) != 1:
            raise Unsupported('few_bodies_beads template: requires 1 '
                              'SoftSphereEV, 1 Pairs, 1 EMIsd, and CrossLinks')
        for pt in prior_ts:
            assert pt.rows_read is not None and len(pt.rows_read) == 0, (
                f'ImpJax term {pt.label} reads moving coordinate rows')
        self.ev_t = ev_t = ev_ts[0]
        self.pairs_t = pairs_t = pair_ts[0]
        self.xl_ts = xl_ts
        self.em_t = em_t = em_ts[0]
        self.prior_ts = prior_ts

        assert np.all(pairs_t.mode == basic.HUB), (
            'few_bodies_beads requires harmonic upper bound Pairs')
        assert em_t.dx0 is not None, (
            'few_bodies_beads requires static EM density particles')

        # ---- row layout (R1/R2): [body EV | beads | body other | static] ---
        kind, ref = s.row_kind, s.row_ref
        ev_rows = ev_t.rows
        is_ev = np.zeros(s.n_rows, bool)
        is_ev[ev_rows] = True
        nb = s.nb
        assert nb == 2, (
            f'few_bodies_beads requires 2 rigid bodies, got nb={nb}')
        assert np.all(is_ev[kind == BEAD]), (
            'few_bodies_beads requires every bead to participate in EV')
        bead_rows = [int(np.nonzero((kind == BEAD) & (ref == k))[0][0])
                     for k in range(s.nbead)]
        em_set = set(em_t.rows.tolist())
        assert np.all(kind[em_t.rows] == MEMBER), (
            'R4 requires all model EM Gaussians to be on moving rigid bodies')
        em_body_of_row = {int(r): int(ref[r]) for r in em_t.rows}

        blocks_ev, blocks_ot = [], []
        for b in range(nb):
            blocks_ev.append(
                [int(i) for i in ev_rows if kind[i] == MEMBER and ref[i] == b])
            mem = [i for i in range(s.n_rows)
                   if kind[i] == MEMBER and ref[i] == b and not is_ev[i]]
            blocks_ot.append([i for i in mem if i in em_set]
                             + [i for i in mem if i not in em_set])
        static_ev = [int(i) for i in ev_rows if kind[i] == STATIC]
        static_ot = [i for i in range(s.n_rows)
                     if kind[i] == STATIC and not is_ev[i]]
        dyn_ev = sum(blocks_ev, []) + bead_rows
        dyn_ot = sum(blocks_ot, [])
        self.n_de = len(dyn_ev)
        self.n_se = len(static_ev)
        self.n_ev = self.n_de + self.n_se
        assert self.n_se > 0, 'R1 requires a non-empty static EV set'

        order = np.array(dyn_ev + dyn_ot + static_ev + static_ot, np.int64)
        self.n_dyn = self.n_de + len(dyn_ot)
        self.se_off = self.n_dyn
        assert len(order) == s.n_rows and self.n_ev == len(ev_rows)
        new = np.empty(s.n_rows, np.int64)
        new[order] = np.arange(s.n_rows)
        self.order, self.new = order, new

        off = np.cumsum([0] + [len(x) for x in blocks_ev])
        self.body_ev = [(int(off[b]), len(blocks_ev[b])) for b in range(nb)]
        self.bead_off = int(off[-1])
        off2 = self.n_de + np.cumsum([0] + [len(x) for x in blocks_ot])
        self.body_ot = [(int(off2[b]), len(blocks_ot[b])) for b in range(nb)]
        self.r = s.row_radius[order]
        self.local = s.row_local[order]
        self.static_xyz = s.row_xyz0[order[self.n_dyn:]].T.copy()

        self.ev_full = np.r_[0:self.n_de, self.se_off:self.se_off + self.n_se]
        lab_of_row = dict(zip(ev_t.rows.tolist(), ev_t.lab.tolist()))
        self.ev_label = np.array([lab_of_row[int(i)]
                                  for i in order[self.ev_full]], np.int64)
        self.ev_k = float(ev_t.kk)

        # ---- EM Gaussian tables in new row order (R4, R5) ------------------
        em_new = new[em_t.rows]
        em_raw_body = np.array([em_body_of_row[int(r)] for r in em_t.rows],
                               np.int64)
        self.em_slices = []
        for b in range(nb):
            o, _ = self.body_ot[b]
            sel = np.nonzero(em_raw_body == b)[0]
            assert len(sel) > 0, f'body {b} has no EM Gaussians'
            assert sorted(em_new[sel].tolist()) == list(
                range(o, o + len(sel))), (
                'EM Gaussians must be contiguous at the start of body_ot')
            self.em_slices.append((o, len(sel)))
        perm = np.argsort(em_new)
        self.em_cloc = em_t.cloc[:, perm]       # (6, n_em)
        self.em_cglob = em_t.cglob[:, perm]     # (6, n_em)
        self.em_mass = em_t.mass[perm]
        self.em_rad = em_t.rad[perm]
        self.em_owner = em_raw_body[perm]       # body index 0..nb-1
        cuts = np.cumsum([0] + [n for _, n in self.em_slices])
        self.em_range = [(int(cuts[b]), int(cuts[b + 1])) for b in range(nb)]
        self.d_xyz = em_t.dx0                   # (3, n_dens)
        self.d_c6 = em_t.dcglob                 # (6, n_dens)
        self.d_mass = em_t.dm
        self.d_rad = em_t.drad
        self.em_md_cut = float(em_t.dcut)
        self.em_mm_cut = float(em_t.mcut)
        self.em_self_mm = float(em_t.self_mm)
        self.em_dd = float(em_t.dd)
        self.em_w = float(em_t.w)
        self.em_slope = float(em_t.slope) if em_t.slope else 0.0

        # ---- Connectivity (Pairs) tables -----------------------------------
        cpairs = np.column_stack([new[pairs_t.ri], new[pairs_t.rj]])
        self.cpairs = cpairs
        self.conn_off = pairs_t.off
        self.conn_wk = pairs_t.w * pairs_t.k

        # ---- Cross-link tables ---------------------------------------------
        x_ca, x_cb, x_same, x_r0 = [], [], [], []
        x_si, x_sj, x_psi = [], [], []
        x_len, x_slope, x_w, x_rw = [], [], [], []
        for t in xl_ts:
            assert t.con.shape[1] == 1, (
                f'few_bodies_beads requires unambiguous cross-links '
                f'(cmax == 1), got cmax={t.con.shape[1]} in {t.label}')
            c = t.con[:, 0]
            x_ca.append(new[t.ca[c]])
            x_cb.append(new[t.cb[c]])
            x_same.append(t.same[c])
            x_r0.append(t.r0[c])
            x_si.append(s.nuis0[t.s0[c]])
            x_sj.append(s.nuis0[t.s1[c]])
            x_psi.append(s.nuis0[t.psi[c]])
            n_r = len(c)
            x_len.append(np.full(n_r, float(t.length)))
            x_slope.append(np.full(
                n_r, 0.0 if t.slope is None else float(t.slope)))
            x_w.append(np.full(n_r, float(t.w)))
            x_rw.append(t.rw)
        if xl_ts:
            self.xrows = np.column_stack([np.concatenate(x_ca),
                                          np.concatenate(x_cb)])
            self.xl_same = np.concatenate(x_same)
            self.xl_r0 = np.concatenate(x_r0)
            self.xl_si = np.concatenate(x_si)
            self.xl_sj = np.concatenate(x_sj)
            self.xl_psi = np.concatenate(x_psi)
            self.xl_len = np.concatenate(x_len)
            self.xl_slope = np.concatenate(x_slope)
            self.xl_w = np.concatenate(x_w)
            self.xl_rw = np.concatenate(x_rw)
            self.xl_has_slope = any(t.slope is not None for t in xl_ts)
            self.xl_len_scalar = (float(self.xl_len[0])
                                  if np.all(self.xl_len == self.xl_len[0])
                                  else None)
            self.xl_slope_scalar = (
                float(self.xl_slope[0])
                if self.xl_has_slope and np.all(
                    self.xl_slope == self.xl_slope[0])
                else None)
        else:
            self.xrows = np.zeros((0, 2), np.int64)
            self.xl_same = np.zeros(0, bool)
            self.xl_r0 = np.zeros(0)
            self.xl_si = self.xl_sj = self.xl_psi = np.zeros(0)
            self.xl_len = self.xl_slope = self.xl_w = self.xl_rw = np.zeros(0)
            self.xl_has_slope = False
            self.xl_len_scalar = 21.0
            self.xl_slope_scalar = 0.0

        # ---- Per-mover static tables (R2) ----------------------------------
        xrows = self.xrows
        nde = self.n_de
        steps = []
        for mv in s.movers:
            assert mv.kind in ('ball', 'rb'), (
                f'few_bodies_beads supports only ball and rb movers, '
                f'got {mv.kind}')
            if mv.kind == 'ball':
                assert len(mv.beads) == 1
                idx = int(mv.beads[0])
                rk = self.bead_off + idx
                ci = np.nonzero((cpairs == rk).any(1))[0]
                co = np.where(cpairs[ci, 0] == rk, cpairs[ci, 1],
                              cpairs[ci, 0])
                assert not (co == rk).any(), 'bond to itself'
                xi = np.nonzero((xrows == rk).any(1)
                                & (xrows[:, 0] != xrows[:, 1]))[0]
                xo = np.where(xrows[xi, 0] == rk, xrows[xi, 1], xrows[xi, 0])
                steps.append(dict(kind='bead', row=int(rk),
                                  radius=float(mv.radius),
                                  ci=ci, co=co, xi=xi, xo=xo,
                                  evmask_dyn=np.arange(nde) != rk,
                                  evmask_all=np.arange(self.n_ev) != rk))
            else:
                assert len(mv.bodies) == 1
                idx = int(mv.bodies[0])
                (o1, n1), (o2, n2) = self.body_ev[idx], self.body_ot[idx]
                rows_b = np.r_[o1:o1 + n1, o2:o2 + n2]
                assert not (np.isin(cpairs[:, 0], rows_b)
                            ^ np.isin(cpairs[:, 1], rows_b)).any(), (
                    'Pairs restraint between a moving rigid body and an '
                    'external row is not supported')
                lab_b = self.ev_label[o1:o1 + n1]
                inb = np.isin(xrows, rows_b)
                xi = np.nonzero(inb.any(1) & ~inb.all(1))[0]
                swap = ~inb[xi, 0]
                xa = np.where(swap, xrows[xi, 1], xrows[xi, 0])
                xb = np.where(swap, xrows[xi, 0], xrows[xi, 1])
                assert (lab_b[:, None] != self.ev_label[None, nde:]).all()
                evmask_dyn = lab_b[:, None] != self.ev_label[None, :nde]
                evmask_dyn[:, o1:o1 + n1] = False
                steps.append(dict(
                    kind='rb', body=int(idx),
                    max_trans=float(mv.max_trans), max_rot=float(mv.max_rot),
                    xi=xi, xa=xa, xb=xb,
                    evmask_dyn=evmask_dyn,
                    evmask_all=lab_b[:, None] != self.ev_label[None, :],
                    local_ev=self.local[o1:o1 + n1].T,
                    local_ot=self.local[o2:o2 + n2].T,
                    local_xa=self.local[xa].T))
        self.steps = steps

        # ---- Preconditions (R1, R3, R6, R7) --------------------------------
        self._bead_steps = np.array(
            [i for i, p in enumerate(self.steps) if p['kind'] == 'bead'],
            dtype=np.int32)
        self._rb_steps = np.array(
            [i for i, p in enumerate(self.steps) if p['kind'] == 'rb'],
            dtype=np.int32)
        self._bead_radii = np.array(
            [self.steps[i]['radius'] for i in self._bead_steps],
            dtype=np.float64)
        assert (len(self._bead_steps) == s.nbead and np.array_equal(
            [self.steps[i]['row'] for i in self._bead_steps],
            np.arange(self.bead_off, self.bead_off + s.nbead))), (
            'few_bodies_beads (R3/R6) requires one move per bead per sweep '
            'in contiguous row order')

        lab = self.ev_label
        self.ev_mask_dd = (np.triu(np.ones((nde, nde), bool), 1)
                           & (lab[:nde, None] != lab[None, :nde]))
        self.ev_mask_ds = lab[:nde, None] != lab[None, nde:]
        assert self.ev_mask_ds.all(), (
            'R1 requires no dynamic EV row to share an exclusion label with '
            'a static EV row')

        # Static-vs-static EV constant
        Xs = s.row_xyz0[order[self.ev_full[nde:]]]
        rs = self.r[self.ev_full[nde:]]
        d_ss = np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1))
        x_ss = np.minimum(d_ss - rs[:, None] - rs[None], 0.)
        m_ss = (np.triu(np.ones((len(rs), len(rs)), bool), 1)
                & (lab[nde:, None] != lab[None, nde:]))
        self.ev_static_const = 0.5 * self.ev_k * float(
            (x_ss * x_ss)[m_ss].sum())

        # Static EV spatial grid (R1)
        S_ev = self.static_xyz[:, :self.n_se].T
        R_ev = self.r[self.se_off:self.se_off + self.n_se]
        rq_max = float(self.r[:self.n_de].max())
        self.static_g = build_grid(S_ev, R_ev, rq_max, grid_h)
        assert self.static_g['P'].shape[1] <= np.iinfo(np.uint16).max, (
            f'static EV size {self.static_g["P"].shape[1]} exceeds uint16')

        self._seg_cache = {}
        self._full_j = {}
        if body_grid:
            self._build_body_grids()

    def _get_exec_segs(self, max_blk):
        if max_blk not in self._seg_cache:
            self._seg_cache[max_blk] = build_exec_segs(
                self.steps, self.n_de, max_blk)
        return self._seg_cache[max_blk]

    def _build_body_grids(self):
        s = self.s
        rq = float(self.r[:self.n_de].max())
        self.body_g = [build_grid(self.local[o1:o1 + n1],
                                  self.r[o1:o1 + n1], rq, self.body_grid_h)
                       for o1, n1 in self.body_ev]
        lab = self.ev_label
        for o1, n1 in self.body_ev:
            assert len(set(lab[o1:o1 + n1])) == 1
        o, nb = self.bead_off, s.nbead
        assert o + nb == self.n_de
        assert len(set(lab[o:o + nb])) == nb
        assert not set(lab[o:o + nb]) & set(lab[:o])

    def variant(self, body_grid):
        """Return a shallow copy with per-body EV grids enabled or disabled."""
        if body_grid and not hasattr(self, 'body_g'):
            self._build_body_grids()
        v = copy.copy(self)
        v.body_grid = body_grid
        return v

    # ---------------- geometry & lookup helpers -----------------------------
    def _static(self, X):
        return jnp.asarray(self.static_xyz, X.dtype)

    def take(self, X, idx):
        """Positions (3, len(idx), C or 1) of rows idx from dynamic X
           (3, n_dyn, C) and static rows."""
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

    def _slice_rows(self, X, rows):
        """Concatenate 1-row static slices along axis 1 to avoid XLA kGather
           fusions for small index lists."""
        return jnp.concatenate([X[:, int(r):int(r) + 1] for r in rows], axis=1)

    def _dist_partners(self, X, idx, xqx, xqy, xqz):
        """Distances (len(idx), C) between partner rows idx and query
           components (xqx, xqy, xqz) without static broadcasts or kGather."""
        idx = np.asarray(idx)
        d = idx < self.n_dyn
        if d.all():
            xc = self._slice_rows(X, idx)
            return dist3(xc[0], xc[1], xc[2], xqx, xqy, xqz)
        S = self._static(X)
        if not d.any():
            sc = S[:, idx - self.n_dyn][:, :, None]
            return dist3(sc[0], sc[1], sc[2], xqx, xqy, xqz)
        xd = self._slice_rows(X, idx[d])
        sc = S[:, idx[~d] - self.n_dyn][:, :, None]
        if xqx.shape[0] == 1:
            dd = dist3(xd[0], xd[1], xd[2], xqx, xqy, xqz)
            ds = dist3(sc[0], sc[1], sc[2], xqx, xqy, xqz)
        else:
            nz_d = np.nonzero(d)[0]
            nz_s = np.nonzero(~d)[0]
            qx_d = jnp.concatenate([xqx[k:k + 1] for k in nz_d], axis=0)
            qy_d = jnp.concatenate([xqy[k:k + 1] for k in nz_d], axis=0)
            qz_d = jnp.concatenate([xqz[k:k + 1] for k in nz_d], axis=0)
            qx_s = jnp.concatenate([xqx[k:k + 1] for k in nz_s], axis=0)
            qy_s = jnp.concatenate([xqy[k:k + 1] for k in nz_s], axis=0)
            qz_s = jnp.concatenate([xqz[k:k + 1] for k in nz_s], axis=0)
            dd = dist3(xd[0], xd[1], xd[2], qx_d, qy_d, qz_d)
            ds = dist3(sc[0], sc[1], sc[2], qx_s, qy_s, qz_s)
        runs = []
        id_ = is_ = 0
        k = 0
        n = len(d)
        while k < n:
            val = bool(d[k])
            j = k + 1
            while j < n and bool(d[j]) == val:
                j += 1
            cnt = j - k
            if val:
                runs.append(dd[id_:id_ + cnt])
                id_ += cnt
            else:
                runs.append(ds[is_:is_ + cnt])
                is_ += cnt
            k = j
        return jnp.concatenate(runs, axis=0)

    def _static_ev_grid(self, xq, rq):
        return grid_ev(self.static_g, xq, rq, xq.dtype)

    def _body_ev(self, b, xq, rq, R, t):
        d = xq - t[:, None]
        xl = (R[0][:, None] * d[0][None] + R[1][:, None] * d[1][None]
              + R[2][:, None] * d[2][None])
        return grid_ev(self.body_g[b], xl, rq, xq.dtype)

    # ---------------- term evaluations (R1, R4, R5) -------------------------
    def _em(self, X, quat):
        """Full EM term (C,); quat (4, nb, C)."""
        s = self.s
        dt = X.dtype
        xm = jnp.concatenate([X[:, o:o + n] for o, n in self.em_slices], 1)
        if s.cov_mode == 'frozen':
            cf = jnp.asarray(self.em_cglob, dt)[..., None]      # (6, n_em, 1)
            cm6 = tuple(cf[k] for k in range(6))
        else:
            R9 = tuple(r[self.em_owner] for r in quat_to_rot_eigen9(quat))
            cl = jnp.asarray(self.em_cloc, dt)[..., None]       # (6, n_em, 1)
            cm6 = rot_cov6(R9, tuple(cl[k] for k in range(6)))
        dx = jnp.asarray(self.d_xyz, dt)                        # (3, n_dens)
        dc = jnp.asarray(self.d_c6, dt)                         # (6, n_dens)
        dc6 = tuple(dc[k] for k in range(6))
        mass_m = jnp.asarray(self.em_mass, dt)
        rad_m = jnp.asarray(self.em_rad, dt)
        md = overlap_cov6(
            xm[:, :, None], tuple(c[:, None] for c in cm6),
            mass_m[:, None, None],
            dx[:, None, :, None], tuple(c[None, :, None] for c in dc6),
            jnp.asarray(self.d_mass, dt)[None, :, None])
        dist = dist3(xm[0, :, None], xm[1, :, None], xm[2, :, None],
                     dx[0, None, :, None], dx[1, None, :, None],
                     dx[2, None, :, None])
        sph = dist - rad_m[:, None, None] \
            - jnp.asarray(self.d_rad, dt)[None, :, None]
        mds = jnp.sum(jnp.where(sph < self.em_md_cut, md, 0.), axis=(0, 1))
        mm = overlap_cov6(
            xm[:, :, None], tuple(c[:, None] for c in cm6),
            mass_m[:, None, None],
            xm[:, None, :], tuple(c[None, :] for c in cm6),
            mass_m[None, :, None])
        dmm = dist3(xm[0, :, None], xm[1, :, None], xm[2, :, None],
                    xm[0, None, :], xm[1, None, :], xm[2, None, :])
        sphm = dmm - rad_m[:, None, None] - rad_m[None, :, None]
        n = xm.shape[1]
        iu = jnp.asarray(np.triu(np.ones((n, n), bool), 1))[..., None]
        mmx = jnp.sum(jnp.where(iu & (sphm < self.em_mm_cut), mm, 0.),
                      axis=(0, 1))
        e = -jnp.log(2. * mds / (self.em_self_mm + 2. * mmx + self.em_dd))
        if self.em_slope:
            e = e + self.em_slope * jnp.sum(dist, axis=(0, 1))
        return self.em_w * e

    def _em_parts(self, X, quat):
        """Per-body model-data overlap sums, distance sums, and intra-body
           model-model overlap sum for incremental rigid-body EM updates."""
        s, dt = self.s, X.dtype
        xm = jnp.concatenate([X[:, o:o + n] for o, n in self.em_slices], 1)
        if s.cov_mode == 'frozen':
            cf = jnp.asarray(self.em_cglob, dt)[..., None]
            cm6 = tuple(cf[k] for k in range(6))
        else:
            R9 = tuple(r[self.em_owner] for r in quat_to_rot_eigen9(quat))
            cl = jnp.asarray(self.em_cloc, dt)[..., None]
            cm6 = rot_cov6(R9, tuple(cl[k] for k in range(6)))
        dx = jnp.asarray(self.d_xyz, dt)
        dc = jnp.asarray(self.d_c6, dt)
        dc6 = tuple(dc[k] for k in range(6))
        mass_m = jnp.asarray(self.em_mass, dt)
        rad_m = jnp.asarray(self.em_rad, dt)
        md = overlap_cov6(
            xm[:, :, None], tuple(c[:, None] for c in cm6),
            mass_m[:, None, None],
            dx[:, None, :, None], tuple(c[None, :, None] for c in dc6),
            jnp.asarray(self.d_mass, dt)[None, :, None])
        dist = dist3(xm[0, :, None], xm[1, :, None], xm[2, :, None],
                     dx[0, None, :, None], dx[1, None, :, None],
                     dx[2, None, :, None])
        sph = dist - rad_m[:, None, None] \
            - jnp.asarray(self.d_rad, dt)[None, :, None]
        mdm = jnp.sum(jnp.where(sph < self.em_md_cut, md, 0.), 1)
        dsm = jnp.sum(dist, 1)
        mm = overlap_cov6(
            xm[:, :, None], tuple(c[:, None] for c in cm6),
            mass_m[:, None, None],
            xm[:, None, :], tuple(c[None, :] for c in cm6),
            mass_m[None, :, None])
        dmm = dist3(xm[0, :, None], xm[1, :, None], xm[2, :, None],
                    xm[0, None, :], xm[1, None, :], xm[2, None, :])
        sphm = dmm - rad_m[:, None, None] - rad_m[None, :, None]
        intra_b = []
        for a, e in self.em_range:
            nb_em = e - a
            iu_b = jnp.asarray(np.triu(np.ones((nb_em, nb_em), bool), 1))[
                ..., None]
            intra_b.append(jnp.sum(
                jnp.where(iu_b & (sphm[a:e, a:e] < self.em_mm_cut),
                          mm[a:e, a:e], 0.),
                axis=(0, 1)))
        return ([mdm[a:b].sum(0) for a, b in self.em_range]
                + [dsm[a:b].sum(0) for a, b in self.em_range]
                + intra_b)

    def _em_rb(self, b, X, Xb2, Rn9, q_other, em_parts, em_intra):
        """Incremental EM score when rigid body b (of 2) is proposed at
           (Xb2, Rn9). When cov_mode == 'rotate' (R4), intra-body overlap
           em_intra is constant across the sweep and not recomputed."""
        s, dt = self.s, X.dtype
        b2 = 1 - b
        a1, e1 = self.em_range[b]
        a2, e2 = self.em_range[b2]
        n_em1 = e1 - a1
        o_em2, n_em2 = self.em_slices[b2]
        xm1 = Xb2[:, :n_em1]
        xm2 = X[:, o_em2:o_em2 + n_em2]
        if s.cov_mode == 'frozen':
            cf1 = jnp.asarray(self.em_cglob[:, a1:e1], dt)[..., None]
            cf2 = jnp.asarray(self.em_cglob[:, a2:e2], dt)[..., None]
            cm1_6 = tuple(cf1[k] for k in range(6))
            cm2_6 = tuple(cf2[k] for k in range(6))
        else:
            cl1 = jnp.asarray(self.em_cloc[:, a1:e1], dt)[..., None]
            cl2 = jnp.asarray(self.em_cloc[:, a2:e2], dt)[..., None]
            cm1_6 = rot_cov6(tuple(r[None, :] for r in Rn9),
                             tuple(cl1[k] for k in range(6)))
            R2_9 = tuple(r[None, :] for r in quat_to_rot_eigen9(q_other))
            cm2_6 = rot_cov6(R2_9, tuple(cl2[k] for k in range(6)))
        dx = jnp.asarray(self.d_xyz, dt)
        dc = jnp.asarray(self.d_c6, dt)
        dc6 = tuple(dc[k] for k in range(6))
        mass1 = jnp.asarray(self.em_mass[a1:e1], dt)
        mass2 = jnp.asarray(self.em_mass[a2:e2], dt)
        rad1 = jnp.asarray(self.em_rad[a1:e1], dt)
        rad2 = jnp.asarray(self.em_rad[a2:e2], dt)
        md1 = overlap_cov6(
            xm1[:, :, None], tuple(c[:, None] for c in cm1_6),
            mass1[:, None, None],
            dx[:, None, :, None], tuple(c[None, :, None] for c in dc6),
            jnp.asarray(self.d_mass, dt)[None, :, None])
        dist1 = dist3(xm1[0, :, None], xm1[1, :, None], xm1[2, :, None],
                      dx[0, None, :, None], dx[1, None, :, None],
                      dx[2, None, :, None])
        sph1 = dist1 - rad1[:, None, None] \
            - jnp.asarray(self.d_rad, dt)[None, :, None]
        mds1 = jnp.sum(jnp.where(sph1 < self.em_md_cut, md1, 0.), axis=(0, 1))
        dsum1 = jnp.sum(dist1, axis=(0, 1))
        mm12 = overlap_cov6(
            xm1[:, :, None], tuple(c[:, None] for c in cm1_6),
            mass1[:, None, None],
            xm2[:, None, :], tuple(c[None, :] for c in cm2_6),
            mass2[None, :, None])
        dmm12 = dist3(xm1[0, :, None], xm1[1, :, None], xm1[2, :, None],
                      xm2[0, None, :], xm2[1, None, :], xm2[2, None, :])
        sphm12 = dmm12 - rad1[:, None, None] - rad2[None, :, None]
        mmx_inter = jnp.sum(jnp.where(sphm12 < self.em_mm_cut, mm12, 0.),
                            axis=(0, 1))
        if s.cov_mode == 'frozen':
            mm11 = overlap_cov6(
                xm1[:, :, None], tuple(c[:, None] for c in cm1_6),
                mass1[:, None, None],
                xm1[:, None, :], tuple(c[None, :] for c in cm1_6),
                mass1[None, :, None])
            dmm11 = dist3(xm1[0, :, None], xm1[1, :, None], xm1[2, :, None],
                          xm1[0, None, :], xm1[1, None, :], xm1[2, None, :])
            sphm11 = dmm11 - rad1[:, None, None] - rad1[None, :, None]
            iu1 = jnp.asarray(np.triu(np.ones((n_em1, n_em1), bool), 1))[
                ..., None]
            intra1 = jnp.sum(
                jnp.where(iu1 & (sphm11 < self.em_mm_cut), mm11, 0.),
                axis=(0, 1))
            mmx = mmx_inter + intra1 + em_parts[4 + b2]
        else:
            intra1 = None
            mmx = mmx_inter + em_intra
        mds = mds1 + em_parts[b2]
        dsum = dsum1 + em_parts[2 + b2]
        e = -jnp.log(2. * mds / (self.em_self_mm + 2. * mmx + self.em_dd))
        if self.em_slope:
            e = e + self.em_slope * dsum
        return self.em_w * e, mds1, dsum1, intra1

    def _xl_terms(self, dist, xi):
        """Weighted -log(w_r * p_r) of cross-links xi at distances
           (len(xi), C), matching basic.CrossLinks.energies."""
        dt = dist.dtype
        s0 = jnp.asarray(self.xl_si[xi], dt)[:, None]
        s1 = jnp.asarray(self.xl_sj[xi], dt)[:, None]
        psi = jnp.asarray(self.xl_psi[xi], dt)[:, None]
        length = (self.xl_len_scalar if self.xl_len_scalar is not None
                  else jnp.asarray(self.xl_len[xi], dt)[:, None])
        slope = (None if not self.xl_has_slope else (
            self.xl_slope_scalar if self.xl_slope_scalar is not None
            else jnp.asarray(self.xl_slope[xi], dt)[:, None]))
        term = geom.xl_term(dist, s0, s1, psi, length, slope, False)
        prob = 1.0 - (1.0 - term)
        rw = jnp.asarray(self.xl_rw[xi], dt)[:, None]
        w = jnp.asarray(self.xl_w[xi], dt)[:, None]
        return -w * jnp.log(prob * rw)

    def _static_ev(self, X):
        """(nbead, C) and (nb, C): EV energy (without 0.5 k) vs the static
           EV rows, through the static grid."""
        nbd = self.s.nbead
        o = self.bead_off
        e_all = self._static_ev_grid(X[:, :self.n_de], self.r[:self.n_de])
        eb = e_all[o:o + nbd]
        ebody = jnp.stack([jnp.sum(e_all[o1:o1 + n1], 0)
                           for o1, n1 in self.body_ev])
        return eb, ebody

    def _full(self, X, quat, nuis, with_static=False):
        """All terms for all chains from scratch."""
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        cp = self.cpairs
        ca, cb = self.take(X, cp[:, 0]), self.take(X, cp[:, 1])
        d_c = dist3(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2])
        x_c = jnp.maximum(
            d_c - jnp.asarray(self.conn_off, dt)[:, None], 0.)
        conn = jnp.sum(
            jnp.asarray(self.conn_wk, dt)[:, None] * (0.5 * x_c * x_c), 0)

        nde, nse = self.n_de, self.n_se
        Xd = X[:, :nde]
        rd = r[:nde]
        dd = dist3(Xd[0, :, None], Xd[1, :, None], Xd[2, :, None],
                   Xd[0, None, :], Xd[1, None, :], Xd[2, None, :])
        if self.full_dense:
            Ss = self._static(X)[:, :nse]
            rs = r[self.se_off:self.se_off + nse]
            ds = dist3(Xd[0, :, None], Xd[1, :, None], Xd[2, :, None],
                       Ss[0, None, :, None], Ss[1, None, :, None],
                       Ss[2, None, :, None])
            e_ds = jnp.sum(soft_pair(ds, (rd[:, None] + rs[None])[..., None]),
                           axis=(0, 1))
            if with_static:
                eb, ebody = self._static_ev(X)
        else:
            eb, ebody = self._static_ev(X)
            e_ds = jnp.sum(eb, 0) + jnp.sum(ebody, 0)
        rsum = (rd[:, None] + rd[None])[..., None]
        ev = 0.5 * self.ev_k * (
            jnp.sum(jnp.where(jnp.asarray(self.ev_mask_dd)[..., None],
                              soft_pair(dd, rsum), 0.), axis=(0, 1))
            + e_ds) + self.ev_static_const

        xr = self.xrows
        if len(xr):
            xa, xb = self.take(X, xr[:, 0]), self.take(X, xr[:, 1])
            dist = dist3(xa[0], xa[1], xa[2], xb[0], xb[1], xb[2])
            if self.xl_same.any():
                dist = jnp.where(
                    jnp.asarray(self.xl_same)[:, None],
                    (36.0 / 35.0 * jnp.asarray(self.xl_r0, dt))[:, None],
                    dist)
            xl = jnp.sum(self._xl_terms(dist, np.arange(len(xr))), 0)
        else:
            xl = jnp.zeros(X.shape[-1], dt)

        em_e = self._em(X, quat)
        priors = jnp.zeros(X.shape[-1], dt)
        if self.prior_ts:
            dummy_core = dict(X=X, quat=quat, nuis=nuis)
            for pt in self.prior_ts:
                priors = priors + pt.full(dummy_core)
        tot = conn + ev + xl + em_e + priors
        return (tot, em_e, eb, ebody) if with_static else (tot, em_e)

    # ---------------- fused step kernels (R2, R3, R4, R5, R6) ---------------
    def _bead_step_fast(self, i, X, X0_use, quat_l, trans_l, energy, es_b,
                        acc_l, kT, nrm, uni, es_n_all):
        s, p = self.s, self.steps[i]
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        rk = p['row']
        Xsrc = X0_use if X0_use is not None else X
        xox, xoy, xoz = Xsrc[0, rk], Xsrc[1, rk], Xsrc[2, rk]
        dx, dy, dz = nrm[i, 0], nrm[i, 1], nrm[i, 2]
        norm_d = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        step_r = p['radius'] * jnp.cbrt(uni[i, 0])
        xnx = xox + (dx / norm_d) * step_r
        xny = xoy + (dy / norm_d) * step_r
        xnz = xoz + (dz / norm_d) * step_r
        xn = jnp.stack([xnx, xny, xnz], 0)
        xo = jnp.stack([xox, xoy, xoz], 0)
        nde, k = self.n_de, rk - self.bead_off
        if self.body_grid:
            o, nb = self.bead_off, s.nbead
            Xd = X[:, o:o + nb]
            rr = (r[rk] + r[o:o + nb])[:, None]
            mask = np.arange(nb) != k
            e = jnp.sum(jnp.where(
                jnp.asarray(mask)[:, None],
                soft_pair(dist3(Xd[0], Xd[1], Xd[2],
                                xnx[None], xny[None], xnz[None]), rr)
                - soft_pair(dist3(Xd[0], Xd[1], Xd[2],
                                  xox[None], xoy[None], xoz[None]), rr),
                0.), 0)
            xq = jnp.stack([xn, xo], 1)
            rq = np.full(2, self.r[rk])
            for b in range(s.nb):
                eg = self._body_ev(b, xq, rq,
                                   quat_to_rot_imp(quat_l[b], jnp),
                                   trans_l[b])
                e = e + eg[0] - eg[1]
        else:
            Xdx, Xdy, Xdz = X[0, :nde], X[1, :nde], X[2, :nde]
            rr = (r[rk] + r[:nde])[:, None]
            e = jnp.sum(jnp.where(
                jnp.asarray(p['evmask_dyn'])[:, None],
                soft_pair(dist3(Xdx, Xdy, Xdz,
                                xnx[None], xny[None], xnz[None]), rr)
                - soft_pair(dist3(Xdx, Xdy, Xdz,
                                  xox[None], xoy[None], xoz[None]), rr),
                0.), 0)
        if es_n_all is not None:
            es_n = es_n_all[k]
        else:
            es_n = self._static_ev_grid(xn[:, None], r[rk][None])[0]
        es_o = es_b[k]
        de = 0.5 * self.ev_k * (e + es_n - es_o)
        if len(p['ci']):
            S = self._static(X)
            c_sum = None
            for m in range(len(p['ci'])):
                ci_m = int(p['ci'][m])
                co_m = int(p['co'][m])
                rc = jnp.asarray(self.conn_off[ci_m], dt)
                wk = jnp.asarray(self.conn_wk[ci_m], dt)
                if co_m < self.n_dyn:
                    cx, cy, cz = X[0, co_m], X[1, co_m], X[2, co_m]
                else:
                    sr = co_m - self.n_dyn
                    cx, cy, cz = S[0, sr], S[1, sr], S[2, sr]
                do = dist3(cx, cy, cz, xox, xoy, xoz)
                dn = dist3(cx, cy, cz, xnx, xny, xnz)
                bo = jnp.maximum(do - rc, 0.)
                bn = jnp.maximum(dn - rc, 0.)
                term = wk * (0.5 * bn * bn) - wk * (0.5 * bo * bo)
                c_sum = term if c_sum is None else (c_sum + term)
            de = de + c_sum
        if len(p['xi']):
            do = self._dist_partners(X, p['xo'],
                                     xox[None], xoy[None], xoz[None])
            dn = self._dist_partners(X, p['xo'],
                                     xnx[None], xny[None], xnz[None])
            de = de + jnp.sum(self._xl_terms(dn, p['xi'])
                              - self._xl_terms(do, p['xi']), 0)
        acc = uni[i, 1] < jnp.exp(-de / kT)
        X = jax.lax.dynamic_update_slice(
            X, jnp.where(acc, xn, xo)[:, None], (0, rk, 0))
        es_b[k] = jnp.where(acc, es_n, es_o)
        energy = energy + jnp.where(acc, de, 0.)
        acc_l[i] = acc_l[i] + acc.astype(jnp.int32)
        return X, energy, acc

    def _bead_block_batched(self, idxs, mask_ext, X, X0_use, energy, es_b,
                            acc_l, kT, nrm, uni, es_n_all, xcx_bar=True):
        """R3 blocked bead steps for smaller chain counts (C <= 4096)."""
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        B = len(idxs)
        o = self.steps[idxs[0]]['row']
        nde = self.n_de
        Xsrc = X0_use if X0_use is not None else X
        Xo_blk = Xsrc[:, o:o + B]
        i0 = idxs[0]
        d_blk = jnp.moveaxis(nrm[i0:i0 + B, :3], 1, 0)
        u0_blk = uni[i0:i0 + B, 0]
        rad_blk = jnp.asarray(
            [self.steps[i]['radius'] for i in idxs], dt)[:, None]
        norm_blk = jnp.sqrt(
            d_blk[0] * d_blk[0] + d_blk[1] * d_blk[1] + d_blk[2] * d_blk[2])
        step_blk = rad_blk * jnp.cbrt(u0_blk)
        Xn_blk = Xo_blk + (d_blk / norm_blk[None]) * step_blk[None]
        Xo_blk, Xn_blk = jax.lax.optimization_barrier((Xo_blk, Xn_blk))

        Xdx, Xdy, Xdz = X[0, :nde], X[1, :nde], X[2, :nde]
        rr = (r[o:o + B, None] + r[None, :nde])[..., None]
        pe_ext = jnp.where(
            jnp.asarray(mask_ext)[..., None],
            soft_pair(dist3(Xn_blk[0, :, None], Xn_blk[1, :, None],
                            Xn_blk[2, :, None],
                            Xdx[None], Xdy[None], Xdz[None]), rr)
            - soft_pair(dist3(Xo_blk[0, :, None], Xo_blk[1, :, None],
                              Xo_blk[2, :, None],
                              Xdx[None], Xdy[None], Xdz[None]), rr),
            0.)
        e_ext = jax.lax.optimization_barrier(jnp.sum(pe_ext, axis=1))

        xox = [Xo_blk[0, m] for m in range(B)]
        xoy = [Xo_blk[1, m] for m in range(B)]
        xoz = [Xo_blk[2, m] for m in range(B)]
        xnx = [Xn_blk[0, m] for m in range(B)]
        xny = [Xn_blk[1, m] for m in range(B)]
        xnz = [Xn_blk[2, m] for m in range(B)]

        S = self._static(X)
        xl_ext = {}
        for m, i in enumerate(idxs):
            p = self.steps[i]
            if len(p['xi']):
                xo_arr = np.asarray(p['xo'])
                in_prev = (xo_arr >= o) & (xo_arr < o + m)
                if not in_prev.any():
                    do = self._dist_partners(
                        X, xo_arr, xox[m][None], xoy[m][None], xoz[m][None])
                    dn = self._dist_partners(
                        X, xo_arr, xnx[m][None], xny[m][None], xnz[m][None])
                    xl_ext[m] = jnp.sum(
                        self._xl_terms(dn, p['xi'])
                        - self._xl_terms(do, p['xi']), 0)
                else:
                    xl_ext[m] = None

        xcx, xcy, xcz = [], [], []
        accs = []
        for m, i in enumerate(idxs):
            p = self.steps[i]
            rk = o + m
            k = rk - self.bead_off
            e_m = e_ext[m]
            for m2 in range(m):
                rr_m2 = jnp.asarray(self.r[rk] + self.r[o + m2], dt)
                dn_m2 = dist3(xcx[m2], xcy[m2], xcz[m2],
                              xnx[m], xny[m], xnz[m])
                do_m2 = dist3(xcx[m2], xcy[m2], xcz[m2],
                              xox[m], xoy[m], xoz[m])
                e_m = e_m + (soft_pair(dn_m2, rr_m2)
                             - soft_pair(do_m2, rr_m2))
            if es_n_all is not None:
                es_n = es_n_all[k]
            else:
                xn_m = jnp.stack([xnx[m], xny[m], xnz[m]], 0)
                es_n = self._static_ev_grid(xn_m[:, None], r[rk][None])[0]
            es_o = es_b[k]
            de = 0.5 * self.ev_k * (e_m + es_n - es_o)
            if len(p['ci']):
                c_sum = None
                for b_idx in range(len(p['ci'])):
                    ci_m = int(p['ci'][b_idx])
                    co_m = int(p['co'][b_idx])
                    rc = jnp.asarray(self.conn_off[ci_m], dt)
                    wk = jnp.asarray(self.conn_wk[ci_m], dt)
                    if o <= co_m < o + m:
                        m2 = co_m - o
                        cx, cy, cz = xcx[m2], xcy[m2], xcz[m2]
                    elif co_m < self.n_dyn:
                        cx, cy, cz = X[0, co_m], X[1, co_m], X[2, co_m]
                    else:
                        sr = co_m - self.n_dyn
                        cx, cy, cz = S[0, sr], S[1, sr], S[2, sr]
                    do = dist3(cx, cy, cz, xox[m], xoy[m], xoz[m])
                    dn = dist3(cx, cy, cz, xnx[m], xny[m], xnz[m])
                    bo = jnp.maximum(do - rc, 0.)
                    bn = jnp.maximum(dn - rc, 0.)
                    term = wk * (0.5 * bn * bn) - wk * (0.5 * bo * bo)
                    c_sum = term if c_sum is None else (c_sum + term)
                de = de + c_sum
            if len(p['xi']):
                if xl_ext[m] is not None:
                    de = de + xl_ext[m]
                else:
                    xo_arr = np.asarray(p['xo'])
                    xi_arr = np.asarray(p['xi'])
                    in_prev = (xo_arr >= o) & (xo_arr < o + m)
                    if (~in_prev).any():
                        do_e = self._dist_partners(
                            X, xo_arr[~in_prev],
                            xox[m][None], xoy[m][None], xoz[m][None])
                        dn_e = self._dist_partners(
                            X, xo_arr[~in_prev],
                            xnx[m][None], xny[m][None], xnz[m][None])
                        de = de + jnp.sum(
                            self._xl_terms(dn_e, xi_arr[~in_prev])
                            - self._xl_terms(do_e, xi_arr[~in_prev]), 0)
                    for idx_p in np.nonzero(in_prev)[0]:
                        m2 = int(xo_arr[idx_p]) - o
                        xi_p = xi_arr[idx_p:idx_p + 1]
                        do_p = dist3(xcx[m2], xcy[m2], xcz[m2],
                                     xox[m], xoy[m], xoz[m])[None]
                        dn_p = dist3(xcx[m2], xcy[m2], xcz[m2],
                                     xnx[m], xny[m], xnz[m])[None]
                        de = de + jnp.sum(
                            self._xl_terms(dn_p, xi_p)
                            - self._xl_terms(do_p, xi_p), 0)
            acc = uni[i, 1] < jnp.exp(-de / kT)
            xcx.append(jnp.where(acc, xnx[m], xox[m]))
            xcy.append(jnp.where(acc, xny[m], xoy[m]))
            xcz.append(jnp.where(acc, xnz[m], xoz[m]))
            es_b[k] = jnp.where(acc, es_n, es_o)
            energy = energy + jnp.where(acc, de, 0.)
            acc_l[i] = acc_l[i] + acc.astype(jnp.int32)
            accs.append(acc)

        if xcx_bar:
            xcx, xcy, xcz = jax.lax.optimization_barrier(
                (tuple(xcx), tuple(xcy), tuple(xcz)))
        Xb_upd = jnp.stack(
            [jnp.stack(xcx, 0), jnp.stack(xcy, 0), jnp.stack(xcz, 0)], 0)
        od = jax.lax.optimization_barrier(jnp.int32(o))
        X = jax.lax.dynamic_update_slice_in_dim(X, Xb_upd, od, axis=1)
        return X, energy, accs

    def _bead_block_chained(self, idxs, mask_ext, X, X0_use, energy, es_b,
                            acc_l, kT, nrm, uni, es_n_all):
        """R3 blocked bead steps for larger chain counts (C > 4096)."""
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        B = len(idxs)
        o = self.steps[idxs[0]]['row']
        nde = self.n_de
        Xsrc = X0_use if X0_use is not None else X

        xox, xoy, xoz = [], [], []
        xnx, xny, xnz = [], [], []
        for m, i in enumerate(idxs):
            rk = o + m
            ox, oy, oz = Xsrc[0, rk], Xsrc[1, rk], Xsrc[2, rk]
            dx, dy, dz = nrm[i, 0], nrm[i, 1], nrm[i, 2]
            norm_d = jnp.sqrt(dx * dx + dy * dy + dz * dz)
            step_r = self.steps[i]['radius'] * jnp.cbrt(uni[i, 0])
            xox.append(ox)
            xoy.append(oy)
            xoz.append(oz)
            xnx.append(ox + (dx / norm_d) * step_r)
            xny.append(oy + (dy / norm_d) * step_r)
            xnz.append(oz + (dz / norm_d) * step_r)

        Xdx, Xdy, Xdz = X[0, :nde], X[1, :nde], X[2, :nde]
        e_ext = []
        for m in range(B):
            rk = o + m
            rr_m = (r[rk] + r[:nde])[:, None]
            mask_m = (jnp.asarray(mask_ext[m])[:, None] if m == 0
                      else (jnp.asarray(mask_ext[m])[:, None]
                            & (e_ext[-1][None, :] < 1e30)))
            em_val = jnp.sum(jnp.where(
                mask_m,
                soft_pair(dist3(Xdx, Xdy, Xdz,
                                xnx[m][None], xny[m][None], xnz[m][None]),
                          rr_m)
                - soft_pair(dist3(Xdx, Xdy, Xdz,
                                  xox[m][None], xoy[m][None], xoz[m][None]),
                            rr_m),
                0.), 0)
            e_ext.append(jax.lax.optimization_barrier(em_val))

        S = self._static(X)
        xcx, xcy, xcz = [], [], []
        accs = []
        for m, i in enumerate(idxs):
            p = self.steps[i]
            rk = o + m
            k = rk - self.bead_off
            e_m = e_ext[m]
            for m2 in range(m):
                rr_m2 = jnp.asarray(self.r[rk] + self.r[o + m2], dt)
                dn_m2 = dist3(xcx[m2], xcy[m2], xcz[m2],
                              xnx[m], xny[m], xnz[m])
                do_m2 = dist3(xcx[m2], xcy[m2], xcz[m2],
                              xox[m], xoy[m], xoz[m])
                e_m = e_m + (soft_pair(dn_m2, rr_m2)
                             - soft_pair(do_m2, rr_m2))
            if es_n_all is not None:
                es_n = es_n_all[k]
            else:
                xn_m = jnp.stack([xnx[m], xny[m], xnz[m]], 0)
                es_n = self._static_ev_grid(xn_m[:, None], r[rk][None])[0]
            es_o = es_b[k]
            de = 0.5 * self.ev_k * (e_m + es_n - es_o)
            if len(p['ci']):
                c_sum = None
                for b_idx in range(len(p['ci'])):
                    ci_m = int(p['ci'][b_idx])
                    co_m = int(p['co'][b_idx])
                    rc = jnp.asarray(self.conn_off[ci_m], dt)
                    wk = jnp.asarray(self.conn_wk[ci_m], dt)
                    if o <= co_m < o + m:
                        m2 = co_m - o
                        cx, cy, cz = xcx[m2], xcy[m2], xcz[m2]
                    elif co_m < self.n_dyn:
                        cx, cy, cz = X[0, co_m], X[1, co_m], X[2, co_m]
                    else:
                        sr = co_m - self.n_dyn
                        cx, cy, cz = S[0, sr], S[1, sr], S[2, sr]
                    do = dist3(cx, cy, cz, xox[m], xoy[m], xoz[m])
                    dn = dist3(cx, cy, cz, xnx[m], xny[m], xnz[m])
                    bo = jnp.maximum(do - rc, 0.)
                    bn = jnp.maximum(dn - rc, 0.)
                    term = wk * (0.5 * bn * bn) - wk * (0.5 * bo * bo)
                    c_sum = term if c_sum is None else (c_sum + term)
                de = de + c_sum
            if len(p['xi']):
                xo_arr = np.asarray(p['xo'])
                xi_arr = np.asarray(p['xi'])
                in_prev = (xo_arr >= o) & (xo_arr < o + m)
                if (~in_prev).any():
                    do_e = self._dist_partners(
                        X, xo_arr[~in_prev],
                        xox[m][None], xoy[m][None], xoz[m][None])
                    dn_e = self._dist_partners(
                        X, xo_arr[~in_prev],
                        xnx[m][None], xny[m][None], xnz[m][None])
                    de = de + jnp.sum(
                        self._xl_terms(dn_e, xi_arr[~in_prev])
                        - self._xl_terms(do_e, xi_arr[~in_prev]), 0)
                for idx_p in np.nonzero(in_prev)[0]:
                    m2 = int(xo_arr[idx_p]) - o
                    xi_p = xi_arr[idx_p:idx_p + 1]
                    do_p = dist3(xcx[m2], xcy[m2], xcz[m2],
                                 xox[m], xoy[m], xoz[m])[None]
                    dn_p = dist3(xcx[m2], xcy[m2], xcz[m2],
                                 xnx[m], xny[m], xnz[m])[None]
                    de = de + jnp.sum(
                        self._xl_terms(dn_p, xi_p)
                        - self._xl_terms(do_p, xi_p), 0)
            acc = uni[i, 1] < jnp.exp(-de / kT)
            xcx.append(jnp.where(acc, xnx[m], xox[m]))
            xcy.append(jnp.where(acc, xny[m], xoy[m]))
            xcz.append(jnp.where(acc, xnz[m], xoz[m]))
            es_b[k] = jnp.where(acc, es_n, es_o)
            energy = energy + jnp.where(acc, de, 0.)
            acc_l[i] = acc_l[i] + acc.astype(jnp.int32)
            accs.append(acc)

        xcx, xcy, xcz = jax.lax.optimization_barrier(
            (tuple(xcx), tuple(xcy), tuple(xcz)))
        Xb_upd = jnp.stack(
            [jnp.stack(xcx, 0), jnp.stack(xcy, 0), jnp.stack(xcz, 0)], 0)
        od = jax.lax.optimization_barrier(jnp.int32(o))
        X = jax.lax.dynamic_update_slice_in_dim(X, Xb_upd, od, axis=1)
        return X, energy, accs

    def _rb_step_fast(self, i, X, quat_l, trans_l, es_body_l, em, em_parts,
                      em_intra, energy, acc_l, kT, nrm, uni,
                      nrm_rb_ax, uni_rb_1, rb_barrier=False):
        s, p = self.s, self.steps[i]
        dt = X.dtype
        r = jnp.asarray(self.r, dt)
        b = p['body']
        qo, to = quat_l[b], trans_l[b]
        d = nrm[i, :3]
        norm_d = jnp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        tn = to + (d / norm_d) * (p['max_trans'] * jnp.cbrt(uni[i, 0]))
        ax = nrm_rb_ax
        ax = ax / jnp.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2])
        half = 0.5 * p['max_rot'] * (2. * uni_rb_1 - 1.)
        qr = jnp.concatenate([jnp.cos(half)[None],
                              ax * jnp.sin(half)[None]], 0)
        qn = quat_mul_cf(qr, qo, jnp)
        qn = qn / jnp.sqrt(
            qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2] + qn[3] * qn[3])
        Rn = quat_to_rot_imp(qn, jnp)
        Rn9 = quat_to_rot_eigen9(qn)
        (o1, n1), (o2, n2) = self.body_ev[b], self.body_ot[b]
        Xb1 = rotate_cf(Rn, jnp.asarray(p['local_ev'], dt), tn)
        Xb2 = rotate_cf(Rn, jnp.asarray(p['local_ot'], dt), tn)
        if rb_barrier:
            Xb1, Xb2 = jax.lax.optimization_barrier((Xb1, Xb2))
        nde = self.n_de
        Xo1 = X[:, o1:o1 + n1]
        rm = r[o1:o1 + n1]
        if self.body_grid:
            o, nb = self.bead_off, s.nbead
            Xd = X[:, o:o + nb]
            rb_ = self.r[o:o + nb]
            e = jnp.sum(
                self._body_ev(b, Xd, rb_, Rn, tn)
                - self._body_ev(b, Xd, rb_, quat_to_rot_imp(qo, jnp), to), 0)
            for b2 in range(s.nb):
                if b2 == b:
                    continue
                R2, t2 = quat_to_rot_imp(quat_l[b2], jnp), trans_l[b2]
                e = e + jnp.sum(self._body_ev(b2, Xb1, self.r[o1:o1 + n1],
                                              R2, t2)
                                - self._body_ev(b2, Xo1, self.r[o1:o1 + n1],
                                                R2, t2), 0)
        else:
            rr = (rm[:, None] + r[None, :nde])[..., None]
            Xd = X[:, None, :nde]
            pe = jnp.where(
                jnp.asarray(p['evmask_dyn'])[..., None],
                soft_pair(dist3(Xb1[0, :, None], Xb1[1, :, None],
                                Xb1[2, :, None], Xd[0], Xd[1], Xd[2]), rr)
                - soft_pair(dist3(Xo1[0, :, None], Xo1[1, :, None],
                                  Xo1[2, :, None], Xd[0], Xd[1], Xd[2]), rr),
                0.)
            if n1 * nde > 8192:
                e = jnp.sum(jax.lax.optimization_barrier(jnp.sum(pe, 1)), 0)
            else:
                e = jnp.sum(pe, (0, 1))
        es_n = jnp.sum(self._static_ev_grid(Xb1, rm), 0)
        es_o = es_body_l[b]
        de = 0.5 * self.ev_k * (e + es_n - es_o)
        if len(p['xi']):
            xa, xb = p['xa'], p['xb']
            xa_new = rotate_cf(Rn, jnp.asarray(p['local_xa'], dt), tn)
            xa_old = self._slice_rows(X, xa)
            dn = self._dist_partners(X, xb, xa_new[0], xa_new[1], xa_new[2])
            do = self._dist_partners(X, xb, xa_old[0], xa_old[1], xa_old[2])
            de = de + jnp.sum(
                self._xl_terms(dn, p['xi'])
                - self._xl_terms(do, p['xi']), 0)
        em_n, mds1, dsum1, intra1 = self._em_rb(
            b, X, Xb2, Rn9, quat_l[1 - b], em_parts, em_intra)
        de = de + (em_n - em)
        acc = uni[i, 1] < jnp.exp(-de / kT)
        od1 = jax.lax.optimization_barrier(jnp.int32(o1))
        X = jax.lax.dynamic_update_slice_in_dim(
            X, jnp.where(
                acc, Xb1, jax.lax.dynamic_slice_in_dim(X, od1, n1, axis=1)),
            od1, axis=1)
        od2 = jax.lax.optimization_barrier(jnp.int32(o2))
        X = jax.lax.dynamic_update_slice_in_dim(
            X, jnp.where(
                acc, Xb2, jax.lax.dynamic_slice_in_dim(X, od2, n2, axis=1)),
            od2, axis=1)
        quat_l[b] = jnp.where(acc, qn, qo)
        trans_l[b] = jnp.where(acc, tn, to)
        es_body_l[b] = jnp.where(acc, es_n, es_o)
        em = jnp.where(acc, em_n, em)
        em_parts[b] = jnp.where(acc, mds1, em_parts[b])
        em_parts[2 + b] = jnp.where(acc, dsum1, em_parts[2 + b])
        if intra1 is not None:
            em_parts[4 + b] = jnp.where(acc, intra1, em_parts[4 + b])
        energy = energy + jnp.where(acc, de, 0.)
        acc_l[i] = acc_l[i] + acc.astype(jnp.int32)
        return X, em, energy, acc

    # ---------------- sweep & protocol --------------------------------------
    def _sweep_unpacked(self, carry, key, sweep, kT, em_intra, record=False):
        (X0, energy, em, quat_t, trans_t,
         es_body_t, es_b_t, acc_t, em_parts_t) = carry
        n, C = self.n_movers, energy.shape[0]
        dt = X0.dtype
        nrm, uni, rb_ax_arr, rb_u1_arr = sliced_draws(
            key, sweep, n, self._rb_steps, dt)
        rb_ax = tuple(rb_ax_arr[k] for k in range(len(self._rb_steps)))
        rb_u1 = tuple(rb_u1_arr[k] for k in range(len(self._rb_steps)))
        nrm, uni, rb_ax, rb_u1 = jax.lax.optimization_barrier(
            (nrm, uni, rb_ax, rb_u1))

        if self.body_grid or not self.blocked:
            exec_segs = None
        elif C > 4096:
            max_blk = 9 if C <= 16384 else (6 if C <= 32768 else 4)
            exec_segs = self._get_exec_segs(max_blk)
        else:
            max_blk = 12 if C <= 2048 else 34
            exec_segs = self._get_exec_segs(max_blk)

        X = X0
        o, nb = self.bead_off, self.s.nbead
        if self.precompute and C <= self._es_thresh:
            xo_all = X0[:, o:o + nb]
            bs = self._bead_steps
            db = jnp.moveaxis(nrm[bs, :3], 1, 0)
            ub = uni[bs, 0]
            norm_db = jnp.sqrt(
                db[0] * db[0] + db[1] * db[1] + db[2] * db[2])
            step_b = jnp.asarray(self._bead_radii, dt)[:, None] * jnp.cbrt(ub)
            xn_all = xo_all + (db / norm_db[None]) * step_b[None]
            es_n_all = jax.lax.optimization_barrier(
                self._static_ev_grid(xn_all, self.r[o:o + nb]))
        else:
            es_n_all = None

        X0_use = X0 if C <= 8192 else None
        quat_l = list(quat_t)
        trans_l = list(trans_t)
        es_body_l = list(es_body_t)
        es_b = list(es_b_t)
        acc_l = list(acc_t)
        em_parts = list(em_parts_t)
        dec_l = [None] * n if record else None

        if self.body_grid or not self.blocked:
            rb_idx = 0
            for i, p in enumerate(self.steps):
                if p['kind'] == 'bead':
                    X, energy, acc = self._bead_step_fast(
                        i, X, X0_use, quat_l, trans_l, energy, es_b, acc_l,
                        kT, nrm, uni, es_n_all)
                else:
                    X, em, energy, acc = self._rb_step_fast(
                        i, X, quat_l, trans_l, es_body_l, em, em_parts,
                        em_intra, energy, acc_l, kT, nrm, uni,
                        rb_ax[rb_idx], rb_u1[rb_idx])
                    rb_idx += 1
                if record:
                    dec_l[i] = acc
        elif C <= 4096:
            xcx_bar = (C <= 2048)
            rb_idx = 0
            for kind, val, mask_ext in exec_segs:
                if kind == 'bead_blk':
                    X, energy, accs = self._bead_block_batched(
                        val, mask_ext, X, X0_use, energy, es_b, acc_l,
                        kT, nrm, uni, es_n_all, xcx_bar=xcx_bar)
                    if record:
                        for i_m, acc in zip(val, accs):
                            dec_l[i_m] = acc
                else:
                    X, em, energy, acc = self._rb_step_fast(
                        val, X, quat_l, trans_l, es_body_l, em, em_parts,
                        em_intra, energy, acc_l, kT, nrm, uni,
                        rb_ax[rb_idx], rb_u1[rb_idx])
                    rb_idx += 1
                    if record:
                        dec_l[val] = acc
        else:
            rb_barrier = (C <= 16384)
            rb_idx = 0
            for kind, val, mask_ext in exec_segs:
                if kind == 'bead_blk':
                    X, energy, accs = self._bead_block_chained(
                        val, mask_ext, X, X0_use, energy, es_b, acc_l,
                        kT, nrm, uni, es_n_all)
                    if record:
                        for i_m, acc in zip(val, accs):
                            dec_l[i_m] = acc
                else:
                    X, em, energy, acc = self._rb_step_fast(
                        val, X, quat_l, trans_l, es_body_l, em, em_parts,
                        em_intra, energy, acc_l, kT, nrm, uni,
                        rb_ax[rb_idx], rb_u1[rb_idx], rb_barrier=rb_barrier)
                    rb_idx += 1
                    if record:
                        dec_l[val] = acc

        new_carry = (X, energy, em, tuple(quat_l), tuple(trans_l),
                     tuple(es_body_l), tuple(es_b), tuple(acc_l),
                     tuple(em_parts))
        if record:
            return new_carry, jnp.stack(dec_l, 0)
        return new_carry

    def init(self, C, seed=0, kT=None, dofs=None, chain0=0):
        s, dt = self.s, self.dtype()
        d = s.initial_dofs(C) if dofs is None else dofs
        d = {k: jnp.asarray(v, dt) for k, v in d.items()}
        X_all = s.positions(d, jnp)
        X = X_all[:, self.order[:self.n_dyn]]
        kT = s.kT if kT is None else kT
        st = dict(
            X=X,
            quat=d['quat'],
            trans=d['trans'],
            nuis=d['nuis'],
            energy=jnp.zeros(C, dt),
            em=jnp.zeros(C, dt),
            es_bead=jnp.zeros((s.nbead, C), dt),
            es_body=jnp.zeros((s.nb, C), dt),
            kT=jnp.broadcast_to(jnp.asarray(kT, dt), (C,)),
            key=E.chain_keys(seed, C, chain0),
            sweep=jnp.zeros((), jnp.int32),
            n_acc=jnp.zeros((self.n_movers, C), jnp.int32),
        )
        return jax.jit(self.resync)(st)

    def resync(self, st):
        """Recompute total energy, EM energy, and static EV caches from
           scratch."""
        e, em_e, eb, ebody = self._full(
            st['X'], st['quat'], st['nuis'], with_static=True)
        return dict(st, energy=e, em=em_e, es_bead=eb, es_body=ebody)

    def run(self, st, n_sweeps, record=False):
        s = self.s
        ep = self._em_parts(st['X'], st['quat'])
        if s.cov_mode == 'frozen':
            em_parts_init = tuple(ep[:6])
            em_intra = None
        else:
            em_parts_init = tuple(ep[:4])
            em_intra = ep[4] + ep[5]
        carry = (
            st['X'], st['energy'], st['em'],
            tuple(st['quat'][:, b] for b in range(s.nb)),
            tuple(st['trans'][:, b] for b in range(s.nb)),
            tuple(st['es_body'][b] for b in range(s.nb)),
            tuple(st['es_bead'][k] for k in range(s.nbead)),
            tuple(st['n_acc'][i] for i in range(self.n_movers)),
            em_parts_init,
        )
        key, sw0, kT = st['key'], st['sweep'], st['kT']
        if record:
            carry, decisions = jax.lax.scan(
                lambda c, sw: self._sweep_unpacked(
                    c, key, sw, kT, em_intra, record=True),
                carry, sw0 + jnp.arange(n_sweeps, dtype=jnp.int32))
        else:
            carry = jax.lax.fori_loop(
                0, n_sweeps,
                lambda i, c: self._sweep_unpacked(
                    c, key, sw0 + i, kT, em_intra, record=False),
                carry)
        (X, energy, em_e,
         quat_t, trans_t, es_body_t, es_b_t, acc_t, _) = carry
        st_out = dict(
            st,
            X=X,
            energy=energy,
            em=em_e,
            quat=jnp.stack(quat_t, axis=1),
            trans=jnp.stack(trans_t, axis=1),
            es_body=jnp.stack(es_body_t, axis=0),
            es_bead=jnp.stack(es_b_t, axis=0),
            n_acc=jnp.stack(acc_t, axis=0),
            sweep=sw0 + n_sweeps,
        )
        return (st_out, decisions) if record else st_out

    def sweep(self, st):
        st_out, dec = self.run(st, 1, record=True)
        return st_out, dec[0]

    def total(self, st):
        return st['energy'] + self.s.constant

    def dofs(self, st):
        """Chain-last degrees of freedom (device arrays, jittable)."""
        s = self.s
        return dict(
            quat=st['quat'],
            trans=st['trans'],
            bead=st['X'][:, self.bead_off:self.bead_off + s.nbead],
            nuis=st['nuis'],
        )
