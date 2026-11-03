"""Recipe R1: Static/movable split and static EV spatial hash grid.

Specialization level:
  Level 1 (Term subclass: `StaticGridEV(SoftSphereEV)` via
  `system.extract(src, replace={SoftSphereEV: StaticGridEV})`), and shared
  grid primitives (`build_grid`, `grid_ev`) used by Level-3 templates.

Preconditions (asserted):
  1. The EV restraint has both static (`n_static > 0`) and moving
     (`n_static < len(rows)`) particles.
  2. The number of static EV particles fits in `uint16` (`n_static <= 65535`).
  3. Static and moving EV particles have disjoint rigid-body exclusion labels.
"""
import numpy as np
import jax.numpy as jnp

from jax_mc import engine as E
from jax_mc import system as S
from jax_mc.terms import basic


def dist3(ax, ay, az, bx, by, bz):
    """Euclidean distance from separate x, y, z component arrays."""
    dx, dy, dz = ax - bx, ay - by, az - bz
    return jnp.sqrt(dx * dx + dy * dy + dz * dz)


def soft_pair(d, rsum):
    """Unweighted soft-sphere overlap squared: min(d - rsum, 0)^2."""
    x = jnp.minimum(d - rsum, 0.)
    return x * x


def build_grid(P, R, rq, h=6.0):
    """Build a 3D cell table mapping each cell to all static spheres (P, R)
       within reach of any query of radius <= rq in that cell, padded with a
       far-away sentinel at index len(P).

       P: (n_static, 3) float64 coordinates
       R: (n_static,) float64 radii
       rq: maximum radius of any moving query sphere
       h: cell width (A)
    """
    P = np.asarray(P, np.float64)
    R = np.asarray(R, np.float64)
    assert P.ndim == 2 and P.shape[1] == 3 and len(P) > 0, (
        f'R1 build_grid requires non-empty (N, 3) points, got {P.shape}')
    assert len(P) <= np.iinfo(np.uint16).max, (
        f'R1 static EV count {len(P)} exceeds uint16 range')
    assert rq > 0. and float(R.max()) > 0. and h > 0., (
        'R1 requires positive radii and cell width')
    pad = rq + float(R.max()) + h
    lo = P.min(0) - pad
    n = np.ceil((P.max(0) + pad - lo) / h).astype(np.int64)
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'),
                   -1).reshape(-1, 3)
    cen = lo + h * (ijk + 0.5)
    dd = np.maximum(np.abs(cen[:, None, :] - P[None]) - h / 2, 0.)
    # 0.01 A margin: a float32 cell index may be off by one at a cell face
    cand = np.sqrt((dd ** 2).sum(-1)) < rq + R[None] + 0.01
    K = max(int(cand.sum(1).max()), 1)
    table = np.full((len(cen) + 1, K), len(P), np.int32)    # last: outside
    for c in np.nonzero(cand.any(1))[0]:
        j = np.nonzero(cand[c])[0]
        table[c, :len(j)] = j
    P_pad = np.vstack([P, [1e4, 1e4, 1e4]]).T
    R_pad = np.r_[R, 0.]
    return dict(
        lo=lo, h=float(h), n=n, table=table, K=K,
        P=P_pad, R=R_pad,
        table_u16=table.astype(np.uint16),
        PR=np.ascontiguousarray(np.column_stack([P_pad.T, R_pad])),
    )


def grid_ev(g, xq, rq, dt=None, sum_rows=False):
    """Sum over static grid spheres of min(|xq - P| - (rq + R), 0)^2 for
       query positions xq (3, m, C) and radii rq (m,) -> (m, C) or (C,)."""
    if dt is None:
        dt = xq.dtype
    n = g['n']
    idx = jnp.floor((xq - jnp.asarray(g['lo'], dt)[:, None, None]) / g['h'])
    idx = jnp.clip(idx, -1, jnp.asarray(n, dt)[:, None, None]).astype(
        jnp.int32)
    idx0, idx1, idx2 = idx[0], idx[1], idx[2]
    n0, n1, n2 = int(n[0]), int(n[1]), int(n[2])
    inside = ((idx0 >= 0) & (idx0 < n0)
              & (idx1 >= 0) & (idx1 < n1)
              & (idx2 >= 0) & (idx2 < n2))
    flat = (idx0 * n1 + idx1) * n2 + idx2
    flat = jnp.where(inside, flat, len(g['table']) - 1)
    cand = jnp.asarray(g['table_u16'])[flat.astype(jnp.uint32)]   # (m, C, K)
    pr = jnp.asarray(g['PR'], dt)[cand]                       # (m, C, K, 4)
    d = dist3(pr[..., 0], pr[..., 1], pr[..., 2],
              xq[0, ..., None], xq[1, ..., None], xq[2, ..., None])
    e = soft_pair(d, jnp.asarray(rq, dt)[:, None, None] + pr[..., 3])
    return jnp.sum(e, axis=(0, -1) if sum_rows else -1)


class StaticGridEV(basic.SoftSphereEV):
    """Level-1 specialization of SoftSphereEV: scores moved particles against
       the static EV set via a 3D cell grid instead of an O(n_static) scan."""
    GRID_H = 6.0

    def finalize(self, s):
        super().finalize(s)
        self._build_static_grid()

    def _build_static_grid(self):
        ns = self.n_static
        n = len(self.rows)
        assert ns > 0, 'R1 StaticGridEV requires n_static > 0'
        assert ns < n, 'R1 StaticGridEV requires moving EV particles'
        assert ns <= np.iinfo(np.uint16).max, (
            f'R1 static EV count {ns} exceeds uint16 range')
        assert not (set(self.lab[:ns]) & set(self.lab[ns:])), (
            'R1 requires disjoint exclusion labels between static and '
            'moving EV particles')
        rq = float(self.r[ns:].max())
        self.grid = build_grid(self.xs0.T, self.r[:ns], rq, self.GRID_H)

    def delta(self, tab, st, cache, mv):
        n, ns = len(self.rows), self.n_static
        idx = tab['idx']
        ok = (idx < n)[:, None]
        ri = jnp.asarray(self.r)[jnp.minimum(idx, n - 1)]
        sl = tab['slot']
        xn, xo = mv.Xn[:, sl], mv.Xo[:, sl]
        # Moving vs static via R1 grid
        es = jnp.sum(jnp.where(
            ok, grid_ev(self.grid, xn, ri) - grid_ev(self.grid, xo, ri), 0.),
            axis=0)
        # Moving vs moving densely (as in SoftSphereEV.delta)
        xm = st['X'][:, self.rows[ns:]]
        rm = self.r[ns:]
        keep_m = tab['keep_m'][..., None]
        dn = basic.geom.norm(xn[:, :, None] - xm[:, None])
        do = basic.geom.norm(xo[:, :, None] - xm[:, None])
        rsum = (ri[:, None] + rm[None])[..., None]
        em = jnp.sum(jnp.where(
            keep_m, soft_pair(dn, rsum) - soft_pair(do, rsum), 0.),
            axis=(0, 1))
        return 0.5 * self.kk * (es + em), cache


def replace_ev(s):
    """System s rebuilt with StaticGridEV in place of SoftSphereEV, through
       the core's extract(replace=...)"""
    return S.extract(s.source, cov_mode=s.cov_mode,
                     replace={basic.SoftSphereEV: StaticGridEV})


def factory(s):
    """Engine with SoftSphereEV replaced by StaticGridEV (for jax_mc.check
       and jax_mc.bench)."""
    return E.Engine(replace_ev(s))
