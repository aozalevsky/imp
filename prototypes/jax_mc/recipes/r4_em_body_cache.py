"""Recipe R4: Per-body EM cache (model-data sums, inter-body sums, and
invariant intra-body overlap under rotating covariances).

Specialization level:
  Level 1 (Term subclass: `BodyCacheEMIsd(EMIsd)`) and Level 3 (tuple-unpacked
  per-body cache functions used in `few_bodies_beads`).

Preconditions (asserted):
  1. `s.cov_mode == 'rotate'`: model Gaussian covariances rotate with their
     rigid body (for the invariant intra-body overlap cache).
  2. Every model Gaussian belongs to a moving rigid body (`row_kind == MEMBER`
     and `owner >= 0`), with no EM Gaussians on beads.
  3. Density Gaussians are static (`dx0 is not None`).
  4. Movers on bodies are single-body `RigidBodyMover`s (no `TransformMover`).
  5. Supports any number of rigid bodies (`nb >= 1`).

Which sums are subtracted:
  None. Unlike `EMIsd.delta` (which updates running totals via
  `cache + new - old`), R4 stores per-body arrays `mds (nb, C)`, `sl (nb, C)`,
  a constant `intra (C,)` (under `rotate`), and per-body-pair cross overlaps
  `inter (n_pairs, C)`. Moving body `b` replaces row `b` (and the pairs
  touching `b`) and sums over bodies, so no floating-point subtraction of old
  from new ever occurs in the cache.
"""
import numpy as np
import jax.numpy as jnp

from jax_mc import engine as E
from jax_mc import system as S
from jax_mc.terms import em, geom


def check_preconditions(s, emt, require_rotate=True):
    """Assert R4 preconditions on System s and EMIsd term emt."""
    if require_rotate:
        assert s.cov_mode == 'rotate', (
            f'R4 requires cov_mode == "rotate" for invariant intra-body '
            f'overlaps, got "{s.cov_mode}"')
    assert np.all(s.row_kind[emt.rows] == S.MEMBER), (
        'R4 requires all EM model Gaussians to be rigid-body members')
    if s.cov_mode == 'rotate':
        assert np.all(emt.owner >= 0), (
            'R4 requires all EM model Gaussians to have a rotating body owner')
    assert emt.dx0 is not None, (
        'R4 requires static density Gaussians (dx0 is not None)')
    assert not any(mv.kind == 'transform' for mv in s.movers), (
        'R4 does not allow TransformMover on EM bodies')
    assert s.nb >= 1, f'R4 requires nb >= 1, got {s.nb}'


class BodyCacheEMIsd(em.EMIsd):
    """Level-1 specialization of EMIsd using R4 per-body caches (any nb >= 1,
       no running-sum subtraction, intra-body overlaps computed once in
       rotate mode)."""
    REQUIRE_ROTATE = True

    def finalize(self, s):
        super().finalize(s)
        check_preconditions(s, self, require_rotate=self.REQUIRE_ROTATE)
        self.nb = s.nb
        self.cov_mode = s.cov_mode
        self.b_owner = s.row_ref[self.rows]
        # Group model Gaussians by rigid body 0 .. nb - 1
        order = np.argsort(self.b_owner, kind='stable')
        self.rows = self.rows[order]
        self.owner = self.owner[order]
        self.b_owner = self.b_owner[order]
        self.cloc = self.cloc[:, order]
        self.cglob = self.cglob[:, order]
        self.mass = self.mass[order]
        self.rad = self.rad[order]
        n = len(self.rows)
        self.iu = np.triu(np.ones((n, n), bool), 1)
        self.body_mask = np.array([self.b_owner == b for b in range(s.nb)],
                                  bool)                         # (nb, n)
        self.intra_mask = np.array(
            [self.iu & (self.b_owner[:, None] == b)
             & (self.b_owner[None, :] == b) for b in range(s.nb)],
            bool)                                               # (nb, n, n)
        pairs, pair_masks = [], []
        for b1 in range(s.nb):
            for b2 in range(b1 + 1, s.nb):
                pairs.append((b1, b2))
                pair_masks.append(
                    (self.b_owner[:, None] == b1)
                    & (self.b_owner[None, :] == b2))
        self.body_pairs = pairs
        self.inter_mask = (np.array(pair_masks, bool) if pairs
                           else np.zeros((0, n, n), bool))

    def cache(self, st):
        X = st['X']
        xm = X[:, self.rows]
        xd = jnp.asarray(self.dx0)[..., None]
        cm = self._cov(st, self.owner, self.cloc, self.cglob)
        cd = jnp.asarray(self.dcglob)[..., None]
        ov = geom.gaussian_overlap(
            xm[:, :, None], cm[:, :, None], self.mass[:, None, None],
            xd[:, None], cd[:, None], self.dm[None, :, None])
        dist = geom.norm(xm[:, :, None] - xd[:, None])
        sph = dist - self.rad[:, None, None] - self.drad[None, :, None]
        md_per_g = jnp.sum(jnp.where(sph < self.dcut, ov, 0.), axis=1)  # (n,C)
        sl_per_g = (jnp.sum(dist, axis=1) if self.slope
                    else jnp.zeros_like(md_per_g))
        bm = jnp.asarray(self.body_mask)[..., None]             # (nb, n, 1)
        mds = jnp.sum(jnp.where(bm, md_per_g[None], 0.), axis=1)
        sl = jnp.sum(jnp.where(bm, sl_per_g[None], 0.), axis=1)

        mm = geom.gaussian_overlap(
            xm[:, :, None], cm[:, :, None], self.mass[:, None, None],
            xm[:, None], cm[:, None], self.mass[None, :, None])
        dmm = geom.norm(xm[:, :, None] - xm[:, None])
        sphm = dmm - (self.rad[:, None] + self.rad[None])[..., None]
        mm_cut = jnp.where(sphm < self.mcut, mm, 0.)
        im = jnp.asarray(self.intra_mask)[..., None]            # (nb, n, n, 1)
        intra = jnp.sum(jnp.where(im, mm_cut[None], 0.), axis=(1, 2))
        if len(self.body_pairs):
            pm = jnp.asarray(self.inter_mask)[..., None]        # (P, n, n, 1)
            inter = jnp.sum(jnp.where(pm, mm_cut[None], 0.), axis=(1, 2))
        else:
            inter = jnp.zeros((0, X.shape[-1]), X.dtype)
        return dict(
            mds=mds, sl=sl, intra=intra, inter=inter,
            md=jnp.sum(mds, axis=0),
            mmx=jnp.sum(intra, axis=0) + jnp.sum(inter, axis=0),
            sl_tot=jnp.sum(sl, axis=0),
        )

    def _energy_from_cache(self, c):
        e = -jnp.log(2. * c['md'] / (self.self_mm + 2. * c['mmx'] + self.dd))
        if self.slope:
            e = e + self.slope * c['sl_tot']
        return self.w * e

    def move_tables(self, s, moves):
        t = super().move_tables(s, moves)
        bodies = [b[0] if len(b) == 1 else -1 for _, b in moves]
        t['body'] = np.array(bodies, np.int64)
        return t

    def delta(self, tab, st, cache, mv):
        n = len(self.rows)
        ok, aa, _, xn, _, cn = self._moved(tab, st, mv)
        b = jnp.maximum(tab['body'], 0)
        ma, ra = jnp.asarray(self.mass)[aa], jnp.asarray(self.rad)[aa]
        xd = jnp.asarray(self.dx0)[..., None]
        cd = jnp.asarray(self.dcglob)[..., None]
        ov = geom.gaussian_overlap(
            xn[:, :, None], cn[:, :, None], ma[:, None, None],
            xd[:, None], cd[:, None], self.dm[None, :, None])
        dist = geom.norm(xn[:, :, None] - xd[:, None])
        sph = dist - ra[:, None, None] - self.drad[None, :, None]
        m = ok[:, None, None] & (sph < self.dcut)
        md_b = jnp.sum(jnp.where(m, ov, 0.), axis=(0, 1))
        sl_b = (jnp.sum(jnp.where(ok[:, None, None], dist, 0.), axis=(0, 1))
                if self.slope else jnp.zeros_like(md_b))
        mds = cache['mds'].at[b].set(md_b)
        sl = cache['sl'].at[b].set(sl_b)

        # Inter-body pairs: moved Gaussians against unmoved Gaussians (ms < 0)
        ms = tab['mslot']
        allq = np.arange(n)
        xall_o = st['X'][:, self.rows]
        call_o = self._covs(allq, st)
        ov_mm = geom.gaussian_overlap(
            xn[:, :, None], cn[:, :, None], ma[:, None, None],
            xall_o[:, None], call_o[:, None], self.mass[None, :, None])
        sph_mm = geom.norm(xn[:, :, None] - xall_o[:, None]) \
            - (ra[:, None] + self.rad[None])[..., None]
        mm_val = jnp.where((sph_mm < self.mcut) & ok[:, None, None], ov_mm, 0.)
        if len(self.body_pairs):
            inter = cache['inter']
            b_own = jnp.asarray(self.b_owner)
            for p_idx, (b1, b2) in enumerate(self.body_pairs):
                touch = (b == b1) | (b == b2)
                other = jnp.where(b == b1, b2, b1)
                mask_o = (ms < 0) & (b_own == other)
                val_p = jnp.sum(
                    jnp.where(mask_o[None, :, None], mm_val, 0.), axis=(0, 1))
                inter = inter.at[p_idx].set(
                    jnp.where(touch, val_p, inter[p_idx]))
        else:
            inter = cache['inter']

        if self.cov_mode == 'frozen':
            # Recompute intra-body overlaps for moved body b
            ov_ii = geom.gaussian_overlap(
                xn[:, :, None], cn[:, :, None], ma[:, None, None],
                xn[:, None], cn[:, None], ma[None, :, None])
            sph_ii = geom.norm(xn[:, :, None] - xn[:, None]) \
                - (ra[:, None] + ra[None])[..., None]
            pair_ii = ok[:, None] & ok[None, :] & (aa[:, None] < aa[None, :])
            intra_b = jnp.sum(
                jnp.where(pair_ii[..., None] & (sph_ii < self.mcut),
                          ov_ii, 0.),
                axis=(0, 1))
            intra = cache['intra'].at[b].set(intra_b)
        else:
            # Under cov_mode == 'rotate', intra-body overlaps are invariant!
            intra = cache['intra']

        new = dict(
            mds=mds, sl=sl, intra=intra, inter=inter,
            md=jnp.sum(mds, axis=0),
            mmx=jnp.sum(intra, axis=0) + jnp.sum(inter, axis=0),
            sl_tot=jnp.sum(sl, axis=0),
        )
        dE = self._energy_from_cache(new) - self._energy_from_cache(cache)
        return dE, new


def replace_em(s):
    """System s rebuilt with BodyCacheEMIsd in place of EMIsd, through the
       core's extract(replace=...)"""
    return S.extract(s.source, cov_mode=s.cov_mode,
                     replace={em.EMIsd: BodyCacheEMIsd})


def factory(s):
    """Engine with EMIsd replaced by BodyCacheEMIsd (for jax_mc.check and
       jax_mc.bench)."""
    return E.Engine(replace_em(s))
