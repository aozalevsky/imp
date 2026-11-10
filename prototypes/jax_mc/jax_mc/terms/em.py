"""Gaussian EM restraints: IMP.isd.GaussianEMRestraint and
IMP.bayesianem.GaussianEMRestraint.

Model Gaussians on moving rigid bodies get their covariance from the body
orientation (cov_mode 'rotate') or keep the global covariance IMP had at
extraction ('frozen', which is what C++ scores while IMP leaves the global
covariance unchanged on rigid-body moves). Other model Gaussians keep their
global covariance.

Both restraints keep close-pair lists with slack in C++, so pairs a little
beyond the cutoff may or may not count there, depending on the list's
history (salilab/imp#1121). `extra` widens the cutoff here.

For the incremental engine both terms cache their pair sums per chain
(isd: MD, MM cross and the slope sum; bayesianem: every Gaussian's overlaps
with the densities and their compensated sums) and update them with the
moved Gaussians' change.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

from . import Term, geom, pad, slots
from .. import imp_hooks
from ..system import STATIC, Unsupported


def _gaussians(s, pis, cov_mode):
    """owner body (-1: fixed), local cov6 (rotating owner), global cov6"""
    import IMP
    import IMP.atom
    import IMP.core
    m = s.m
    own, cloc, cglob, mass, rad = [], [], [], [], []
    for p in pis:
        pp = IMP.ParticleIndex(int(p))
        g = IMP.core.Gaussian(m, pp)
        var = np.asarray(g.get_variances())
        cg = np.asarray(g.get_global_covariance())
        cglob.append(cg)
        b = s._member_body.get(int(p))
        if b is not None and cov_mode == 'rotate':
            if not IMP.core.RigidMember.get_is_setup(m, pp):
                raise Unsupported('EM Gaussian as a non-rigid member')
            rot = IMP.core.RigidMember(m, pp).get_internal_transformation() \
                .get_rotation()
            R = np.array([rot.get_rotation_matrix_row(i) for i in range(3)])
            cloc.append(R @ np.diag(var) @ R.T)
            own.append(s.body_index[b])
        else:
            cloc.append(cg)
            own.append(-1)
        mass.append(IMP.atom.Mass(m, pp).get_mass())
        rad.append(IMP.core.XYZR(m, pp).get_radius())
    c6 = lambda a: np.stack([a[:, 0, 0], a[:, 0, 1], a[:, 0, 2],  # noqa
                             a[:, 1, 1], a[:, 1, 2], a[:, 2, 2]])
    return (np.array(own), c6(np.array(cloc)), c6(np.array(cglob)),
            np.array(mass), np.array(rad))


def _data(m, pis):
    import IMP
    import IMP.atom
    import IMP.core
    xyz, cov, mass, rad = [], [], [], []
    for p in pis:
        pp = IMP.ParticleIndex(int(p))
        xyz.append(IMP.core.XYZ(m, pp).get_coordinates())
        cov.append(np.asarray(IMP.core.Gaussian(m, pp)
                              .get_global_covariance()))
        mass.append(IMP.atom.Mass(m, pp).get_mass())
        rad.append(IMP.core.XYZR(m, pp).get_radius())
    cov = np.array(cov)
    c6 = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1],
                   cov[:, 1, 2], cov[:, 2, 2]])
    return np.array(xyz).T, c6, np.array(mass), np.array(rad)


class _EM(Term):
    TAG = None
    SPLIT = True

    @classmethod
    def claims(cls, d, s):
        try:
            tag = imp_hooks.recorded(d)[0]
        except KeyError:
            return None
        return int(d.this) if tag == cls.TAG else None

    def cpp_masks(self):
        """(model x data, model x model) pairs in the C++ restraint's
           close-pair containers right now (after it was evaluated)"""
        import IMP.container
        (_, d), = self.restraints
        im = {int(p): i for i, p in enumerate(self.model_pis)}
        idn = {int(p): i for i, p in enumerate(self.dens_pis)}
        md = np.zeros((len(im), len(idn)), bool)
        mm = np.zeros((len(im), len(im)), bool)
        for o in d.get_inputs():
            tn = o.get_type_name()
            if tn.startswith('CloseBipartitePairContainer'):
                c = IMP.container.CloseBipartitePairContainer.get_from(o)
                for a, b in c.get_indexes():
                    md[im[int(a)], idn[int(b)]] = True
            elif tn.startswith('ClosePairContainer'):
                c = IMP.container.ClosePairContainer.get_from(o)
                for a, b in c.get_indexes():
                    i, j = sorted((im[int(a)], im[int(b)]))
                    mm[i, j] = True
        return md, mm

    @staticmethod
    def _cov(st, owner, cloc, cglob):
        """(6, n, C) covariances: rotating with the owner body, or fixed"""
        C = st['X'].shape[-1]
        fixed = jnp.broadcast_to(jnp.asarray(cglob)[..., None],
                                 cglob.shape + (C,))
        if not np.any(owner >= 0):
            return fixed
        R = geom.quat_to_rot(st['quat'])[:, :, np.maximum(owner, 0)]
        rot = geom.rotate_cov6(R, jnp.asarray(cloc)[..., None])
        return jnp.where((owner >= 0)[None, :, None], rot, fixed)

    def _model_cov(self, st):
        return self._cov(st, self.owner, self.cloc, self.cglob)

    def _covs(self, q, st, mv=None, mbslot=None):
        """covariances (6, len(q), C or 1) of model Gaussians q, before the
           move mv or after it (mbslot: per Gaussian, the slot of its owner
           among the mover's bodies, -1: not moved)"""
        fixed = jnp.asarray(self.cglob)[:, q][..., None]
        if not np.any(self.owner >= 0):
            return fixed
        own = jnp.asarray(self.owner)[q]
        quat = st['quat'][:, jnp.maximum(own, 0)]
        if mv is not None and mv.q_new is not None:
            bs = mbslot[q]
            quat = jnp.where((bs >= 0)[None, :, None],
                             mv.q_new[:, jnp.maximum(bs, 0)], quat)
        rot = geom.rotate_cov6(geom.quat_to_rot(quat),
                               jnp.asarray(self.cloc)[:, q][..., None])
        return jnp.where((own >= 0)[None, :, None], rot, fixed)

    def _md(self, st):
        """model x data overlaps, distances (n, D, C)"""
        xm = st['X'][:, self.rows]                          # (3, n, C)
        cm = self._model_cov(st)
        ov = geom.gaussian_overlap(
            xm[:, :, None], cm[:, :, None], self.mass[:, None, None],
            self.dx[:, None, :, None], self.dc[:, None, :, None],
            self.dm[None, :, None])
        v = xm[:, :, None] - self.dx[:, None, :, None]
        d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        return xm, cm, ov, d2

    def touched(self, rows):
        """the mover's model Gaussians (at least 1 if it moves anything
           else the term reads)"""
        return max(int(np.isin(self.rows, rows).sum()),
                   int(np.isin(self.rows_read, rows).any()))

    def move_tables(self, s, moves):
        """the mover's model Gaussians (a: index, aslot: slot), and per
           model Gaussian its slot (mslot) and its owner's slot among the
           mover's bodies (mbslot), -1 if not moved"""
        n = len(self.rows)
        a_l, as_l, ms_l, mb_l = [], [], [], []
        for rows, bodies in moves:
            slot = slots(rows)
            ms = np.array([slot.get(int(r), -1) for r in self.rows])
            a = np.nonzero(ms >= 0)[0]
            bpos = {int(b): k for k, b in enumerate(bodies)}
            a_l.append(a)
            as_l.append(ms[a])
            ms_l.append(ms)
            mb_l.append([bpos.get(int(o), -1) if o >= 0 else -1
                         for o in self.owner])
        return dict(a=pad(a_l, n), aslot=pad(as_l, 0),
                    mslot=pad(ms_l, -1, n), mbslot=pad(mb_l, -1, n))

    def _moved(self, tab, st, mv):
        """positions and covariances of the moved model Gaussians before
           and after the move"""
        n = len(self.rows)
        a = tab['a']
        aa = jnp.minimum(a, n - 1)
        sl = tab['aslot']
        return (a < n, aa, mv.Xo[:, sl], mv.Xn[:, sl], self._covs(aa, st),
                self._covs(aa, st, mv, tab['mbslot']))


class EMIsd(_EM):
    """isd.GaussianEMRestraint(m, model_ps, density_ps, ...):
       w * (-log(2 MD / (MM_self + 2 MM_cross + DD)) + slope * sum |m - d|)

       MD and MM_cross use current positions (pairs within the cutoffs);
       MM_self and DD are what C++ cached at construction. Either set may
       move: the mature NPC protocol passes (data, model), so there the
       moving "density" is the model and DD is its setup-time overlap."""
    name = 'em_isd'
    TAG = 'IMP.isd.GaussianEMRestraint'
    ARGS = ('model', 'model_ps', 'density_ps', 'global_sigma',
            'model_cutoff_dist', 'density_cutoff_dist', 'slope',
            'update_model', 'backbone_slope', 'local')

    def __init__(self, key, extra=0.):
        super().__init__(key)
        self.extra = extra

    def add(self, s, w, d):
        _, args, kw = imp_hooks.recorded(d)
        a = dict(zip(self.ARGS, args), **kw)
        if a.get('backbone_slope') or a.get('local'):
            raise Unsupported('isd EM backbone_slope / local')
        self.model_pis = imp_hooks.pis(a['model_ps'])
        self.dens_pis = imp_hooks.pis(a['density_ps'])
        for p in self.model_pis + self.dens_pis:
            s.row(p)
        self.mcut = a['model_cutoff_dist'] + self.extra
        self.dcut = a['density_cutoff_dist'] + self.extra
        self.slope = a['slope']
        self.w = w * d.get_weight()
        self.snap = imp_hooks.SNAPSHOTS[int(d.this)]
        self.restraints.append((w, d))

    def finalize(self, s):
        self.rows = s.rows(self.model_pis)
        self.drows = s.rows(self.dens_pis)
        (self.owner, self.cloc, self.cglob, self.mass,
         self.rad) = _gaussians(s, self.model_pis, s.cov_mode)
        (self.downer, self.dcloc, self.dcglob, self.dm,
         self.drad) = _gaussians(s, self.dens_pis, s.cov_mode)

        def snap(ps):
            x = np.array([self.snap[p][0] for p in ps]).T
            c = np.array([self.snap[p][1] for p in ps])
            c6 = np.stack([c[:, 0, 0], c[:, 0, 1], c[:, 0, 2], c[:, 1, 1],
                           c[:, 1, 2], c[:, 2, 2]])
            return x, c6, np.array([self.snap[p][2] for p in ps])
        dx, dc, dm = snap(self.dens_pis)
        self.dd = float(np.sum(geom.gaussian_overlap(
            dx[:, :, None], dc[:, :, None], dm[:, None],
            dx[:, None], dc[:, None], dm[None], np)))
        # self terms depend on det(cov) only, i.e. not on orientation
        mx, mc, mm = snap(self.model_pis)
        self.self_mm = float(np.sum(geom.gaussian_overlap(
            mx, mc, mm, mx, mc, mm, np)))
        n = len(self.model_pis)
        self.iu = np.triu(np.ones((n, n), bool), 1)
        self.rows_read = np.unique(np.concatenate([self.rows, self.drows]))
        self.nuis_read = np.zeros(0, np.int64)
        # static density particles: one copy for all chains (delta)
        self.dx0 = s.row_xyz0[self.drows].T if np.all(
            s.row_kind[self.drows] == STATIC) else None

    def _sums(self, st, masks=None):
        """MD, MM cross, slope sum (C,) each. masks: (md (n, D, C),
           mm (n, n, C)) pair masks to use instead of the cutoffs (from
           cpp_masks, for exact checks)"""
        X = st['X']
        xm, xd = X[:, self.rows], X[:, self.drows]
        cm = self._cov(st, self.owner, self.cloc, self.cglob)
        cd = self._cov(st, self.downer, self.dcloc, self.dcglob)
        ov = geom.gaussian_overlap(
            xm[:, :, None], cm[:, :, None], self.mass[:, None, None],
            xd[:, None], cd[:, None], self.dm[None, :, None])
        dist = geom.norm(xm[:, :, None] - xd[:, None])
        sph = dist - self.rad[:, None, None] - self.drad[None, :, None]
        mdm = sph < self.dcut if masks is None else masks[0]
        md = jnp.sum(jnp.where(mdm, ov, 0.), axis=(0, 1))
        mm = geom.gaussian_overlap(
            xm[:, :, None], cm[:, :, None], self.mass[:, None, None],
            xm[:, None], cm[:, None], self.mass[None, :, None])
        dmm = geom.norm(xm[:, :, None] - xm[:, None])
        sphm = dmm - (self.rad[:, None] + self.rad[None])[..., None]
        mmm = (self.iu[..., None] & (sphm < self.mcut)) if masks is None \
            else masks[1]
        mmx = jnp.sum(jnp.where(mmm, mm, 0.), axis=(0, 1))
        sl = jnp.sum(dist, axis=(0, 1)) if self.slope else jnp.zeros_like(md)
        return dict(md=md, mmx=mmx, sl=sl)

    def _energy(self, c):
        e = -jnp.log(2. * c['md'] / (self.self_mm + 2. * c['mmx'] + self.dd))
        if self.slope:
            e = e + self.slope * c['sl']
        return self.w * e

    def full(self, st, masks=None):
        return self._energy(self._sums(st, masks))

    def cache(self, st):
        return self._sums(st)

    def move_tables(self, s, moves):
        for rows, _ in moves:
            if np.isin(self.drows, rows).any():
                raise Unsupported('isd EM with moving density particles '
                                  '(collective template)')
        return super().move_tables(s, moves)

    def delta(self, tab, st, cache, mv):
        """moved Gaussians against the density, and against all model
           Gaussians (pairs among moved ones once each: with frozen
           covariances a rigid move still changes their overlap)"""
        n = len(self.rows)
        ok, aa, xo, xn, co, cn = self._moved(tab, st, mv)
        ma, ra = jnp.asarray(self.mass)[aa], jnp.asarray(self.rad)[aa]
        X = st['X']
        xd = X[:, self.drows] if self.dx0 is None else \
            jnp.asarray(self.dx0)[..., None]
        cd = jnp.asarray(self.dcglob)[..., None]

        def md(x, c):
            ov = geom.gaussian_overlap(
                x[:, :, None], c[:, :, None], ma[:, None, None],
                xd[:, None], cd[:, None], self.dm[None, :, None])
            dist = geom.norm(x[:, :, None] - xd[:, None])
            sph = dist - ra[:, None, None] - self.drad[None, :, None]
            m = ok[:, None, None] & (sph < self.dcut)
            sl = jnp.sum(jnp.where(ok[:, None, None], dist, 0.),
                         axis=(0, 1)) if self.slope else 0.
            return jnp.sum(jnp.where(m, ov, 0.), axis=(0, 1)), sl

        ms = tab['mslot']
        allq = np.arange(n)
        xall_o, xall_n = X[:, self.rows], mv.pos(jnp.asarray(self.rows), ms)
        call_o = self._covs(allq, st)
        call_n = self._covs(allq, st, mv, tab['mbslot'])
        pair = ok[:, None] & ((ms[None] < 0) | (aa[:, None] < allq[None]))

        def mm(x, c, xall, call):
            ov = geom.gaussian_overlap(
                x[:, :, None], c[:, :, None], ma[:, None, None],
                xall[:, None], call[:, None], self.mass[None, :, None])
            sph = geom.norm(x[:, :, None] - xall[:, None]) \
                - (ra[:, None] + self.rad[None])[..., None]
            return jnp.sum(jnp.where(pair[..., None] & (sph < self.mcut),
                                     ov, 0.), axis=(0, 1))

        mdo, slo = md(xo, co)
        mdn, sln = md(xn, cn)
        new = dict(md=cache['md'] + mdn - mdo,
                   mmx=cache['mmx'] + mm(xn, cn, xall_n, call_n)
                   - mm(xo, co, xall_o, call_o),
                   sl=cache['sl'] + sln - slo)
        return self._energy(new) - self._energy(cache), new


class EMBayes(_EM):
    """bayesianem.GaussianEMRestraint (log-normal on per-density overlap
       ratios, with a sigma nuisance and a nearest-center slope term).

       For the incremental engine the cache keeps every model Gaussian's
       overlaps with the densities (ov) and its slope distance (sd), and
       their sums over the Gaussians as compensated sums (geom.xadd). A
       move scores its own Gaussians only, at their new positions; what
       they contributed before is read back from the cache. The sums can't
       be plain running sums: a density with no model nearby has a sum
       near 0 next to single overlaps of order 10, the energy depends on
       log(sum + 1e-10), and in float32 the residue of adding and later
       subtracting an overlap moved it by kT. full() sums from scratch, in
       blocks of BLOCK Gaussians."""
    name = 'em_bayes'
    TAG = 'IMP.bayesianem.GaussianEMRestraint'
    BLOCK = 16

    def __init__(self, key, extra=0.):
        super().__init__(key)
        self.extra = extra

    def add(self, s, w, d):
        import IMP
        import IMP.isd
        m = d.get_model()
        d.debug()
        dens = sorted(int(p) for p in d.get_indexes())
        dset = set(dens)
        model, sigma = [], None
        for p in imp_hooks.input_particles(d):
            if p in dset:
                continue
            if IMP.isd.Scale.get_is_setup(m, IMP.ParticleIndex(p)):
                sigma = p
            else:
                model.append(p)
        self.model_pis, self.dens_pis, self.sigma_pi = model, dens, sigma
        for p in model:
            s.row(p)
        s.nuisance(sigma)
        self.cut = imp_hooks.info_floats(d)['density cutoff'] + self.extra
        self.slope = d.get_slope()
        self.w = w * d.get_weight()
        self.restraints.append((w, d))

    def finalize(self, s):
        # row order, so that a mover's Gaussians share few blocks
        self.model_pis = sorted(self.model_pis, key=lambda p: s.row_of[p])
        self.rows = s.rows(self.model_pis)
        self.sig = s.nuis_of[self.sigma_pi]
        (self.owner, self.cloc, self.cglob, self.mass,
         self.rad) = _gaussians(s, self.model_pis, s.cov_mode)
        self.dx, self.dc, self.dm, self.drad = _data(s.m, self.dens_pis)
        dd = np.asarray(geom.gaussian_overlap(
            self.dx[:, :, None], self.dc[:, :, None], self.dm[:, None],
            self.dx[:, None], self.dc[:, None], self.dm[None], np))
        self.sdd = dd.sum(axis=1)
        nd = len(self.dens_pis)
        self.cached = (np.sum(np.log(self.sdd)) + (3.0 + nd) / 2.0
                       * math.log(2.0) + 0.5 * math.log(math.pi)
                       - math.lgamma(nd / 2.0))
        n, B = len(self.rows), self.BLOCK
        blk = np.arange(-(-n // B) * B).reshape(-1, B)
        self.blk = np.where(blk < n, blk, n)                # (nblk, B)
        self.rows_read = self.rows
        self.nuis_read = np.array([self.sig], np.int64)

    def _slope(self, x):
        """distance (..., C) from Gaussians x (3, ..., C) to the nearest
           density center. From a barrier'd copy: reducing the same
           distances over the density axis here and over the Gaussian axis
           for the overlaps would keep XLA GPU from fusing either reduction
           with its producer."""
        xb = jax.lax.optimization_barrier(x)
        v = xb[..., None, :] - self.dx.reshape((3,) + (1,) * (x.ndim - 2)
                                               + (-1, 1))
        return jnp.sqrt(jnp.min(v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
                                axis=-2))

    def _blocks(self, q, x, c):
        """per-block sums for blocks of Gaussians q (K, B) (n: padding) at
           x (3, K, B, C) with covariances c (6, K, B, C or 1): MD sums
           (K, D, C) and slope sums (K, C)"""
        n = len(self.rows)
        ok = q < n
        qq = jnp.minimum(q, n - 1)
        ma, ra = jnp.asarray(self.mass)[qq], jnp.asarray(self.rad)[qq]
        dx = self.dx[:, None, None, :, None]
        ov = geom.gaussian_overlap(
            x[..., None, :], c[..., None, :], ma[..., None, None],
            dx, self.dc[:, None, None, :, None], self.dm[:, None])
        v = x[..., None, :] - dx
        sph = jnp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) \
            - ra[..., None, None] - self.drad[:, None]
        m = ok[..., None, None] & (sph < self.cut)
        bs = jnp.sum(jnp.where(m, ov, 0.), axis=1)
        if self.slope > 0.:
            bsl = jnp.sum(jnp.where(ok[..., None], self._slope(x), 0.), axis=1)
        else:
            bsl = jnp.zeros_like(bs[:, 0])
        return bs, bsl

    def _sums(self, st, masks=None):
        """per-density MD sums sdm (D, C) and the slope sum sl (C,), summed
           in blocks. masks: (md (n, D, C), _) model x data pair mask to use
           instead of the cutoff (from cpp_masks, for exact checks; no
           blocks)"""
        if masks is not None:
            xm, cm, ov, d2 = self._md(st)
            sdm = jnp.sum(jnp.where(masks[0], ov, 0.), axis=0)
            sl = jnp.sum(self._slope(xm), axis=0) if self.slope > 0. else \
                jnp.zeros_like(sdm[0])
            return dict(sdm=sdm, sl=sl)
        qq = np.minimum(self.blk, len(self.rows) - 1)
        bs, bsl = self._blocks(self.blk, st['X'][:, self.rows[qq]],
                               self._covs(qq, st))
        return dict(sdm=jnp.sum(bs, axis=0), sl=jnp.sum(bsl, axis=0))

    def _energy(self, c, nuis):
        scale = nuis[self.sig]
        lt = jnp.log(scale * (c['sdm'] + 1e-10) / (self.sdd[:, None] + 1e-10))
        nd = len(self.dens_pis)
        e = nd * 0.5 * jnp.log(jnp.sum(lt * lt, axis=0)) + self.cached
        if self.slope > 0.:
            e = e + self.slope * c['sl']
        return self.w * e

    def full(self, st, masks=None):
        return self._energy(self._sums(st, masks), st['nuis'])

    def cache(self, st):
        q = np.arange(len(self.rows))
        ov, sd = self._blocks(q[:, None], st['X'][:, self.rows][:, :, None],
                              self._covs(q, st)[:, :, None])
        k = geom.limbs(ov.dtype)
        sdm, sl = geom.xsum(ov, k), geom.xsum(sd, k)
        return dict(ov=ov, sd=sd, sdm=sdm[-1], sdm_x=jnp.stack(sdm[:-1]),
                    sl=sl[-1], sl_x=jnp.stack(sl[:-1]))

    def move_tables(self, s, moves):
        """the mover's model Gaussians (a: index, aslot: slot) and, per
           Gaussian, its owner's slot among the mover's bodies (mbslot)"""
        t = super().move_tables(s, moves)
        return dict(a=t['a'], aslot=t['aslot'], mbslot=t['mbslot'])

    def delta(self, tab, st, cache, mv):
        n = len(self.rows)
        a = tab['a']                                        # (K,), n: none
        aa = jnp.minimum(a, n - 1)
        x = mv.Xn[:, tab['aslot']]
        ov_n, sd_n = self._blocks(
            a[:, None], x[:, :, None],
            self._covs(aa, st, mv, tab['mbslot'])[:, :, None])
        ok = a < n
        ov_o = jnp.where(ok[:, None, None], cache['ov'][aa], 0.)
        sd_o = jnp.where(ok[:, None], cache['sd'][aa], 0.)
        sdm = [*cache['sdm_x'], cache['sdm']]
        sl = [*cache['sl_x'], cache['sl']]
        for k in range(a.shape[0]):
            sdm = geom.xadd(geom.xadd(sdm, ov_n[k]), -ov_o[k])
            sl = geom.xadd(geom.xadd(sl, sd_n[k]), -sd_o[k])
        new = dict(sdm=sdm[-1], sdm_x=jnp.stack(sdm[:-1]), sl=sl[-1],
                   sl_x=jnp.stack(sl[:-1]))
        dE = self._energy(new, st['nuis']) - self._energy(cache, st['nuis'])

        def commit(acc):
            out = {k: jnp.where(acc, v, cache[k]) for k, v in new.items()}
            out['ov'] = cache['ov'].at[a].set(jnp.where(acc, ov_n, ov_o),
                                              mode='drop')
            out['sd'] = cache['sd'].at[a].set(jnp.where(acc, sd_n, sd_o),
                                              mode='drop')
            return out
        return dE, commit

    def delta_nuis(self, st, cache, nuis):
        return self._energy(cache, nuis) - self._energy(cache, st['nuis']), \
            None
