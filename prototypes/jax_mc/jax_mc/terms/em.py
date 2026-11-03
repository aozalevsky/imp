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
"""
import math

import numpy as np
import jax.numpy as jnp

from . import Term, geom
from .. import imp_hooks
from ..system import Unsupported


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

    def _model_cov(self, st):
        """(6, n, C) model covariances"""
        C = st['X'].shape[-1]
        fixed = jnp.broadcast_to(jnp.asarray(self.cglob)[..., None],
                                 self.cglob.shape + (C,))
        if not np.any(self.owner >= 0):
            return fixed
        R = geom.quat_to_rot(st['quat'])[:, :, np.maximum(self.owner, 0)]
        rot = geom.rotate_cov6(R, jnp.asarray(self.cloc)[..., None])
        return jnp.where((self.owner >= 0)[None, :, None], rot, fixed)

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


class EMIsd(_EM):
    """isd.GaussianEMRestraint:
       w * (-log(2 MD / (MM_self + 2 MM_cross + DD)) + slope * sum |m - d|)"""
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
        for p in self.model_pis:
            s.row(p)
        self.mcut = a['model_cutoff_dist'] + self.extra
        self.dcut = a['density_cutoff_dist'] + self.extra
        self.slope = a['slope']
        self.w = w * d.get_weight()
        self.restraints.append((w, d))

    def finalize(self, s):
        self.rows = s.rows(self.model_pis)
        (self.owner, self.cloc, self.cglob, self.mass,
         self.rad) = _gaussians(s, self.model_pis, s.cov_mode)
        self.dx, self.dc, self.dm, self.drad = _data(s.m, self.dens_pis)
        dd = geom.gaussian_overlap(
            self.dx[:, :, None], self.dc[:, :, None], self.dm[:, None],
            self.dx[:, None], self.dc[:, None], self.dm[None], np)
        self.dd = float(np.sum(np.asarray(dd)))
        # self terms depend on det(cov) only, i.e. not on orientation
        z = np.zeros((3, len(self.mass)))
        self.self_mm = float(np.sum(np.asarray(geom.gaussian_overlap(
            z, self.cglob, self.mass, z, self.cglob, self.mass, np))))
        n = len(self.model_pis)
        self.iu = np.triu(np.ones((n, n), bool), 1)

    def full(self, st, masks=None):
        """masks: (md (n, D, C), mm (n, n, C)) pair masks to use instead of
           the cutoffs (from cpp_masks, for exact checks)"""
        xm, cm, ov, d2 = self._md(st)
        dist = jnp.sqrt(d2)
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
        denom = self.self_mm + 2. * mmx + self.dd
        return self.w * (-jnp.log(2. * md / denom)
                         + self.slope * jnp.sum(dist, axis=(0, 1)))


class EMBayes(_EM):
    """bayesianem.GaussianEMRestraint (log-normal on per-density overlap
       ratios, with a sigma nuisance and a nearest-center slope term)"""
    name = 'em_bayes'
    TAG = 'IMP.bayesianem.GaussianEMRestraint'

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

    def full(self, st, masks=None):
        """masks: (md (n, D, C), _) model x data pair mask to use instead of
           the cutoff (from cpp_masks, for exact checks)"""
        xm, cm, ov, d2 = self._md(st)
        sph = jnp.sqrt(d2) - self.rad[:, None, None] \
            - self.drad[None, :, None]
        mdm = sph < self.cut if masks is None else masks[0]
        sdm = jnp.sum(jnp.where(mdm, ov, 0.), axis=0)              # (D, C)
        scale = st['nuis'][self.sig]
        lt = jnp.log(scale * (sdm + 1e-10) / (self.sdd[:, None] + 1e-10))
        nd = len(self.dens_pis)
        e = nd * 0.5 * jnp.log(jnp.sum(lt * lt, axis=0)) + self.cached
        if self.slope > 0.:
            e = e + self.slope * jnp.sum(jnp.sqrt(jnp.min(d2, axis=1)),
                                         axis=0)
        return self.w * e
