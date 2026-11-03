"""Restraints of the mature NPC refinement (PM_NPC_Assembly): bounding box,
membrane binding and depth (npc.SlabWithToroidalPore), overall position.
Math as pm_jax.py (validated against C++ there)."""
import numpy as np
import jax.numpy as jnp

from . import RowTerm, geom
from .. import imp_hooks
from ..system import Unsupported


class BoundingBox(RowTerm):
    """container.SingletonsRestraint over a BoundingBox3DSingletonScore with
       a HarmonicUpperBound"""
    name = 'bbox'

    @classmethod
    def claims(cls, d, s):
        import IMP.container
        import IMP.core
        if isinstance(d, IMP.container.SingletonsRestraint):
            so = d.get_score_object().get_derived_object()
            if isinstance(so, IMP.core.BoundingBox3DSingletonScore):
                return int(d.this)
        return None

    def add(self, s, w, d):
        import IMP.core
        so = d.get_score_object().get_derived_object()
        uf = imp_hooks.recorded(so)[1][0].get_derived_object()
        if not isinstance(uf, IMP.core.HarmonicUpperBound):
            raise Unsupported('bounding box with ' + type(uf).__name__)
        bb = so.get_bounding_box()
        self.lo = np.array(bb.get_corner(0))
        self.hi = np.array(bb.get_corner(1))
        self.mean, self.k = uf.get_mean(), uf.get_k()
        self.ps = [int(p) for p in d.get_container().get_contents()]
        for p in self.ps:
            s.row(p)
        self.w = w * d.get_weight()
        self.restraints.append((w, d))

    def finalize(self, s):
        self.rows = s.rows(self.ps)

    def energies(self, q, P):
        cp = jnp.clip(P, self.lo[:, None, None], self.hi[:, None, None])
        t = jnp.maximum(geom.norm(P - cp) - self.mean, 0.)
        return self.w * (0.5 * self.k * t * t)


class _Membrane(RowTerm):
    """container.PairsRestraint over AllBipartitePairContainer(membrane,
       particles) with an npc membrane score"""
    SCORE = None

    @classmethod
    def claims(cls, d, s):
        import IMP.container
        import IMP.npc
        if isinstance(d, IMP.container.PairsRestraint):
            so = d.get_score_object()
            try:
                so = so.get_derived_object()
            except Exception:
                pass
            if type(so).__name__ == cls.SCORE:
                return int(d.this)
        return None

    def add(self, s, w, d):
        import IMP
        import IMP.npc
        so = d.get_score_object()
        self.par = imp_hooks.recorded(so)[1]
        pairs = [(int(a), int(b)) for a, b in d.get_container().get_contents()]
        mem = {a for a, _ in pairs}
        if len(mem) != 1:
            raise Unsupported('membrane restraint with several membranes')
        g = IMP.npc.SlabWithToroidalPore(d.get_model(),
                                         IMP.ParticleIndex(mem.pop()))
        self.R, self.h = g.get_pore_radius(), g.get_thickness()
        self.rmin = g.get_vertical_minor_radius()
        self.ps = [b for _, b in pairs]
        for p in self.ps:
            s.row(p)
        self.w = w * d.get_weight()
        self.restraints.append((w, d))

    def finalize(self, s):
        self.rows = s.rows(self.ps)

    def _geom(self, xyz):
        x, y, z = xyz
        rxy = x * x + y * y
        r = jnp.maximum(jnp.sqrt(rxy), 1e-9)
        rscl = self.R / r
        dist = jnp.sqrt((x - rscl * x) ** 2 + (y - rscl * y) ** 2 + z * z)
        return z, rxy, dist


class MembraneBinding(_Membrane):
    name = 'membrane_binding'
    SCORE = 'SlabWithToroidalPoreMBMScore'

    def energies(self, q, xyz):
        z, rxy, dist = self._geom(xyz)
        x0u, x0l, k = self.par
        h, rmin = self.h, self.rmin
        zu = jnp.where(z > 0., 0.5 * h + x0u, -0.5 * h - x0u)
        zl = jnp.where(z > 0., 0.5 * h + x0l, -0.5 * h - x0l)
        up = jnp.where(z > 0., z > zu, z < zu)
        lw = jnp.where(z > 0., z < zl, z > zl)
        out_s = jnp.where(up, 0.5 * k * (z - zu) ** 2,
                          jnp.where(lw, 0.5 * k * (z - zl) ** 2, 0.))
        du, dl = rmin + x0u, rmin + x0l
        in_s = jnp.where(dist > du, 0.5 * k * (dist - du) ** 2,
                         jnp.where(dist < dl, 0.5 * k * (dist - dl) ** 2,
                                   0.))
        return self.w * jnp.where(rxy > self.R * self.R, out_s, in_s)


class MembraneDepth(_Membrane):
    name = 'membrane_depth'
    SCORE = 'ToroidalPoreSurfaceDepthPairScore'

    def energies(self, q, xyz):
        z, rxy, dist = self._geom(xyz)
        (k,) = self.par
        h, rmin = self.h, self.rmin
        inside_slab = (z <= 0.5 * h) & (z >= -0.5 * h)
        z0 = jnp.where(z >= 0., 0.5 * h, -0.5 * h)
        out_s = jnp.where(inside_slab, 0.5 * k * (z - z0) ** 2, 0.)
        in_s = jnp.where(dist < rmin, 0.5 * k * (dist - rmin) ** 2, 0.)
        return self.w * jnp.where(rxy > self.R * self.R, out_s, in_s)


class OverallPosition(RowTerm):
    """npc.OverallPositionRestraint (one particle each, consider_radius
       off): w / sigma * max(|x - x0| - tolerance, 0)^2"""
    name = 'overall_position'

    @classmethod
    def claims(cls, d, s):
        import IMP.npc
        if isinstance(d, IMP.npc.OverallPositionRestraint):
            return 'overall_position'
        return None

    def add(self, s, w, d):
        info = imp_hooks.info_floats(d)
        ps = imp_hooks.input_particles(d)
        if len(ps) != 1:
            raise Unsupported('OverallPositionRestraint on several particles')
        s.row(ps[0])
        self._items = getattr(self, '_items', []) + [(
            ps[0], info['x start'], info['y start'], info['z start'],
            info['tolerance'], w * d.get_weight() / info['sigma'])]
        self.restraints.append((w, d))

    def finalize(self, s):
        a = np.array(self._items, np.float64)
        self.rows = s.rows(a[:, 0].astype(np.int64))
        self.x0 = a[:, 1:4].T
        self.tol, self.wsig = a[:, 4], a[:, 5]

    def energies(self, q, x):
        g = lambda a: jnp.asarray(a)[..., q]                # noqa
        diff = geom.norm(x - g(self.x0)[:, :, None]) - g(self.tol)[:, None]
        return g(self.wsig)[:, None] * jnp.where(diff > 0, diff * diff, 0.)


TERMS = [BoundingBox, MembraneBinding, MembraneDepth, OverallPosition]
