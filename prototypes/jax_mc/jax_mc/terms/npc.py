"""IMP.npc restraints used by the NPC basket: z-axial position and membrane
surface location (math as IMP::npc, consider_radius off as PMI sets it)."""
import numpy as np
import jax.numpy as jnp

from . import Term
from .. import imp_hooks


class _PerParticle(Term):
    CLS = None

    @classmethod
    def claims(cls, d, s):
        import IMP.npc
        return cls.name if isinstance(d, getattr(IMP.npc, cls.CLS)) else None

    def add(self, s, w, d):
        info = imp_hooks.info_floats(d)
        ps = imp_hooks.input_particles(d)
        for p in ps:
            s.row(p)
        ww = w * d.get_weight() / info['sigma']
        self._items = getattr(self, '_items', []) + [
            (p, ww, self.params(info)) for p in ps]
        self.restraints.append((w, d))

    def finalize(self, s):
        self.rows = s.rows([p for p, _, _ in self._items])
        self.w = np.array([w for _, w, _ in self._items])
        self.par = np.array([q for _, _, q in self._items], np.float64)


class ZAxial(_PerParticle):
    name = 'zaxial'
    CLS = 'ZAxialPositionRestraint'

    @staticmethod
    def params(info):
        return (info['lower bound'], info['upper bound'])

    def full(self, st):
        z = st['X'][2, self.rows]                       # (n, C)
        lo = jnp.minimum(z - self.par[:, 0, None], 0.)
        up = jnp.maximum(z - self.par[:, 1, None], 0.)
        return jnp.sum(self.w[:, None] * (lo * lo + up * up), axis=0)


class MembraneSurface(_PerParticle):
    name = 'membrane_surface'
    CLS = 'MembraneSurfaceLocationRestraint'

    @staticmethod
    def params(info):
        return (info['major radius'], info['minor radius'],
                info['thickness'])

    def full(self, st):
        x, y, z = st['X'][:, self.rows]
        R, r, h = (self.par[:, i, None] for i in range(3))
        dxy = jnp.sqrt(x * x + y * y)
        safe = jnp.where(dxy > 1e-9, dxy, 1.)
        dtx = jnp.where(dxy > 1e-9, x - x / safe * R, x - R)
        dty = jnp.where(dxy > 1e-9, y - y / safe * R, y)
        inside = jnp.sqrt(z * z + dtx * dtx + dty * dty) - r
        outside = jnp.abs(z) - r
        d = jnp.where(dxy < R, inside, outside)
        return jnp.sum(self.w[:, None] * jnp.where(jnp.abs(d) > h, d * d, 0.),
                       axis=0)
