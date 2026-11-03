"""Scoring terms. A term covers one kind of IMP restraint (possibly many
restraints of that kind) and computes, chain-last:

    full(st) -> (C,) energy per chain, st = dict(X=(3, rows, C),
                quat=(4, nb, C), nuis=(N, C))

(delta() for single moves comes with the engine.) Each term remembers the
IMP restraints it replaces, with their parent weights, so cpp() gives the
C++ value of the same restraints for checks.

Extraction protocol (see system.extract):
    key = Cls.claims(restraint, system)   # None: not mine
    term = Cls(key); term.add(system, parent_weight, restraint)  # register
    term.finalize(system)                  # after the row layout is fixed
"""
import numpy as np


class Term:
    name = 'term'

    def __init__(self, key):
        self.key = key
        self.restraints = []        # (parent weight, IMP restraint)

    @classmethod
    def claims(cls, d, s):
        return None

    def add(self, s, w, d):
        raise NotImplementedError

    def finalize(self, s):
        pass

    def full(self, st):
        raise NotImplementedError

    def cpp(self):
        return sum(w * d.evaluate(False) for w, d in self.restraints)

    @property
    def label(self):
        return self.name if self.key in (None, self.name) else \
            f'{self.name}:{self.key}'


class ImpJax(Term):
    """Any restraint IMP itself can score in JAX (Restraint._get_jax), as a
       full rescore. Used for nuisance priors, and as a fallback for moving
       restraints without a jax_mc term (no delta: slow but exact)."""
    name = 'imp_jax'
    PRIORS = ('UniformPrior', 'JeffreysRestraint')

    @classmethod
    def claims(cls, d, s):
        if type(d).__name__ in cls.PRIORS:
            return 'priors'
        return None

    @classmethod
    def claims_fallback(cls, d, s):
        return hasattr(d, '_get_jax')

    def add(self, s, w, d):
        import IMP
        import IMP.isd
        import IMP.jax
        from .. import imp_hooks
        info = d._get_jax(IMP.jax.FreeSpace)
        for p in imp_hooks.input_particles(d):
            if IMP.isd.Scale.get_is_setup(d.get_model(),
                                          IMP.ParticleIndex(p)):
                s.nuisance(p)
            else:
                s.row(p)
        self.restraints.append((w, d))
        self._infos = getattr(self, '_infos', []) + [(w, info)]

    def finalize(self, s):
        import IMP.isd
        import IMP.jax
        n_attr = max(int(s.row_pi.max(initial=0)),
                     int(s.nuis_pi.max(initial=0))) + 1
        rmap = np.full(n_attr, s.n_rows, np.int32)
        rmap[s.row_pi] = np.arange(s.n_rows, dtype=np.int32)
        nmap = np.full(n_attr, s.n_nuis, np.int32)
        nmap[s.nuis_pi] = np.arange(s.n_nuis, dtype=np.int32)
        dummy = np.zeros(n_attr)
        self._rremap = IMP.jax._Remap(dummy, s.row_pi, rmap)
        self._nremap = IMP.jax._Remap(dummy, s.nuis_pi, nmap)
        self._radius = np.concatenate([s.row_radius, [np.inf]])
        self._scale_key = IMP.isd.Scale.get_scale_key().get_string()
        for w, info in self._infos:
            for k in info._keys:
                ks = k.get_string() if hasattr(k, 'get_string') else str(k)
                if ks != self._scale_key:
                    from ..system import Unsupported
                    raise Unsupported(f'IMP JAX key {ks}')

    def full(self, st):
        import jax
        import jax.numpy as jnp
        import IMP.jax
        X, nuis = st['X'], st['nuis']

        def one(xc, nc):
            pad3 = jnp.full((1, 3), jnp.inf, xc.dtype)
            jm = IMP.jax._JAXModel(
                xyz=IMP.jax._CompactArray(
                    jnp.concatenate([xc.T, pad3]), self._rremap),
                r=IMP.jax._CompactArray(jnp.asarray(self._radius, xc.dtype),
                                        self._rremap))
            jm[self._scale_key] = IMP.jax._CompactArray(
                jnp.concatenate([nc, jnp.full(1, jnp.inf, xc.dtype)]),
                self._nremap)
            return sum(w * info.score_func(jm) for w, info in self._infos)
        return jax.vmap(one, in_axes=(2, 1))(X, nuis)


def _registry():
    from . import basic, em, npc, pm
    return [basic.Pairs, basic.SoftSphereEV, basic.CrossLinks, em.EMIsd,
            em.EMBayes, npc.ZAxial, npc.MembraneSurface, *pm.TERMS, ImpJax]


class _Lazy(list):
    def __iter__(self):
        if not len(self):
            self.extend(_registry())
        return super().__iter__()


REGISTRY = _Lazy()
