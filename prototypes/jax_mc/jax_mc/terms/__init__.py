"""Scoring terms. A term covers one kind of IMP restraint (possibly many
restraints of that kind) and computes, chain-last:

    full(st) -> (C,) energy per chain, st = dict(X=(3, rows, C),
                quat=(4, nb, C), nuis=(N, C))

and, for the incremental engine (jax_mc.engine):

    cache(st)                       per-chain partial sums the term keeps
                                    (None if it keeps none)
    touched(rows)                   how many of its items a mover of these
                                    rows changes (0: none). With SPLIT, the
                                    engine groups movers by it, so a mover
                                    is padded to its own size class and
                                    skips the term where it is 0
    move_tables(s, moves)           host tables, one row per mover of a group
    delta(tab, st, cache, mv)       (dE (C,), new) for a coordinate move mv
                                    (engine.Move); new is None (cache
                                    unchanged), the new cache (the engine
                                    keeps it where the move is accepted) or
                                    commit(acc) -> cache, for in-place
                                    updates of large caches
    delta_nuis(st, cache, nuis)     the same for a new nuisance vector

rows_read / nuis_read say which rows and nuisances the term reads (None:
all), so the engine skips terms a mover can't change. The defaults for
delta and delta_nuis rescore the term with the move applied: slow but
exact, and only valid for terms without a cache.

Each term remembers the IMP restraints it replaces, with their parent
weights, so cpp() gives the C++ value of the same restraints for checks.

Extraction protocol (see system.extract):
    key = Cls.claims(restraint, system)   # None: not mine
    term = Cls(key); term.add(system, parent_weight, restraint)  # register
    term.finalize(system)                  # after the row layout is fixed
"""
import numpy as np
import jax.numpy as jnp


def pad(lists, fill, width=None, dtype=np.int64):
    """Ragged lists -> (len(lists), width) array padded with fill"""
    w = max([len(x) for x in lists] + [1]) if width is None else width
    out = np.full((len(lists), w), fill, dtype)
    for i, x in enumerate(lists):
        out[i, :len(x)] = x
    return out


def slots(rows):
    """{row: slot} for a mover's rows"""
    return {int(r): k for k, r in enumerate(rows)}


class Term:
    name = 'term'
    rows_read = None            # np array of rows, None: all
    nuis_read = None            # np array of nuisance indexes, None: all
    SPLIT = False               # group movers by touched(): for terms whose
    #                             cost per item matters (every group is one
    #                             more branch to compile)

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

    # ---- incremental interface (jax_mc.engine) ------------------------
    def cache(self, st):
        return None

    def touched(self, rows):
        """number of this term's items that a mover of `rows` changes: the
           width its tables need for that mover"""
        return int(self.rows_read is None
                   or np.isin(self.rows_read, rows).any())

    def move_tables(self, s, moves):
        """moves: [(rows, bodies)] for the movers of one group, rows in
           slot order. Returns {name: array with one row per mover}."""
        return {}

    def delta(self, tab, st, cache, mv):
        assert cache is None, f'{self.label}: cached term needs a delta'
        return self.full(mv.apply(st)) - self.full(st), cache

    def delta_nuis(self, st, cache, nuis):
        assert cache is None, f'{self.label}: cached term needs a delta'
        return self.full(dict(st, nuis=nuis)) - self.full(st), cache

    def cpp(self):
        return sum(w * d.evaluate(False) for w, d in self.restraints)

    @property
    def label(self):
        return self.name if self.key in (None, self.name) else \
            f'{self.name}:{self.key}'


class RowTerm(Term):
    """A sum over items of one particle each (self.rows, one per item).
       Subclasses give energies(q, x) -> (len(q), C) for item indexes q
       (numpy or traced) and positions x (3, len(q), C)."""
    nuis_read = np.zeros(0, np.int64)

    @property
    def rows_read(self):
        return self.rows

    def full(self, st):
        q = np.arange(len(self.rows))
        return jnp.sum(self.energies(q, st['X'][:, self.rows]), axis=0)

    def touched(self, rows):
        return int(np.isin(self.rows, rows).sum())

    def move_tables(self, s, moves):
        items, sl = [], []
        for rows, _ in moves:
            slot = slots(rows)
            q = [i for i, r in enumerate(self.rows) if int(r) in slot]
            items.append(q)
            sl.append([slot[int(self.rows[i])] for i in q])
        return dict(item=pad(items, len(self.rows)), slot=pad(sl, 0))

    def delta(self, tab, st, cache, mv):
        n = len(self.rows)
        q = tab['item']
        qq = jnp.minimum(q, n - 1)
        e = self.energies(qq, mv.Xn[:, tab['slot']]) \
            - self.energies(qq, mv.Xo[:, tab['slot']])
        return jnp.sum(jnp.where((q < n)[:, None], e, 0.), axis=0), cache


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
        self._rp = getattr(self, '_rp', [])
        self._np = getattr(self, '_np', [])
        for p in imp_hooks.input_particles(d):
            if IMP.isd.Scale.get_is_setup(d.get_model(),
                                          IMP.ParticleIndex(p)):
                self._np.append(s.nuisance(p))
            else:
                self._rp.append(s.row(p))
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
        self.rows_read = np.unique(s.rows(self._rp)).astype(np.int64)
        self.nuis_read = np.unique(s.nuis_rows(self._np)).astype(np.int64)
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
        import IMP.jax
        X, nuis = st['X'], st['nuis']
        dt = X.dtype

        def one(xc, nc):
            xyz = jnp.full((xc.shape[1] + 1, 3), jnp.inf, dt) \
                if xc.shape[0] == 0 else \
                jnp.concatenate([xc.T, jnp.full((1, 3), jnp.inf, dt)])
            jm = IMP.jax._JAXModel(
                xyz=IMP.jax._CompactArray(xyz, self._rremap),
                r=IMP.jax._CompactArray(jnp.asarray(self._radius, dt),
                                        self._rremap))
            jm[self._scale_key] = IMP.jax._CompactArray(
                jnp.concatenate([nc, jnp.full(1, jnp.inf, dt)]),
                self._nremap)
            return sum(w * info.score_func(jm) for w, info in self._infos)
        if len(self.rows_read) == 0:        # nuisance priors: no positions
            X = jnp.zeros((0, X.shape[1], X.shape[2]), dt)
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
