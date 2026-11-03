"""Pair restraints, soft-sphere excluded volume, cross-links."""
import numpy as np
import jax
import jax.numpy as jnp

from . import Term, geom
from ..system import Unsupported

HUB, HARMONIC, HLB, LINEAR = 0, 1, 2, 3


class Pairs(Term):
    """core.PairRestraint / core.DistanceRestraint with a distance or sphere
       distance score: harmonic upper bound, harmonic, harmonic lower bound
       or linear (PMI connectivity, distance restraints)"""
    name = 'pairs'

    @classmethod
    def _parse(cls, d):
        import IMP.core
        if type(d) not in (IMP.core.PairRestraint, IMP.core.DistanceRestraint):
            return None
        ps = d.get_score_object()
        try:
            ps = ps.get_derived_object()
        except Exception:
            pass
        if isinstance(ps, IMP.core.SphereDistancePairScore):
            ps = ps._proxied
        if isinstance(ps, IMP.core.HarmonicUpperBoundSphereDistancePairScore):
            return (1., ps.get_x0(), ps.get_k(), HUB)
        if isinstance(ps, IMP.core.DistancePairScore):
            uf = ps.get_score_functor().get_unary_function() \
                .get_derived_object()
            mode = {IMP.core.HarmonicUpperBound: HUB,
                    IMP.core.Harmonic: HARMONIC,
                    IMP.core.HarmonicLowerBound: HLB}.get(type(uf))
            if mode is not None:
                return (0., uf.get_mean(), uf.get_k(), mode)
            if isinstance(uf, IMP.core.Linear):
                return (0., uf.get_offset(), uf.get_slope(), LINEAR)
        return None             # not ours: constant or IMP JAX fallback

    @classmethod
    def claims(cls, d, s):
        return 'pairs' if cls._parse(d) is not None else None

    def add(self, s, w, d):
        sph, x0, k, mode = self._parse(d)
        i, j = (int(x) for x in d.get_index())
        s.row(i)
        s.row(j)
        self.restraints.append((w, d))
        self._items = getattr(self, '_items', []) + [
            (i, j, sph, x0, k, w * d.get_weight(), mode)]

    def finalize(self, s):
        a = np.array(self._items, np.float64).reshape(-1, 7)
        self.ri = s.rows(a[:, 0].astype(np.int64))
        self.rj = s.rows(a[:, 1].astype(np.int64))
        rad = s.row_radius
        self.off = a[:, 2] * (rad[self.ri] + rad[self.rj]) + a[:, 3]
        self.k, self.w = a[:, 4], a[:, 5]
        self.mode = a[:, 6].astype(np.int64)

    def energies(self, xi, xj):
        """(n, C) energy per pair from positions (3, n, C)"""
        t = geom.norm(xi - xj) - self.off[:, None]
        mode = self.mode[:, None]
        tc = jnp.where(mode == HUB, jnp.maximum(t, 0.),
                       jnp.where(mode == HLB, jnp.minimum(t, 0.), t))
        e = jnp.where(mode == LINEAR, t, 0.5 * tc * tc)
        return (self.w * self.k)[:, None] * e

    def full(self, st):
        X = st['X']
        return jnp.sum(self.energies(X[:, self.ri], X[:, self.rj]), axis=0)


class SoftSphereEV(Term):
    """SoftSpherePairScore over all pairs of a particle set, except pairs
       within one rigid body (container.PairsRestraint over a
       ClosePairContainer with the default RigidClosePairsFinder, or
       core.ExcludedVolumeRestraint), or between the two sets of a
       CloseBipartitePairContainer. The score is zero for pairs that don't
       overlap, so it doesn't depend on the close-pair list."""
    name = 'ev'
    DENSE_MAX = 2048            # rows; above this, full() works in blocks

    @classmethod
    def claims(cls, d, s):
        import IMP.container
        import IMP.core
        if isinstance(d, IMP.container.PairsRestraint):
            so = d.get_score_object().get_derived_object()
            tn = d.get_container().get_type_name()
            if isinstance(so, IMP.core.SoftSpherePairScore) and tn.startswith(
                    ('ClosePairContainer', 'CloseBipartitePairContainer')):
                return int(d.this)
        if isinstance(d, IMP.core.ExcludedVolumeRestraint):
            return int(d.this)
        return None

    def add(self, s, w, d):
        import IMP
        import IMP.container
        import IMP.core
        m = d.get_model()
        if isinstance(d, IMP.core.ExcludedVolumeRestraint):
            k = d.get_score_object().get_derived_object().get_k() \
                if hasattr(d, 'get_score_object') else None
            (lsc,) = [o for o in d.get_inputs()
                      if 'Container' in type(o).__name__
                      or hasattr(o, 'get_contents')] or [None]
            ps = [int(p) for p in IMP.SingletonContainer.get_from(
                lsc).get_contents()]
            bipartite = None
        else:
            cpc = d.get_container().get_derived_object()
            if cpc.get_number_of_pair_filters():
                raise Unsupported('EV with pair filters')
            k = d.get_score_object().get_derived_object().get_k()
            if cpc.get_type_name().startswith('CloseBipartitePairContainer'):
                pairs = np.asarray([[int(a), int(b)] for a, b in
                                    cpc.get_range_indexes()], np.int64)
                A, B = np.unique(pairs[:, 0]), np.unique(pairs[:, 1])
                ps = list(A) + list(B)
                bipartite = len(A)
            else:
                ps = [int(p) for p in cpc.get_all_possible_indexes()]
                bipartite = None
        label = np.empty(len(ps), np.int64)
        for i, p in enumerate(ps):
            pi = IMP.ParticleIndex(int(p))
            if bipartite is not None:
                label[i] = 0 if i < bipartite else 1
            elif IMP.core.RigidMember.get_is_setup(m, pi):
                label[i] = int(IMP.core.RigidMember(m, pi).get_rigid_body()
                               .get_particle_index())
            else:
                label[i] = -1 - i
        for p in ps:
            s.row(p)
        self.ps, self.lab = ps, label
        self.kk = k * w * d.get_weight()
        self.restraints.append((w, d))

    def finalize(self, s):
        self.rows = s.rows(self.ps)
        self.r = s.row_radius[self.rows]

    def full(self, st):
        X = st['X'][:, self.rows]                   # (3, n, C)
        n = X.shape[1]
        lab, r = self.lab, self.r
        if n <= self.DENSE_MAX:
            d = geom.norm(X[:, :, None] - X[:, None, :])
            x = jnp.minimum(d - (r[:, None] + r[None])[..., None], 0.)
            keep = np.triu(np.ones((n, n), bool), 1) & \
                (lab[:, None] != lab[None])
            return 0.5 * self.kk * jnp.sum(
                jnp.where(keep[..., None], x * x, 0.), axis=(0, 1))
        B = 512
        nb_ = -(-n // B)
        pad = nb_ * B - n
        Xp = jnp.pad(X, ((0, 0), (0, pad), (0, 0)))
        rp = np.concatenate([r, np.zeros(pad)])
        lp = np.concatenate([lab, -10 ** 12 - np.arange(pad)])
        idx = np.arange(nb_ * B)

        def block(b):
            xi = jax.lax.dynamic_slice_in_dim(Xp, b * B, B, axis=1)
            ri = jax.lax.dynamic_slice_in_dim(jnp.asarray(rp), b * B, B)
            li = jax.lax.dynamic_slice_in_dim(jnp.asarray(lp), b * B, B)
            ii = jax.lax.dynamic_slice_in_dim(jnp.asarray(idx), b * B, B)
            d = geom.norm(xi[:, :, None] - X[:, None, :])
            x = jnp.minimum(d - (ri[:, None] + r[None])[..., None], 0.)
            keep = ((ii[:, None] < np.arange(n)[None]) & (ii[:, None] < n)
                    & (li[:, None] != lab[None]))
            return jnp.sum(jnp.where(keep[..., None], x * x, 0.),
                           axis=(0, 1))
        return 0.5 * self.kk * jnp.sum(jax.lax.map(block, jnp.arange(nb_)),
                                       axis=0)


class CrossLinks(Term):
    """isd.LogWrapper of CrossLinkMSRestraints (ambiguous contributions,
       sigma and psi nuisances): w * sum_r -log(w_r * p_r), with
       p_r = 1 - prod_c (1 - term_c)"""
    name = 'xl'
    CPP_FLOAT = False           # check.terms sets it to match C++ exactly

    @classmethod
    def claims(cls, d, s):
        import IMP.isd
        if isinstance(d, IMP.isd.LogWrapper):
            return int(d.this)
        return None

    def add(self, s, w, d):
        import IMP.isd
        rs = [x.get_derived_object() for x in d.restraints]
        if not all(isinstance(x, IMP.isd.CrossLinkMSRestraint) for x in rs):
            raise Unsupported('LogWrapper of non-crosslinks')
        r0 = rs[0]
        length = r0.get_length()
        slope = r0.get_slope() if r0.get_has_slope() else None
        for r in rs:
            if (r.get_is_length_variable() or r.get_length() != length
                    or r.get_has_slope() != (slope is not None)
                    or (slope is not None and r.get_slope() != slope)
                    or r.get_log_prob()):
                raise Unsupported('cross-link variant')
        cp, cs, cpsi, rcon = [], [], [], []
        for r in rs:
            lst = []
            for k in range(r.get_number_of_contributions()):
                lst.append(len(cp))
                cp.append([int(x) for x in
                           r.get_contribution_particle_indexes(k)])
                cs.append([int(x) for x in
                           r.get_contribution_sigma_indexes(k)])
                cpsi.append(int(r.get_contribution_psi_index(k)))
            rcon.append(lst)
        for a, b in cp:
            s.row(a)
            s.row(b)
        for a in [x for c in cs for x in c] + cpsi:
            s.nuisance(a)
        self.length, self.slope = length, slope
        self.w = w * d.get_weight()
        self.rw = np.array([r.get_weight() for r in rs])
        self.cp, self.cs, self.cpsi, self.rcon = cp, cs, cpsi, rcon
        self.restraints.append((w, d))

    def finalize(self, s):
        cp = np.array(self.cp, np.int64)
        self.ca, self.cb = s.rows(cp[:, 0]), s.rows(cp[:, 1])
        self.same = cp[:, 0] == cp[:, 1]
        self.r0 = s.row_radius[self.ca]
        cs = np.array(self.cs, np.int64)
        self.s0, self.s1 = s.nuis_rows(cs[:, 0]), s.nuis_rows(cs[:, 1])
        self.psi = s.nuis_rows(self.cpsi)
        nc = len(self.cp)
        cmax = max(len(x) for x in self.rcon)
        tab = np.full((len(self.rcon), cmax), nc, np.int64)
        for i, lst in enumerate(self.rcon):
            tab[i, :len(lst)] = lst
        self.con = tab                      # (nres, cmax), nc = padding

    def full(self, st):
        X, nu = st['X'], st['nuis']
        d = jnp.where(self.same[:, None], 36.0 / 35.0 * self.r0[:, None],
                      geom.norm(X[:, self.ca] - X[:, self.cb]))
        term = geom.xl_term(d, nu[self.s0], nu[self.s1], nu[self.psi],
                            self.length, self.slope,
                            self.CPP_FLOAT)                 # (nc, C)
        onem = jnp.concatenate([1.0 - term, jnp.ones_like(term[:1])])
        prob = 1.0 - jnp.prod(onem[self.con], axis=1)      # (nres, C)
        return self.w * jnp.sum(-jnp.log(prob * self.rw[:, None]), axis=0)
