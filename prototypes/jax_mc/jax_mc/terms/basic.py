"""Pair restraints, soft-sphere excluded volume, cross-links."""
import numpy as np
import jax
import jax.numpy as jnp

from . import Term, geom, pad, slots
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
        self.rows_read = np.unique(np.concatenate([self.ri, self.rj]))
        self.nuis_read = np.zeros(0, np.int64)

    def energies(self, q, xi, xj):
        """(len(q), C) energy of pairs q from positions (3, len(q), C)"""
        g = lambda a: jnp.asarray(a)[q]                         # noqa
        t = geom.norm(xi - xj) - g(self.off)[:, None]
        mode = g(self.mode)[:, None]
        tc = jnp.where(mode == HUB, jnp.maximum(t, 0.),
                       jnp.where(mode == HLB, jnp.minimum(t, 0.), t))
        e = jnp.where(mode == LINEAR, t, 0.5 * tc * tc)
        return g(self.w * self.k)[:, None] * e

    def full(self, st):
        X = st['X']
        q = np.arange(len(self.ri))
        return jnp.sum(self.energies(q, X[:, self.ri], X[:, self.rj]),
                       axis=0)

    def touched(self, rows):
        return int((np.isin(self.ri, rows) ^ np.isin(self.rj, rows)).sum())

    def move_tables(self, s, moves):
        """pairs with exactly one end moved (the mover is rigid on its
           rows, so pairs inside it don't change), and the slots of their
           moved ends (-1: not moved)"""
        items, si, sj = [], [], []
        for rows, _ in moves:
            slot = slots(rows)
            a = np.isin(self.ri, rows)
            b = np.isin(self.rj, rows)
            q = np.nonzero(a ^ b)[0]
            items.append(q)
            si.append([slot.get(int(self.ri[x]), -1) for x in q])
            sj.append([slot.get(int(self.rj[x]), -1) for x in q])
        return dict(item=pad(items, len(self.ri)), si=pad(si, -1),
                    sj=pad(sj, -1))

    def delta(self, tab, st, cache, mv):
        n = len(self.ri)
        q = tab['item']
        qq = jnp.minimum(q, n - 1)
        ri, rj = jnp.asarray(self.ri)[qq], jnp.asarray(self.rj)[qq]
        X = st['X']
        e = self.energies(qq, mv.pos(ri, tab['si']), mv.pos(rj, tab['sj'])) \
            - self.energies(qq, X[:, ri], X[:, rj])
        return jnp.sum(jnp.where((q < n)[:, None], e, 0.), axis=0), cache


class SoftSphereEV(Term):
    """SoftSpherePairScore over all pairs of a particle set, except pairs
       within one rigid body (container.PairsRestraint over a
       ClosePairContainer with the default RigidClosePairsFinder, or
       core.ExcludedVolumeRestraint), or between the two sets of a
       CloseBipartitePairContainer. The score is zero for pairs that don't
       overlap, so it doesn't depend on the close-pair list."""
    name = 'ev'
    SPLIT = True
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
            so = d.get_score_object().get_derived_object()
            if not isinstance(so, IMP.core.SoftSpherePairScore):
                raise Unsupported('ExcludedVolumeRestraint score '
                                  + type(so).__name__)
            k = so.get_k()
            ps = [int(p) for p in d.get_container().get_contents()]
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
        rows = s.rows(self.ps)
        order = np.argsort(rows, kind='stable')     # static, bodies, beads
        self.rows, self.lab = rows[order], self.lab[order]
        self.r = s.row_radius[self.rows]
        kind = s.row_kind[self.rows]
        self.n_static = int(np.sum(kind == 0))
        self.xs0 = s.row_xyz0[self.rows[:self.n_static]].T  # (3, n_static)
        self.row_kind, self.row_ref = kind, s.row_ref[self.rows]
        self.rows_read = self.rows
        self.nuis_read = np.zeros(0, np.int64)

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
        C = X.shape[-1]
        B = int(max(8, min(512, 2 ** 22 // (n * C))))
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
        acc = jax.lax.fori_loop(0, nb_, lambda b, a: a + block(b),
                                jnp.zeros(C, X.dtype))
        return 0.5 * self.kk * acc

    def touched(self, rows):
        return int(np.isin(self.rows, rows).sum())

    def move_tables(self, s, moves):
        """the mover's rows that are EV particles (EV index, slot), and which
           pairs (moved particle, other particle) count: different labels,
           the other one not moved (keep_s: static others, keep_m: the
           rest)"""
        n, ns = len(self.rows), self.n_static
        pos = {int(r): i for i, r in enumerate(self.rows)}
        idx, sl = [], []
        for rows, _ in moves:
            q = [(pos[int(r)], k) for k, r in enumerate(rows) if int(r) in pos]
            idx.append([a for a, _ in q])
            sl.append([k for _, k in q])
        idx = pad(idx, n)
        keep = np.zeros(idx.shape + (n,), bool)
        for i, a in enumerate(idx):
            ok = a < n
            moved = np.zeros(n, bool)
            moved[a[ok]] = True
            la = self.lab[np.minimum(a, n - 1)]
            keep[i] = ok[:, None] & ~moved[None] & (la[:, None] !=
                                                    self.lab[None])
        keep_s = np.concatenate([keep[..., :ns], np.zeros(
            keep.shape[:2] + (self._static_pad(),), bool)], -1)
        return dict(idx=idx, slot=pad(sl, 0), keep_s=keep_s,
                    keep_m=keep[..., ns:])

    def _static_pad(self):
        """1 on the CPU: the static particles then get one more entry that
           never counts. Without it, a mover whose particles pair with
           every static one has a mask that is true everywhere; XLA drops
           the select, and its CPU backend (jaxlib 0.11) runs the reduction
           that is left about 10 times slower (Pol II's body moves). On a
           GPU the dropped select saves a kernel, so nothing is added."""
        import jax
        return int(self.n_static > 0 and jax.default_backend() == 'cpu')

    def delta(self, tab, st, cache, mv):
        """moved particles against all unmoved ones (pairs among moved
           particles don't change: the mover is rigid on its rows). Static
           particles are read from one copy shared by all chains."""
        n, ns = len(self.rows), self.n_static
        ri = jnp.asarray(self.r)[jnp.minimum(tab['idx'], n - 1)]
        k = self._static_pad()
        xs = np.concatenate([self.xs0, np.zeros((3, k))], 1)
        rs = np.concatenate([self.r[:ns], np.zeros(k)])
        parts = [(jnp.asarray(xs)[..., None], rs, tab['keep_s']),
                 (st['X'][:, self.rows[ns:]], self.r[ns:], tab['keep_m'])]

        def e(xi):                                          # (3, A, C)
            out = 0.
            for xa, ra, keep in parts:
                if xa.shape[1] == 0:
                    continue
                d = geom.norm(xi[:, :, None] - xa[:, None])
                x = jnp.minimum(d - (ri[:, None] + ra[None])[..., None], 0.)
                out = out + jnp.sum(jnp.where(keep[..., None], x * x, 0.),
                                    axis=(0, 1))
            return out
        sl = tab['slot']
        return 0.5 * self.kk * (e(mv.Xn[:, sl]) - e(mv.Xo[:, sl])), cache


class CrossLinks(Term):
    """isd.LogWrapper of CrossLinkMSRestraints (ambiguous contributions,
       sigma and psi nuisances): w * sum_r -log(w_r * p_r), with
       p_r = 1 - prod_c (1 - term_c)"""
    name = 'xl'
    SPLIT = True
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
        # PMI orders the restraints, their ambiguous contributions and the
        # two ends of each by set iteration, which depends on Python's hash
        # seed. In a canonical order (lower particle first, contributions
        # sorted, restraints sorted by them) the sums and products are the
        # same in every process.
        canon = []
        for r in rs:
            con = []
            for k in range(r.get_number_of_contributions()):
                pa, pb = (int(x) for x in
                          r.get_contribution_particle_indexes(k))
                sa, sb = (int(x) for x in
                          r.get_contribution_sigma_indexes(k))
                if pa > pb:
                    pa, pb, sa, sb = pb, pa, sb, sa
                con.append(((pa, pb), (sa, sb),
                            int(r.get_contribution_psi_index(k))))
            canon.append((sorted(con), r.get_weight()))
        canon.sort()
        cp, cs, cpsi, rcon = [], [], [], []
        for con, _ in canon:
            rcon.append(list(range(len(cp), len(cp) + len(con))))
            for p_, s_, psi in con:
                cp.append(list(p_))
                cs.append(list(s_))
                cpsi.append(psi)
        for a, b in cp:
            s.row(a)
            s.row(b)
        for a in [x for c in cs for x in c] + cpsi:
            s.nuisance(a)
        self.length, self.slope = length, slope
        self.w = w * d.get_weight()
        self.rw = np.array([w_ for _, w_ in canon])
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
        self.rows_read = np.unique(np.concatenate([self.ca, self.cb]))
        self.nuis_read = np.unique(np.concatenate([self.s0, self.s1,
                                                   self.psi]))

    def dists(self, res, xa, xb):
        """(len(res), cmax, C) distances of restraints res' contributions
           from the positions (3, len(res), cmax, C) of their ends"""
        c = jnp.minimum(jnp.asarray(self.con)[res], len(self.cp) - 1)
        g = lambda a: jnp.asarray(a)[c]                     # noqa
        return jnp.where(g(self.same)[..., None], 36.0 / 35.0
                         * g(self.r0)[..., None], geom.norm(xa - xb))

    def energies(self, res, d, nu):
        """(len(res), C) -log(w_r p_r) of restraints res from their
           contributions' distances d"""
        nc = len(self.cp)
        cidx = jnp.asarray(self.con)[res]                   # (R, cmax)
        c = jnp.minimum(cidx, nc - 1)
        g = lambda a: jnp.asarray(a)[c]                     # noqa
        term = geom.xl_term(d, nu[g(self.s0)], nu[g(self.s1)],
                            nu[g(self.psi)], self.length, self.slope,
                            self.CPP_FLOAT)                 # (R, cmax, C)
        onem = jnp.where((cidx >= nc)[..., None], 1.0, 1.0 - term)
        prob = 1.0 - jnp.prod(onem, axis=1)                 # (R, C)
        return -jnp.log(prob * jnp.asarray(self.rw)[res][:, None])

    def _ends(self, res):
        c = np.minimum(self.con[res], len(self.cp) - 1)
        return self.ca[c], self.cb[c]

    def _all_dists(self, X):
        res = np.arange(len(self.rcon))
        a, b = self._ends(res)
        return res, self.dists(res, X[:, a], X[:, b])

    def full(self, st):
        res, d = self._all_dists(st['X'])
        return self.w * jnp.sum(self.energies(res, d, st['nuis']), axis=0)

    def delta_nuis(self, st, cache, nuis):
        res, d = self._all_dists(st['X'])
        return self.w * jnp.sum(self.energies(res, d, nuis)
                                - self.energies(res, d, st['nuis']),
                                axis=0), cache

    def _crossing(self, rows):
        """per contribution (plus one False for the padding index): exactly
           one of its ends is in rows"""
        return np.concatenate([np.isin(self.ca, rows)
                               ^ np.isin(self.cb, rows), [False]])

    def touched(self, rows):
        return int(self._crossing(rows)[self.con].any(axis=1).sum())

    def move_tables(self, s, moves):
        """restraints with a contribution that has exactly one end moved,
           and for those contributions the slots of the ends (-1: use the
           old position; contributions inside the mover don't change)"""
        nres, cmax = self.con.shape
        res, sa, sb = [], [], []
        for rows, _ in moves:
            slot = slots(rows)
            cross = self._crossing(rows)                    # nc = padding
            q = np.nonzero(cross[self.con].any(axis=1))[0]
            res.append(q)
            sa.append([[slot.get(int(self.ca[c]), -1) if cross[c] else -1
                        for c in self.con[i]] for i in q])
            sb.append([[slot.get(int(self.cb[c]), -1) if cross[c] else -1
                        for c in self.con[i]] for i in q])
        A = max([len(q) for q in res] + [1])
        tsa = np.full((len(moves), A, cmax), -1, np.int64)
        tsb = np.full((len(moves), A, cmax), -1, np.int64)
        for i in range(len(moves)):
            if len(res[i]):
                tsa[i, :len(res[i])] = sa[i]
                tsb[i, :len(res[i])] = sb[i]
        return dict(res=pad(res, nres, A), sa=tsa, sb=tsb)

    def delta(self, tab, st, cache, mv):
        nres = len(self.rcon)
        q = tab['res']
        rr = jnp.minimum(q, nres - 1)
        c = jnp.minimum(jnp.asarray(self.con)[rr], len(self.cp) - 1)
        a, b = jnp.asarray(self.ca)[c], jnp.asarray(self.cb)[c]
        X, nu = st['X'], st['nuis']
        e = self.energies(rr, self.dists(rr, mv.pos(a, tab['sa']),
                                         mv.pos(b, tab['sb'])), nu) \
            - self.energies(rr, self.dists(rr, X[:, a], X[:, b]), nu)
        return self.w * jnp.sum(jnp.where((q < nres)[:, None], e, 0.),
                                axis=0), cache
