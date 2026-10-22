"""Proof of concept: a whole sweep (47 bead moves and the 2 rigid-body
moves) as one Pallas (Triton) kernel.

Each GPU program takes a block of BC chains and runs every move of the sweep
in order. The dynamic EV rows stay in registers for the dense EV sums;
positions of single rows (the moved bead, bond and cross-link partners)
are read from the state in memory, where each move writes its result
(with a barrier, so the whole program sees it). A rigid-body move computes
its proposed rows into a scratch buffer, scores dense EV, static EV, its
cross-links and the full EM term (both bodies' model Gaussians against the
50 data Gaussians, and model-model), and copies the scratch rows into the
state if accepted.

With em_cache (the default) a rigid-body move recomputes only the moved
body's model-data sums and the model-model pairs across the two bodies.
The other body's sums are kept per chain (st.em_parts), and the pairs
within one body do not change under rigid motion.

Same formulas as TutSamplerSoA (cov_mode 'rotate'); float32 sums run in a
different order, so energies agree to rounding.

    smp = TutSamplerFused(s, bc=16, num_warps=2)
    st = smp.initial_state(32768, seed=3)
    st = smp.make_run(10)(st)
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
import scoring as T                                     # noqa: E402
from sampler_soa import quat_mul_cf, quat_to_rot_cf, _state_cls  # noqa: E402
from pallas_sampler import TutSamplerPallas, NPAD, KPAD, f32  # noqa: E402

DPAD = 64           # data Gaussians, padded
XS_OT = 32          # scratch rows: proposed EV rows at 0.., other rows at 32..
_FSTATE = None


def _fused_state_cls():
    """SoAState plus em_parts (5, C), see TutSamplerFused._em_parts"""
    global _FSTATE
    if _FSTATE is None:
        import jax
        from dataclasses import dataclass

        @jax.tree_util.register_dataclass
        @dataclass
        class FusedState(_state_cls()):
            em_parts: jax.Array
        _FSTATE = FusedState
    return _FSTATE


class _Lists:
    """xp shim: stack() returns a list, so the component-first helpers
       work on lists of (BC,) vectors"""
    @staticmethod
    def stack(xs, axis=0):
        return list(xs)


class TutSamplerFused(TutSamplerPallas):
    def __init__(self, s, bc=16, num_warps=2, num_stages=1, layout='reg128',
                 em_cache=True, **kw):
        """layout of the rows for the dense EV sums:
           'reg128': the dynamic rows padded to 128, in registers through
                     the bead moves, updated in place (fastest);
           'reg96':  the 84 EV rows as 64 + 32 rows, in registers;
           'mem96':  the 84 EV rows as 64 + 32 rows, read from memory by
                     every move"""
        super().__init__(s, bc=bc, num_warps=num_warps,
                         num_stages=num_stages, mode='gmem', **kw)
        assert layout in ('reg128', 'reg96', 'mem96')
        self.layout = layout
        self.xreg = layout != 'mem96'
        assert self.n_de <= 96
        assert s.cov_mode == 'rotate' and s.nb == 2
        self._build_rb_tables()
        self._build_em_tables()
        self._fused = None
        # profiling switches (results are wrong when set): 'no_rb' skips
        # the rigid-body moves, 'no_em' their EM term
        self.prof = set()
        # em_cache: a rigid-body move recomputes only the moved body's
        # model-data sums and the model-model pairs across the bodies;
        # the other body's sums are kept per chain in st.em_parts, and the
        # pairs within a body do not change under rigid motion
        self.em_cache = em_cache
        cuts = np.cumsum([0] + [n for _, n in self.em_slices])
        self.em_range = [(int(cuts[b]), int(cuts[b + 1]))
                         for b in range(len(self.em_slices))]

    # ------------------------------------------------------------------
    def _build_rb_tables(self):
        """Per rigid-body move: Python constants and offsets into rbf/rbi"""
        s, r = self.s, self.r.astype(np.float32)
        F, I, self.rb = [], [], {}
        for i, p in enumerate(self.steps):
            if p['kind'] != 'rb':
                continue
            b = p['body']
            (o1, n1), (o2, n2) = self.body_ev[b], self.body_ot[b]
            assert n1 <= XS_OT and n2 <= 64 - XS_OT
            own = np.zeros(self.n_de, bool)
            own[o1:o1 + n1] = True
            assert (np.asarray(p['evmask_dyn']) == ~own[None]).all()
            d = dict(body=b, o1=o1, n1=n1, o2=o2, n2=n2,
                     max_trans=f32(p['max_trans']), max_rot=f32(p['max_rot']),
                     nxl=len(p['xi']))
            d['f_ev'] = len(F)
            for j in range(n1):
                F += [*p['local_ev'][:, j], r[o1 + j]]
            d['f_ot'] = len(F)
            for j in range(n2):
                F += list(p['local_ot'][:, j])
            d['f_xl'], d['i_xl'] = len(F), len(I)
            for j, (xi, xa, xb) in enumerate(zip(p['xi'], p['xa'], p['xb'])):
                assert xa < self.n_dyn
                if xb < self.n_dyn:
                    prow, spos = int(xb), (0., 0., 0.)
                else:
                    prow = NPAD - 1
                    spos = tuple(self.static_xyz[:, xb - self.n_dyn])
                F += [*p['local_xa'][:, j], *spos, s.xl_si[xi], s.xl_sj[xi],
                      s.xl_psi[xi]]
                I += [int(xa), prow]
            self.rb[i] = d
        self.rbf = np.asarray(F, np.float32)
        self.rbi = np.asarray(I or [0], np.int32)

    def _build_em_tables(self):
        """Model Gaussians (owner, row in X, index in the owner's other
           block, local covariance, mass, radius) and the data Gaussians,
           padded to DPAD"""
        s = self.s
        self.n_em = len(self.em_owner)
        I, F = [], []
        k_in = {}
        for m in range(self.n_em):
            b = int(self.em_owner[m])
            o, n = self.em_slices[b]
            k = k_in.get(b, 0)
            k_in[b] = k + 1
            I += [b, o + k, k]
            F += [*self.em_cov_local[m].reshape(-1), self.em_mass[m],
                  self.em_rad[m]]
        for b, (o, n) in enumerate(self.em_slices):
            assert k_in.get(b, 0) == n
        self.emi = np.asarray(I, np.int32)
        self.emf = np.asarray(F, np.float32)
        nd = len(s.d_mass)
        assert nd <= DPAD
        D = np.zeros((12, DPAD), np.float32)
        D[0:3, :nd] = s.d_xyz.T
        cov = s.d_cov
        for c, (a, b_) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2),
                                     (2, 2))):
            D[3 + c, :nd] = cov[:, a, b_]
        D[9, :nd] = s.d_mass
        D[10, :nd] = s.d_rad
        D[11, :nd] = 1.
        self.emd = D.reshape(-1)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _em_parts(self, X, quat):
        """(5, C): model-data overlap sums of body 0 and 1, distance sums
           of body 0 and 1, model-model overlaps within the bodies; as
           TutSamplerSoA._em"""
        import jax.numpy as jnp
        from sampler_soa import _dist
        s, dt = self.s, X.dtype
        xm = jnp.concatenate([X[:, o:o + n] for o, n in self.em_slices], 1)
        R = quat_to_rot_cf(jnp.moveaxis(quat, 1, 0), jnp)[:, :, self.em_owner]
        cl = jnp.asarray(np.moveaxis(self.em_cov_local, 0, -1), dt)[..., None]
        tmp = jnp.sum(R[:, :, None] * cl[None], axis=1)
        cm = jnp.sum(tmp[:, None] * R[None], axis=2)
        dx = jnp.asarray(s.d_xyz.T, dt)
        dc = jnp.asarray(np.moveaxis(s.d_cov, 0, -1), dt)
        mass_m = jnp.asarray(self.em_mass, dt)
        rad_m = jnp.asarray(self.em_rad, dt)
        md = self._overlap_cf(
            xm[:, :, None], cm[:, :, :, None], mass_m[:, None, None],
            dx[:, None, :, None], dc[:, :, None, :, None],
            jnp.asarray(s.d_mass, dt)[None, :, None])        # (10, 50, C)
        dist = _dist(xm[:, :, None], dx[:, None, :, None], jnp)
        sph = dist - rad_m[:, None, None] \
            - jnp.asarray(s.d_rad, dt)[None, :, None]
        mdm = jnp.sum(jnp.where(sph < s.em_md_cut, md, 0.), 1)   # (10, C)
        dsm = jnp.sum(dist, 1)
        mm = self._overlap_cf(
            xm[:, :, None], cm[:, :, :, None], mass_m[:, None, None],
            xm[:, None, :], cm[:, :, None, :], mass_m[None, :, None])
        sphm = _dist(xm[:, :, None], xm[:, None, :], jnp) \
            - rad_m[:, None, None] - rad_m[None, :, None]
        own = np.asarray(self.em_owner)
        same = np.triu(own[:, None] == own[None], 1)[..., None]
        intra = jnp.sum(jnp.where(jnp.asarray(same) & (sphm < s.em_mm_cut),
                                  mm, 0.), axis=(0, 1))
        return jnp.stack([mdm[a:b].sum(0) for a, b in self.em_range]
                         + [dsm[a:b].sum(0) for a, b in self.em_range]
                         + [intra])

    def _with_parts(self, st):
        """st as a state with em_parts, computed from its positions"""
        import dataclasses
        import jax
        if not hasattr(self, '_em_parts_j'):
            self._em_parts_j = jax.jit(self._em_parts)
        parts = self._em_parts_j(st.X, st.quat)
        if hasattr(st, 'em_parts'):
            st.em_parts = parts
            return st
        return _fused_state_cls()(
            **{f.name: getattr(st, f.name) for f in dataclasses.fields(st)},
            em_parts=parts)

    def resync(self, st, terms=False):
        out = super().resync(st, terms)
        if terms:
            return self._with_parts(out[0]), out[1]
        return self._with_parts(out)

    # ------------------------------------------------------------------
    def _sweep_kernel(self):
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt
        s, BC = self.s, self.bc
        nde, nse = self.n_de, self.n_se
        NI, NF, NBMAX = self.NI, self.NF, self.NBMAX
        XO = 2 + 5 * NBMAX
        lo = [f32(v) for v in self.g_lo]
        h = f32(self.g_h)
        gn = [int(v) for v in self.g_n]
        ncell = len(self.g_table) - 1
        ev_half = f32(0.5 * s.ev_k)
        n_em = 0 if 'no_em' in self.prof else self.n_em
        cache = self.em_cache and 'no_em' not in self.prof
        em_range = self.em_range
        xreg = self.xreg
        CHUNKS = ((0, NPAD),) if self.layout == 'reg128' else \
            ((0, 64), (64, 32))
        L = _Lists
        gnorm = T._GAUSS_NORM
        md_cut, mm_cut = f32(s.em_md_cut), f32(s.em_mm_cut)
        em_w, em_slope = f32(s.em_w), f32(s.em_slope)
        em_den = f32(s.em_self_mm), f32(s.em_dd)

        def soft(d, rr):
            x = jnp.minimum(d - rr, 0.)
            return x * x

        def dist(a, b):
            d0, d1, d2 = a[0] - b[0], a[1] - b[1], a[2] - b[2]
            return jnp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

        def rot(R, v, t):
            return [(R[a][0] * v[0] + R[a][1] * v[1] + R[a][2] * v[2]) + t[a]
                    for a in range(3)]

        def overlap(xa, ca, ma, xb, cb, mb):
            """TutSamplerSoA._overlap_cf on lists; c = 6 upper entries"""
            a, b, c = ca[0] + cb[0], ca[1] + cb[1], ca[2] + cb[2]
            d, e, f = ca[3] + cb[3], ca[4] + cb[4], ca[5] + cb[5]
            c00 = d * f - e * e
            c01 = c * e - b * f
            c02 = b * e - c * d
            det = a * c00 + b * c01 + c * c02
            c11 = a * f - c * c
            c12 = b * c - a * e
            c22 = a * d - b * b
            vx, vy, vz = xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]
            quad = (c00 * vx * vx + c11 * vy * vy + c22 * vz * vz
                    + 2. * (c01 * vx * vy + c02 * vx * vz + c12 * vy * vz)) \
                / det
            return ma * mb * gnorm / jnp.sqrt(det) * jnp.exp(-0.5 * quad)

        def kernel(*refs):
            (_, _, _, _, _, _, _, _, _, kt_ref, nrm_ref, uni_ref, rdyn_ref,
             tab_ref, gs_ref, gr_ref, it_ref, ft_ref, rbf_ref, rbi_ref,
             emf_ref, emi_ref, emd_ref,
             x_ref, q_ref, t_ref, e_ref, em_ref, esb_ref, esr_ref, acc_ref,
             ep_ref,
             xs_ref, ms_ref) = refs        # outputs alias the state inputs
            cs = pl.ds(pl.program_id(0) * BC, BC)
            kk = jax.lax.broadcasted_iota(jnp.int32, (KPAD, BC), 0)
            kT = kt_ref[cs]
            zero = jnp.zeros((BC,), jnp.float32)
            crow = [jax.lax.broadcasted_iota(jnp.int32, (n, BC), 0) + o
                    for o, n in CHUNKS]
            crad = [rdyn_ref[pl.ds(o, n)][:, None] for o, n in CHUNKS]

            def row(rw, sp=(0., 0., 0.)):
                return [x_ref[c, rw, cs] + sp[c] for c in range(3)]

            def static_ev(xn, rq):
                idx = [jnp.clip(jnp.floor((xn[c] - lo[c]) / h), -1.,
                                float(gn[c])).astype(jnp.int32)
                       for c in range(3)]
                inside = ((idx[0] >= 0) & (idx[0] < gn[0]) & (idx[1] >= 0)
                          & (idx[1] < gn[1]) & (idx[2] >= 0)
                          & (idx[2] < gn[2]))
                flat = (idx[0] * gn[1] + idx[1]) * gn[2] + idx[2]
                flat = jnp.where(inside, flat, ncell)
                cand = tab_ref[flat[None] * KPAD + kk]
                Sc = [gs_ref[c * (nse + 1) + cand] for c in range(3)]
                return jnp.sum(soft(dist(Sc, [v[None] for v in xn]),
                                    rq + gr_ref[cand]), axis=0)

            def dense_ev(X, xn, xo, rq, evm):
                e = zero
                for (o, n), Xc, rrow, rr0 in zip(CHUNKS, X, crow, crad):
                    rr = rq + rr0
                    pe = soft(dist(Xc, [v[None] for v in xn]), rr) \
                        - soft(dist(Xc, [v[None] for v in xo]), rr)
                    e = e + jnp.sum(jnp.where(evm(rrow), pe, 0.), axis=0)
                return e

            def tile():
                """the EV rows for the dense EV, as [(64, BC), (32, BC)]
                   chunks per coordinate ([chunk][coordinate])"""
                return tuple([x_ref[c, pl.ds(o, n), cs] for c in range(3)]
                             for o, n in CHUNKS)

            def xl_term(dn, do, si, sj, psi):
                args = (si, sj, psi, s.xl_length, s.xl_slope, jnp)
                return T.xl_neglogp(dn, *args) - T.xl_neglogp(do, *args)

            def metropolis(de, u1):
                return (de <= 0) | (u1 < jnp.exp(-de / kT))

            # ---------------------------------------------------- beads
            def bead_move(i, carry):
                X, E = carry
                rk, k = it_ref[i * NI], it_ref[i * NI + 1]
                nxl = it_ref[i * NI + 3]
                radius, rrk = ft_ref[i * NF], ft_ref[i * NF + 1]
                xo = row(rk)
                d = [nrm_ref[i, c, cs] for c in range(3)]
                u0, u1 = uni_ref[i, 0, cs], uni_ref[i, 1, cs]
                nd = jnp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                step = radius * u0 ** (1. / 3.)
                xn = [xo[c] + d[c] / nd * step for c in range(3)]
                if not xreg:
                    X = tile()
                e = dense_ev(X, xn, xo, rrk,
                             lambda rw: (rw < nde) & (rw != rk))
                es_n = static_ev(xn, rrk)
                es_o = esb_ref[k, cs]
                de = ev_half * (e + es_n - es_o)
                for j in range(NBMAX):      # unused slots have k = 0
                    f = i * NF + 2 + 5 * j
                    xc = row(it_ref[i * NI + 4 + j],
                             (ft_ref[f + 2], ft_ref[f + 3], ft_ref[f + 4]))
                    rc, kb = ft_ref[f], ft_ref[f + 1]
                    bo = jnp.maximum(dist(xc, xo) - rc, 0.)
                    bn = jnp.maximum(dist(xc, xn) - rc, 0.)
                    de = de + 0.5 * kb * (bn * bn - bo * bo)

                def xl(j, de):
                    f = i * NF + XO + 6 * j
                    xc = row(it_ref[i * NI + 4 + NBMAX + j],
                             (ft_ref[f + 3], ft_ref[f + 4], ft_ref[f + 5]))
                    return de + xl_term(dist(xc, xn), dist(xc, xo),
                                        ft_ref[f], ft_ref[f + 1],
                                        ft_ref[f + 2])
                de = jax.lax.fori_loop(0, nxl, xl, de)
                acc = metropolis(de, u1)
                if xreg:
                    X = tuple([jnp.where((rw == rk) & acc[None], xn[c][None],
                                         Xc[c]) for c in range(3)]
                              for Xc, rw in zip(X, crow))
                esb_ref[k, cs] = jnp.where(acc, es_n, es_o)
                acc_ref[i, cs] = acc_ref[i, cs] + acc.astype(jnp.int32)
                for c in range(3):
                    x_ref[c, rk, cs] = jnp.where(acc, xn[c], xo[c])
                plt.debug_barrier()
                return (X if xreg else None), E + jnp.where(acc, de, 0.)

            # ----------------------------------------------- rigid bodies
            def rb_move(i, X, E):
                p = self.rb[i]
                b, o1, n1, o2, n2 = p['body'], p['o1'], p['n1'], p['o2'], \
                    p['n2']
                fe, fo, fx, ix = p['f_ev'], p['f_ot'], p['f_xl'], p['i_xl']
                qo = [q_ref[b, c, cs] for c in range(4)]
                to = [t_ref[b, c, cs] for c in range(3)]
                d = [nrm_ref[i, c, cs] for c in range(3)]
                ax = [nrm_ref[i, c, cs] for c in range(3, 6)]
                u0, u1, u2 = (uni_ref[i, c, cs] for c in range(3))
                nd = jnp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                step = p['max_trans'] * u0 ** (1. / 3.)
                tn = [to[c] + d[c] / nd * step for c in range(3)]
                na = jnp.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2])
                ax = [v / na for v in ax]
                ang = (2. * u2 - 1.) * p['max_rot']
                sa = jnp.sin(ang / 2)
                qr = [jnp.cos(ang / 2), sa * ax[0], sa * ax[1], sa * ax[2]]
                qn = quat_mul_cf(qr, qo, L)
                nq = jnp.sqrt(qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2]
                              + qn[3] * qn[3])
                qn = [v / nq for v in qn]
                Rn = quat_to_rot_cf(qn, L)

                def evm(rw):
                    return (rw < nde) & ((rw < o1) | (rw >= o1 + n1))
                if not xreg:
                    X = tile()

                def ev_row(j, carry):
                    e, esn = carry
                    f = fe + 4 * j
                    xn = rot(Rn, [rbf_ref[f + c] for c in range(3)], tn)
                    rq = rbf_ref[f + 3]
                    for c in range(3):
                        xs_ref[c, j, cs] = xn[c]
                    return (e + dense_ev(X, xn, row(o1 + j), rq, evm),
                            esn + static_ev(xn, rq))
                e, es_n = jax.lax.fori_loop(0, n1, ev_row, (zero, zero))
                es_o = esr_ref[b, cs]
                de = ev_half * (e + es_n - es_o)

                def ot_row(j, z):
                    f = fo + 3 * j
                    xn = rot(Rn, [rbf_ref[f + c] for c in range(3)], tn)
                    for c in range(3):
                        xs_ref[c, XS_OT + j, cs] = xn[c]
                    return z
                jax.lax.fori_loop(0, n2, ot_row, 0)

                def xl(j, de):
                    f = fx + 9 * j
                    xa_n = rot(Rn, [rbf_ref[f + c] for c in range(3)], tn)
                    xa_o = row(rbi_ref[ix + 2 * j])
                    xc = row(rbi_ref[ix + 2 * j + 1],
                             [rbf_ref[f + 3 + c] for c in range(3)])
                    return de + xl_term(dist(xa_n, xc), dist(xa_o, xc),
                                        rbf_ref[f + 6], rbf_ref[f + 7],
                                        rbf_ref[f + 8])
                de = jax.lax.fori_loop(0, p['nxl'], xl, de)
                plt.debug_barrier()          # scratch rows visible

                # EM: model Gaussians at the proposal into ms_ref
                Ro = quat_to_rot_cf([q_ref[1 - b, c, cs] for c in range(4)],
                                    L)

                def model(m, z):
                    owner, rw, k = (emi_ref[3 * m + c] for c in range(3))
                    mv = owner == b
                    R = [[jnp.where(mv, Rn[a][c], Ro[a][c]) for c in range(3)]
                         for a in range(3)]
                    xn = [xs_ref[c, XS_OT + k, cs] for c in range(3)]
                    xc = row(rw)
                    pos = [jnp.where(mv, xn[c], xc[c]) for c in range(3)]
                    Cl = [[emf_ref[11 * m + 3 * a + c] for c in range(3)]
                          for a in range(3)]
                    tmp = [[R[a][0] * Cl[0][c] + R[a][1] * Cl[1][c]
                            + R[a][2] * Cl[2][c] for c in range(3)]
                           for a in range(3)]
                    cov = [tmp[a][0] * R[c][0] + tmp[a][1] * R[c][1]
                           + tmp[a][2] * R[c][2]
                           for a, c in ((0, 0), (0, 1), (0, 2), (1, 1),
                                        (1, 2), (2, 2))]
                    for c, v in enumerate(pos + cov):
                        ms_ref[c, m, cs] = v
                    return z
                jax.lax.fori_loop(0, n_em, model, 0)
                plt.debug_barrier()

                dat = [emd_ref[pl.ds(c * DPAD, DPAD)][:, None]
                       for c in range(12)]
                dvalid = dat[11] > 0.5

                def mdl(m):
                    v = [ms_ref[c, m, cs] for c in range(9)]
                    return v[:3], v[3:], emf_ref[11 * m + 9], \
                        emf_ref[11 * m + 10]

                def md_m(m, carry):
                    mds, dsum = carry
                    xm, cm, mass, rad = mdl(m)
                    xm2 = [v[None] for v in xm]
                    ov = overlap(xm2, [v[None] for v in cm], mass,
                                 dat[0:3], dat[3:9], dat[9])
                    dd = dist(xm2, dat[0:3])
                    sph = dd - rad - dat[10]
                    return (mds + jnp.sum(jnp.where(dvalid & (sph < md_cut),
                                                    ov, 0.), axis=0),
                            dsum + jnp.sum(jnp.where(dvalid, dd, 0.),
                                           axis=0))
                if cache:
                    (m0, m1), (k0, k1) = em_range[b], em_range[1 - b]
                    mds_b, dsum_b = jax.lax.fori_loop(m0, m1, md_m,
                                                      (zero, zero))
                    mds = mds_b + ep_ref[1 - b, cs]
                    dsum = dsum_b + ep_ref[3 - b, cs]
                else:
                    mds, dsum = jax.lax.fori_loop(0, n_em, md_m,
                                                  (zero, zero))

                def mm_m(m, mmx):
                    xm, cm, mass, rad = mdl(m)

                    def mm_k(k2, mmx):
                        xk, ck, mk, rk2 = mdl(k2)
                        sphm = dist(xm, xk) - rad - rk2
                        return mmx + jnp.where(
                            sphm < mm_cut, overlap(xm, cm, mass, xk, ck, mk),
                            0.)
                    return jax.lax.fori_loop(m + 1, n_em, mm_k, mmx)
                if cache:
                    def mm_x(m, mmx):
                        xm, cm, mass, rad = mdl(m)

                        def mm_k(k2, mmx):
                            xk, ck, mk, rk2 = mdl(k2)
                            sphm = dist(xm, xk) - rad - rk2
                            return mmx + jnp.where(
                                sphm < mm_cut,
                                overlap(xm, cm, mass, xk, ck, mk), 0.)
                        return jax.lax.fori_loop(k0, k1, mm_k, mmx)
                    mmx = jax.lax.fori_loop(m0, m1, mm_x, ep_ref[4, cs])
                else:
                    mmx = jax.lax.fori_loop(0, n_em, mm_m, zero)
                em_n = em_w * (-jnp.log(2. * mds / (em_den[0] + 2. * mmx
                                                    + em_den[1]))
                               + em_slope * dsum)
                em_o = em_ref[cs]
                if 'no_em' in self.prof:
                    em_n = em_o
                de = de + em_n - em_o
                acc = metropolis(de, u1)

                def copy(j, z, src, dst):
                    for c in range(3):
                        x_ref[c, dst + j, cs] = jnp.where(
                            acc, xs_ref[c, src + j, cs], x_ref[c, dst + j, cs])
                    return z
                jax.lax.fori_loop(0, n1, lambda j, z: copy(j, z, 0, o1), 0)
                jax.lax.fori_loop(0, n2, lambda j, z: copy(j, z, XS_OT, o2),
                                  0)
                for c in range(4):
                    q_ref[b, c, cs] = jnp.where(acc, qn[c], qo[c])
                for c in range(3):
                    t_ref[b, c, cs] = jnp.where(acc, tn[c], to[c])
                em_ref[cs] = jnp.where(acc, em_n, em_o)
                if cache:
                    ep_ref[b, cs] = jnp.where(acc, mds_b, ep_ref[b, cs])
                    ep_ref[2 + b, cs] = jnp.where(acc, dsum_b,
                                                  ep_ref[2 + b, cs])
                esr_ref[b, cs] = jnp.where(acc, es_n, es_o)
                acc_ref[i, cs] = acc_ref[i, cs] + acc.astype(jnp.int32)
                plt.debug_barrier()
                return (tile() if xreg else None), E + jnp.where(acc, de, 0.)

            X = tile() if xreg else None     # else loaded by each move
            E = e_ref[cs]
            for seg in self.segments:
                if isinstance(seg, list):
                    X, E = jax.lax.fori_loop(seg[0], seg[-1] + 1, bead_move,
                                             (X, E))
                elif 'no_rb' not in self.prof:
                    X, E = rb_move(seg, X, E)
            e_ref[cs] = E
        return kernel

    def _sweep(self, st):
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt
        key, k1, k2 = jax.random.split(st.rkey, 3)
        n, C = self.n_movers, st.energy.shape[0]
        assert C % self.bc == 0
        nrm = jax.random.normal(k1, (n, 6, C), st.X.dtype)
        uni = jax.random.uniform(k2, (n, 3, C), st.X.dtype)
        st.rkey = key
        if self._fused is None:
            self._fused = self._sweep_kernel()
        dt = st.X.dtype

        def sds(a):
            return jax.ShapeDtypeStruct(a.shape, a.dtype)
        state = (st.X, st.quat, st.trans, st.energy, st.em, st.es_bead,
                 st.es_body, st.accepted, st.em_parts)
        call = pl.pallas_call(
            self._fused,
            out_shape=tuple(sds(a) for a in state) + (
                jax.ShapeDtypeStruct((3, 64, C), dt),
                jax.ShapeDtypeStruct((9, max(self.n_em, 1), C), dt)),
            grid=(C // self.bc,),
            input_output_aliases={k: k for k in range(len(state))},
            compiler_params=plt.CompilerParams(
                num_warps=self.num_warps, num_stages=self.num_stages),
            name='sweep')
        out = call(*state, st.kT, nrm, uni, jnp.asarray(self.rdyn, dt),
                   jnp.asarray(self.tab), jnp.asarray(self.gS, dt),
                   jnp.asarray(self.gR, dt), jnp.asarray(self.itab),
                   jnp.asarray(self.ftab), jnp.asarray(self.rbf, dt),
                   jnp.asarray(self.rbi), jnp.asarray(self.emf, dt),
                   jnp.asarray(self.emi), jnp.asarray(self.emd, dt))
        (st.X, st.quat, st.trans, st.energy, st.em, st.es_bead, st.es_body,
         st.accepted, st.em_parts) = out[:9]
        return st
