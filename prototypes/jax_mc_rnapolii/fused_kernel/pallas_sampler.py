"""Proof of concept: the bead moves of TutSamplerSoA as one fused Pallas
(Triton) kernel per run of consecutive bead moves.

Each GPU program takes a block of BC chains, loads their dynamic rows (105,
padded to 128) into registers once, runs every bead move of the run in
order (proposal, dense EV against the dynamic EV rows, static EV through
the grid, bonds, cross-links, Metropolis, update) and writes the rows back
at the end. Per-bead constants (radii, bond and cross-link parameters,
static partner positions) come from small tables (see the modes below).
The two rigid-body moves stay in XLA (TutSamplerSoA._rb_step).
pallas_sweep.TutSamplerFused builds on this class.

Same formulas as TutSamplerSoA._bead_step; float32 sums run in a different
order, so energies agree to rounding, not bit for bit.

    smp = TutSamplerPallas(s, bc=32, num_warps=4)
    st = smp.initial_state(32768, seed=3)    # X padded to 128 rows
    st = smp.make_run(10)(st)
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
import scoring as T                      # noqa: E402
from sampler_soa import TutSamplerSoA    # noqa: E402

NPAD = 128          # dynamic rows, padded to a power of 2 (Triton shapes)
KPAD = 32           # static grid candidates per cell, padded
FAR = 0.            # position of padding rows (static partners point here)


def f32(x):
    return float(np.float32(x))


class TutSamplerPallas(TutSamplerSoA):
    def __init__(self, s, bc=32, num_warps=4, num_stages=1, mode='loop',
                 **kw):
        """mode: 'loop' (one move body in a loop over moves, per-move
           constants from small tables: compiles fast), 'gmem' (as loop,
           but the moved bead's and its partners' positions are read from
           memory, where every move writes its result, instead of being
           picked out of the registers) or 'unroll' (every move compiled
           separately with literal constants)"""
        super().__init__(s, **kw)
        # row NPAD - 1 is the padding row (position FAR, radius 0)
        assert self.n_dyn < NPAD and self.g_K <= KPAD
        assert not self.body_grid
        self.bc, self.num_warps, self.num_stages = bc, num_warps, num_stages
        self.mode = mode
        r = np.zeros(NPAD, np.float32)
        r[:self.n_dyn] = self.r[:self.n_dyn]
        self.rdyn = r
        nse = self.n_se
        tab = np.full((len(self.g_table), KPAD), nse, np.int32)
        tab[:, :self.g_K] = self.g_table
        self.tab = tab.reshape(-1)
        self.gS = np.asarray(self.g_S, np.float32).reshape(-1)  # (3*(nse+1))
        self.gR = np.asarray(self.g_R, np.float32)
        # the kernel's EV mask: every dynamic EV row but the bead itself
        for p in self.steps:
            if p['kind'] == 'bead':
                m = np.arange(self.n_de) != p['row']
                assert (np.asarray(p['evmask_dyn']) == m).all()
        # runs of consecutive bead moves
        self.segments, cur = [], []
        for i, p in enumerate(self.steps):
            if p['kind'] == 'bead':
                cur.append(i)
            else:
                if cur:
                    self.segments.append(cur)
                self.segments.append(i)
                cur = []
        if cur:
            self.segments.append(cur)
        self._kernels = {}
        self._build_move_tables()

    def _build_move_tables(self):
        """Per-move constants for mode 'loop' (rows indexed by move):
           ints  [row, k, nbond, nxl, bond rows (NBMAX), xl rows (NXMAX)];
           floats [radius, r_row, per bond (rc, k, x, y, z),
                   per cross-link (si, sj, psi, x, y, z)].
           A static partner points at a padding row (position 0) and has
           its position in the floats; a dynamic one has position 0 there."""
        s = self.s
        bead = [p for p in self.steps if p['kind'] == 'bead']
        self.NBMAX = max(len(p['ci']) for p in bead)
        self.NXMAX = max(len(p['xi']) for p in bead)
        self.NI = 4 + self.NBMAX + self.NXMAX
        self.NF = 2 + 5 * self.NBMAX + 6 * self.NXMAX
        # unused partner slots point at a padding row (position 0); their
        # constants are 0, so they add nothing
        it = np.full((self.n_movers, self.NI), NPAD - 1, np.int32)
        ft = np.zeros((self.n_movers, self.NF), np.float32)
        r = self.r.astype(np.float32)

        def pos(row):
            if row < self.n_dyn:
                return row, (0., 0., 0.)
            # static: a padding row (position 0) plus the literal position
            return NPAD - 1, tuple(self.static_xyz[:, row - self.n_dyn])
        for i, p in enumerate(self.steps):
            if p['kind'] != 'bead':
                continue
            rk = int(p['row'])
            it[i, :4] = rk, rk - self.bead_off, len(p['ci']), len(p['xi'])
            ft[i, :2] = p['radius'], r[rk]
            for j, (ci, co) in enumerate(zip(p['ci'], p['co'])):
                row, xyz = pos(int(co))
                it[i, 4 + j] = row
                ft[i, 2 + 5 * j:7 + 5 * j] = (
                    r[rk] + r[co] + np.float32(s.conn_mean[ci]),
                    s.conn_k[ci], *xyz)
            o = 2 + 5 * self.NBMAX
            for j, (xi, xr) in enumerate(zip(p['xi'], p['xo'])):
                row, xyz = pos(int(xr))
                it[i, 4 + self.NBMAX + j] = row
                ft[i, o + 6 * j:o + 6 * j + 6] = (
                    s.xl_si[xi], s.xl_sj[xi], s.xl_psi[xi], *xyz)
        self.itab, self.ftab = it.reshape(-1), ft.reshape(-1)

    # ------------------------------------------------------------------
    def initial_state(self, n_chains, **kw):
        st = super().initial_state(n_chains, **kw)
        return self.pad(st)

    def pad(self, st):
        import jax.numpy as jnp
        C = st.energy.shape[0]
        if st.X.shape[1] < NPAD:
            st.X = jnp.concatenate(
                [st.X, jnp.full((3, NPAD - st.X.shape[1], C), FAR,
                                st.X.dtype)], 1)
        return st

    # ------------------------------------------------------------------
    def _partner(self, row, X, rows):
        """Position (3 values, (BC,) or Python floats) of row `row`: from
           the registers if dynamic, a literal if static"""
        import jax.numpy as jnp
        if row < self.n_dyn:
            sel = rows == row
            return [jnp.sum(jnp.where(sel, X[c], 0.), axis=0)
                    for c in range(3)]
        return [f32(self.static_xyz[c, row - self.n_dyn]) for c in range(3)]

    def _kernel(self, moves):
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        s, BC = self.s, self.bc
        nde, nse = self.n_de, self.n_se
        r = self.r
        lo = [f32(v) for v in self.g_lo]
        h = f32(self.g_h)
        gn = [int(v) for v in self.g_n]
        ncell = len(self.g_table) - 1
        ev_half = f32(0.5 * s.ev_k)

        def soft(d, rr):
            x = jnp.minimum(d - rr, 0.)
            return x * x

        def dist(a, b):
            d0, d1, d2 = a[0] - b[0], a[1] - b[1], a[2] - b[2]
            return jnp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

        def kernel(x_ref, esb_ref, e_ref, acc_ref, kt_ref, nrm_ref, uni_ref,
                   rdyn_ref, tab_ref, gs_ref, gr_ref,
                   xo_ref, esbo_ref, eo_ref, acco_ref):
            cs = pl.ds(pl.program_id(0) * BC, BC)
            X = [x_ref[c, :, cs] for c in range(3)]            # (NPAD, BC)
            rows = jax.lax.broadcasted_iota(jnp.int32, (NPAD, BC), 0)
            kk = jax.lax.broadcasted_iota(jnp.int32, (KPAD, BC), 0)
            rdyn = rdyn_ref[:][:, None]                         # (NPAD, 1)
            E = e_ref[cs]
            kT = kt_ref[cs]
            for i in moves:
                p = self.steps[i]
                rk = int(p['row'])
                k = rk - self.bead_off
                sel = rows == rk
                xo = [jnp.sum(jnp.where(sel, X[c], 0.), axis=0)
                      for c in range(3)]
                d = [nrm_ref[i, c, cs] for c in range(3)]
                u0, u1 = uni_ref[i, 0, cs], uni_ref[i, 1, cs]
                nd = jnp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                step = f32(p['radius']) * u0 ** (1. / 3.)
                xn = [xo[c] + d[c] / nd * step for c in range(3)]
                # dense EV against the dynamic EV rows
                rr = f32(r[rk]) + rdyn
                evm = (rows < nde) & (rows != rk)
                pe = soft(dist([X[c] for c in range(3)],
                               [xn[c][None] for c in range(3)]), rr) \
                    - soft(dist([X[c] for c in range(3)],
                                [xo[c][None] for c in range(3)]), rr)
                e = jnp.sum(jnp.where(evm, pe, 0.), axis=0)
                # static EV through the grid
                idx = [jnp.clip(jnp.floor((xn[c] - lo[c]) / h), -1.,
                                float(gn[c])).astype(jnp.int32)
                       for c in range(3)]
                inside = ((idx[0] >= 0) & (idx[0] < gn[0]) & (idx[1] >= 0)
                          & (idx[1] < gn[1]) & (idx[2] >= 0)
                          & (idx[2] < gn[2]))
                flat = (idx[0] * gn[1] + idx[1]) * gn[2] + idx[2]
                flat = jnp.where(inside, flat, ncell)
                cand = tab_ref[flat[None] * KPAD + kk]           # (KPAD, BC)
                Sc = [gs_ref[c * (nse + 1) + cand] for c in range(3)]
                Rc = gr_ref[cand]
                es_n = jnp.sum(soft(dist(Sc, [xn[c][None] for c in range(3)]),
                                    f32(r[rk]) + Rc), axis=0)
                es_o = esb_ref[k, cs]
                de = ev_half * (e + es_n - es_o)
                # bonds
                for ci, co in zip(p['ci'], p['co']):
                    xc = self._partner(int(co), X, rows)
                    rc = f32(np.float32(r[rk]) + np.float32(r[co])
                             + np.float32(s.conn_mean[ci]))
                    kb = f32(s.conn_k[ci])
                    bo = jnp.maximum(dist(xc, xo) - rc, 0.)
                    bn = jnp.maximum(dist(xc, xn) - rc, 0.)
                    de = de + 0.5 * kb * (bn * bn - bo * bo)
                # cross-links
                for xi, xr in zip(p['xi'], p['xo']):
                    xc = self._partner(int(xr), X, rows)
                    args = (f32(s.xl_si[xi]), f32(s.xl_sj[xi]),
                            f32(s.xl_psi[xi]), s.xl_length, s.xl_slope, jnp)
                    de = de + (T.xl_neglogp(dist(xc, xn), *args)
                               - T.xl_neglogp(dist(xc, xo), *args))
                acc = (de <= 0) | (u1 < jnp.exp(-de / kT))
                upd = sel & acc[None]
                X = [jnp.where(upd, xn[c][None], X[c]) for c in range(3)]
                esbo_ref[k, cs] = jnp.where(acc, es_n, es_o)
                E = E + jnp.where(acc, de, 0.)
                acco_ref[i, cs] = acc_ref[i, cs] + acc.astype(jnp.int32)
            for c in range(3):
                xo_ref[c, :, cs] = X[c]
            eo_ref[cs] = E
        return kernel

    def _kernel_loop(self, moves, gmem=False):
        """The same move as _kernel, as one loop body over moves[0] ..
           moves[-1] (consecutive), with per-move constants from itab/ftab"""
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        s, BC = self.s, self.bc
        nde, nse = self.n_de, self.n_se
        NI, NF, NBMAX = self.NI, self.NF, self.NBMAX
        XO = 2 + 5 * NBMAX
        lo = [f32(v) for v in self.g_lo]
        h = f32(self.g_h)
        gn = [int(v) for v in self.g_n]
        ncell = len(self.g_table) - 1
        ev_half = f32(0.5 * s.ev_k)
        m0, m1 = moves[0], moves[-1] + 1
        assert list(moves) == list(range(m0, m1))

        def soft(d, rr):
            x = jnp.minimum(d - rr, 0.)
            return x * x

        def dist(a, b):
            d0, d1, d2 = a[0] - b[0], a[1] - b[1], a[2] - b[2]
            return jnp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

        def kernel(x_ref, esb_ref, e_ref, acc_ref, kt_ref, nrm_ref, uni_ref,
                   rdyn_ref, tab_ref, gs_ref, gr_ref, it_ref, ft_ref,
                   xo_ref, esbo_ref, eo_ref, acco_ref):
            from jax.experimental.pallas import triton as plt
            cs = pl.ds(pl.program_id(0) * BC, BC)
            rows = jax.lax.broadcasted_iota(jnp.int32, (NPAD, BC), 0)
            kk = jax.lax.broadcasted_iota(jnp.int32, (KPAD, BC), 0)
            rdyn = rdyn_ref[:][:, None]
            kT = kt_ref[cs]

            def partner(X, row, sx, sy, sz):
                if gmem:
                    # xo_ref aliases x_ref and holds every accepted move
                    return [xo_ref[c, row, cs] + v
                            for c, v in enumerate((sx, sy, sz))]
                sel = rows == row
                return [jnp.sum(jnp.where(sel, X[c], 0.), axis=0) + v
                        for c, v in enumerate((sx, sy, sz))]

            def move(i, carry):
                X, E = carry
                rk, k = it_ref[i * NI], it_ref[i * NI + 1]
                nxl = it_ref[i * NI + 3]
                radius, rrk = ft_ref[i * NF], ft_ref[i * NF + 1]
                sel = rows == rk
                xo = partner(X, rk, 0., 0., 0.)
                d = [nrm_ref[i, c, cs] for c in range(3)]
                u0, u1 = uni_ref[i, 0, cs], uni_ref[i, 1, cs]
                nd = jnp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                step = radius * u0 ** (1. / 3.)
                xn = [xo[c] + d[c] / nd * step for c in range(3)]
                rr = rrk + rdyn
                evm = (rows < nde) & (rows != rk)
                pe = soft(dist(X, [v[None] for v in xn]), rr) \
                    - soft(dist(X, [v[None] for v in xo]), rr)
                e = jnp.sum(jnp.where(evm, pe, 0.), axis=0)
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
                es_n = jnp.sum(soft(dist(Sc, [v[None] for v in xn]),
                                    rrk + gr_ref[cand]), axis=0)
                es_o = esb_ref[k, cs]
                de = ev_half * (e + es_n - es_o)
                for j in range(NBMAX):      # unused slots have k = 0
                    f = i * NF + 2 + 5 * j
                    xc = partner(X, it_ref[i * NI + 4 + j], ft_ref[f + 2],
                                 ft_ref[f + 3], ft_ref[f + 4])
                    rc, kb = ft_ref[f], ft_ref[f + 1]
                    bo = jnp.maximum(dist(xc, xo) - rc, 0.)
                    bn = jnp.maximum(dist(xc, xn) - rc, 0.)
                    de = de + 0.5 * kb * (bn * bn - bo * bo)

                def xl(j, de):
                    f = i * NF + XO + 6 * j
                    xc = partner(X, it_ref[i * NI + 4 + NBMAX + j],
                                 ft_ref[f + 3], ft_ref[f + 4], ft_ref[f + 5])
                    args = (ft_ref[f], ft_ref[f + 1], ft_ref[f + 2],
                            s.xl_length, s.xl_slope, jnp)
                    return de + (T.xl_neglogp(dist(xc, xn), *args)
                                 - T.xl_neglogp(dist(xc, xo), *args))
                de = jax.lax.fori_loop(0, nxl, xl, de)
                acc = (de <= 0) | (u1 < jnp.exp(-de / kT))
                upd = sel & acc[None]
                X = tuple(jnp.where(upd, xn[c][None], X[c]) for c in range(3))
                esbo_ref[k, cs] = jnp.where(acc, es_n, es_o)
                acco_ref[i, cs] = acc_ref[i, cs] + acc.astype(jnp.int32)
                if gmem:
                    for c in range(3):
                        xo_ref[c, rk, cs] = jnp.where(acc, xn[c], xo[c])
                    plt.debug_barrier()     # visible to the whole program
                return X, E + jnp.where(acc, de, 0.)

            X0 = tuple(x_ref[c, :, cs] for c in range(3))
            X, E = jax.lax.fori_loop(m0, m1, move, (X0, e_ref[cs]))
            if not gmem:
                for c in range(3):
                    xo_ref[c, :, cs] = X[c]
            eo_ref[cs] = E
        return kernel

    def _segment(self, moves, st, nrm, uni):
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt
        C = st.energy.shape[0]
        assert C % self.bc == 0
        key = tuple(moves)
        if key not in self._kernels:
            self._kernels[key] = (
                self._kernel(moves) if self.mode == 'unroll' else
                self._kernel_loop(moves, gmem=self.mode == 'gmem'))
        extra = ((jnp.asarray(self.itab), jnp.asarray(self.ftab))
                 if self.mode != 'unroll' else ())

        def sds(a):
            return jax.ShapeDtypeStruct(a.shape, a.dtype)
        call = pl.pallas_call(
            self._kernels[key],
            out_shape=(sds(st.X), sds(st.es_bead), sds(st.energy),
                       sds(st.accepted)),
            grid=(C // self.bc,),
            input_output_aliases={0: 0, 1: 1, 2: 2, 3: 3},
            compiler_params=plt.CompilerParams(
                num_warps=self.num_warps, num_stages=self.num_stages),
            name=f'beads_{moves[0]}_{moves[-1]}')
        dt = st.X.dtype
        st.X, st.es_bead, st.energy, st.accepted = call(
            st.X, st.es_bead, st.energy, st.accepted, st.kT, nrm, uni,
            jnp.asarray(self.rdyn, dt), jnp.asarray(self.tab),
            jnp.asarray(self.gS, dt), jnp.asarray(self.gR, dt), *extra)
        return st

    def _sweep(self, st):
        import jax
        key, k1, k2 = jax.random.split(st.rkey, 3)
        n, C = self.n_movers, st.energy.shape[0]
        nrm = jax.random.normal(k1, (n, 6, C), st.X.dtype)
        uni = jax.random.uniform(k2, (n, 3, C), st.X.dtype)
        st.rkey = key
        for seg in self.segments:
            if isinstance(seg, list):
                st = self._segment(seg, st, nrm, uni)
            else:
                st = self._rb_step(seg, st, nrm, uni)
        return st
