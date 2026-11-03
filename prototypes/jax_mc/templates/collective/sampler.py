"""Collective template: every body moves at every MC step.

For systems where IMP's MonteCarlo proposes all its movers at once (the
mature NPC of PM_NPC_Assembly): one rigid-body mover per body, one
Metropolis decision per step. Ported from pm_sampler2 (PM_NPC_Assembly
bench) onto jax_mc's System and terms, and checked against the reference:

  python -m jax_mc.check decisions pm \\
      --sampler templates.collective.sampler:factory \\
      --chains 64 --ref-chains 4 --kT 30

Proposals are the reference's (CollectiveEngine.propose, same draws). The
energy is rescored from scratch every step, but over short lists. Each
technique has a precondition, asserted in __init__:

  R8   EV over neighbour lists. An outer list (pairs within r_i + r_j +
       outer_skin, built on the GPU from a cell grid) and an inner list
       (skin) refiltered from it. A chain rebuilds its lists, inside the
       step and from its current state, when its displacement bound since
       the last build would exceed half a skin. Exact while half the inner
       skin exceeds the largest one-step displacement of any EV bead.
  OP   overall position in closed form: with tolerance 0 and reference
       points rigid within each body (Kabsch residual < 1e-6),
       sum_i w |R l_i + t - x0_i|^2 is, per body,
       tr(dR M2 dR^T) + 2 dt . (dR m1) + W |dt|^2 with dR = R - R_a,
       dt = t - t_a; O(bodies), and written in differences so float32 does
       not cancel.
  EM   isd EM with one side static and frozen covariances on the other
       (the protocol's swapped GaussianEMRestraint): every pair has a
       constant combined covariance, so its inverse, normalization and
       cutoff are tables; a step scores the pairs within cutoff + skin/2,
       listed with the inner list.
  MEM  membrane depth over the beads that can reach the membrane before
       the next rebuild (margin skin/2).

State: quat (4, nb, C) and trans (3, nb, C) chain-last, as the engine;
neighbour lists chain-first (C, ...). A chain whose lists overflow gets a
NaN energy (its moves are then all rejected, and total() shows it).
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

from jax_mc import engine as E
from jax_mc.system import MEMBER, STATIC, Unsupported, quat_to_rot
from jax_mc.terms import basic, em, geom, pm

TERMS = (basic.SoftSphereEV, pm.OverallPosition, pm.MembraneBinding,
         pm.MembraneDepth, pm.BoundingBox, em.EMIsd)
G = 128                     # cell grid per axis (keys wrap-checked)


def _kabsch(loc, x):
    lc, xc = loc.mean(0), x.mean(0)
    U, _, Vt = np.linalg.svd((loc - lc).T @ (x - xc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, xc - R @ lc


def factory(s):
    return CollectiveNL(s)


class CollectiveNL:
    incremental = True
    steps_per_sweep = 1
    dtype = staticmethod(E.Engine.dtype)

    def __init__(self, s, skin=8.0, outer_skin=30.0):
        if s.mode != 'all':
            raise Unsupported('collective template: SerialMover system')
        self.s, self.skin, self.outer_skin = s, skin, outer_skin
        self.ref = E.CollectiveEngine(s)
        if len(self.ref.i_rb) != len(s.movers) or len(self.ref.i_rb) != s.nb:
            raise Unsupported('collective template: one rigid-body mover '
                              'per body, nothing else')
        terms = {}
        for t in s.terms:
            if type(t) not in TERMS or type(t) in terms:
                raise Unsupported(f'collective template: term {t.label}')
            terms[type(t)] = t
        self.T = terms
        rk = s.row_kind

        def members(rows):
            if not np.all(rk[rows] == MEMBER):
                raise Unsupported('collective template: particles not on '
                                  'moving bodies')
            return s.row_ref[rows], s.row_local[rows]

        d0 = s.initial_dofs(1)
        X0 = s.positions(d0)[..., 0].T                      # (rows, 3)
        marg = 0.5 * skin
        f = np.asarray
        C = {}
        ext_rows = []

        # ---- EV (R8) -------------------------------------------------------
        ev = terms[basic.SoftSphereEV]
        er = ev.rows
        evb, evl = members(er)
        lab = ev.lab
        if len(set(zip(lab, evb))) != len(set(evb)) or \
                len(set(lab)) != len(set(evb)):
            raise Unsupported('collective template: EV exclusions are not '
                              'exactly the same-body pairs')
        ext_rows.append(er)
        self.nev = nev = len(er)
        self.ev_k = ev.kk
        bt = np.zeros((nev, 8))
        bt[:, :3], bt[:, 3], bt[:, 4] = evl, ev.r, evb
        C['beadtab'] = bt
        C['evb'], C['evl'], C['evr'] = evb, evl, f(ev.r)
        Pev0 = X0[er]
        pairs_at = lambda sk: self._pairs_host(Pev0, ev.r, evb, sk)  # noqa
        outer0, inner0 = pairs_at(outer_skin), pairs_at(skin)
        self.cell = cell = 2 * float(ev.r.max()) + outer_skin
        ci = np.floor(Pev0 / cell).astype(np.int64)
        offs = [(0, 0, 0)] + [(a, b, c) for a in (-1, 0, 1)
                              for b in (-1, 0, 1) for c in (-1, 0, 1)
                              if (a, b, c) > (0, 0, 0)]
        cnt_cell = {tuple(c): n for c, n in zip(*np.unique(
            ci, axis=0, return_counts=True))}
        cnt_cb = {tuple(c): n for c, n in zip(*np.unique(
            np.concatenate([ci, evb[:, None]], 1), axis=0,
            return_counts=True))}
        tot = 0
        for c, b in zip(map(tuple, ci), evb):
            for o in offs:
                nc = (c[0] + o[0], c[1] + o[1], c[2] + o[2])
                tot += cnt_cell.get(nc, 0) - cnt_cb.get(nc + (b,), 0)
        self.offs = np.array(offs, np.int32)
        self.TCAP = int(1.4 * tot) + 65536
        self.CAP = max(4096, int(1.5 * len(inner0)))
        self.OCAP = max(16384, int(1.3 * len(outer0)))
        self.NBP = 128 if s.nb < 128 else 1 << int(math.ceil(
            math.log2(s.nb + 1)))

        # ---- overall position, closed form (OP) ---------------------------
        nb = s.nb
        Ra, ta = np.tile(np.eye(3), (nb, 1, 1)), np.zeros((nb, 3))
        W, m1, M2 = np.zeros(nb), np.zeros((nb, 3)), np.zeros((nb, 3, 3))
        self.has_op = pm.OverallPosition in terms
        if self.has_op:
            op = terms[pm.OverallPosition]
            if np.any(op.tol != 0):
                raise Unsupported('collective template: overall position '
                                  'tolerance != 0')
            opb, opl = members(op.rows)
            x0 = op.x0.T
            resid = 0.
            for b in range(nb):
                sel = opb == b
                if not sel.any():
                    continue
                R_, t_ = _kabsch(opl[sel], x0[sel])
                resid = max(resid, np.abs(opl[sel] @ R_.T + t_
                                          - x0[sel]).max())
                w = op.wsig[sel]
                Ra[b], ta[b], W[b] = R_, t_, w.sum()
                m1[b] = (w[:, None] * opl[sel]).sum(0)
                M2[b] = np.einsum('n,ni,nj->ij', w, opl[sel], opl[sel])
            if resid > 1e-6:
                raise Unsupported(f'collective template: overall position '
                                  f'reference points not rigid ({resid})')
        C.update(Ra=Ra, ta=ta, W=W, m1=m1, M2=M2)

        # ---- membrane (MEM), membrane binding, bounding box ---------------
        self.dep = terms.get(pm.MembraneDepth)
        self.MCAP = 1
        if self.dep is not None:
            C['depb'], C['depl'] = members(self.dep.rows)
            ext_rows.append(self.dep.rows)
            self.MCAP = max(1024, int(1.5 * self._mem_mask(
                X0[self.dep.rows].T, np).sum()))
        self.mbm = terms.get(pm.MembraneBinding)
        if self.mbm is not None:
            C['mbmb'], C['mbml'] = members(self.mbm.rows)
        self.bb = terms.get(pm.BoundingBox)
        if self.bb is not None:
            C['bbb'], C['bbl'] = members(self.bb.rows)

        # ---- EM pair tables (EM) ------------------------------------------
        self.emt = emt = terms.get(em.EMIsd)
        self.ECAP = 1
        if emt is not None:
            if not (np.all(rk[emt.rows] == STATIC) and s.cov_mode == 'frozen'
                    and not emt.slope):
                raise Unsupported('collective template: isd EM needs a '
                                  'static model side, frozen covariances, '
                                  'no slope')
            C['emb'], C['eml'] = members(emt.drows)
            ext_rows.append(emt.drows)
            cf, cm = emt.cglob.T, emt.dcglob.T              # (n, 6)
            full6 = lambda c: c[..., [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(  # noqa
                c.shape[:-1] + (3, 3))
            Sg = full6(cf[:, None] + cm[None])              # (nf, nm, 3, 3)
            Sinv = np.linalg.inv(Sg)
            logpref = (np.log(emt.mass)[:, None] + np.log(emt.dm)[None]
                       + math.log(geom.GAUSS_NORM)
                       - 0.5 * np.log(np.linalg.det(Sg)))
            thr = emt.dcut + emt.rad[:, None] + emt.drad[None]
            nf, nm = thr.shape
            et = np.zeros((nf * nm, 16))
            et[:, 0] = Sinv[..., 0, 0].ravel()
            et[:, 1] = Sinv[..., 1, 1].ravel()
            et[:, 2] = Sinv[..., 2, 2].ravel()
            et[:, 3] = 2 * Sinv[..., 0, 1].ravel()
            et[:, 4] = 2 * Sinv[..., 0, 2].ravel()
            et[:, 5] = 2 * Sinv[..., 1, 2].ravel()
            et[:, 6] = logpref.ravel()
            et[:, 7] = (thr * thr).ravel()
            xf = X0[emt.rows]
            et[:, 8:11] = np.repeat(xf, nm, axis=0)
            et[:, 11] = np.tile(np.arange(nm), nf)
            C['emtab'] = et
            C['thrm2'] = ((thr + marg) ** 2).ravel()
            C['emi'] = np.repeat(np.arange(nf), nm)
            C['emj'] = np.tile(np.arange(nm), nf)
            C['xf'] = xf
            # MM among the static side is constant: from the term itself
            st0 = dict(X=s.positions(d0), quat=d0['quat'], nuis=d0['nuis'])
            mmx = float(np.asarray(emt._sums(st0)['mmx'])[0])
            self.em_denom = emt.self_mm + 2. * mmx + emt.dd
            xm0 = X0[emt.drows]
            dd0 = ((xm0[C['emj']] - xf[C['emi']]) ** 2).sum(-1)
            self.ECAP = int(1.25 * (dd0 < C['thrm2']).sum()) + 64

        # ---- displacement bound ---------------------------------------------
        allr = np.unique(np.concatenate(ext_rows))
        ext = np.zeros(nb)
        np.maximum.at(ext, s.row_ref[allr],
                      np.sqrt((s.row_local[allr] ** 2).sum(1)))
        C['ext'] = ext
        mvs = s.movers
        self.step_max = max(mv.max_trans + 2 * math.sin(mv.max_rot / 2)
                            * ext[mv.bodies[0]] for mv in mvs)
        if marg <= self.step_max:
            raise Unsupported(f'collective template: skin/2 = {marg} must '
                              f'exceed the largest one-step displacement '
                              f'{self.step_max:.2f}')
        self.info = dict(nev=nev, outer0=len(outer0), inner0=len(inner0),
                         cap=self.CAP, ocap=self.OCAP, tcap=self.TCAP,
                         mcap=self.MCAP, ecap=self.ECAP,
                         step_max=self.step_max)
        self.C = C
        self._jit_init = jax.jit(self._init_one)

    # ---------------- host helpers ----------------------------------------
    @staticmethod
    def _pairs_host(P, r, body, skin):
        from scipy.spatial import cKDTree
        pr = cKDTree(P).query_pairs(2 * r.max() + skin, output_type='ndarray')
        d = np.sqrt(((P[pr[:, 0]] - P[pr[:, 1]]) ** 2).sum(1))
        ok = (d < r[pr[:, 0]] + r[pr[:, 1]] + skin) & \
            (body[pr[:, 0]] != body[pr[:, 1]])
        return pr[ok]

    def _mem_mask(self, X, xp):
        """beads (X (3, n)) that can reach the membrane within skin/2"""
        dep, marg = self.dep, 0.5 * self.skin
        x, y, z = X
        rxy = xp.sqrt(x * x + y * y)
        r = xp.maximum(rxy, 1e-9)
        rs = dep.R / r
        dist = xp.sqrt((x - rs * x) ** 2 + (y - rs * y) ** 2 + z * z)
        return ((rxy > dep.R - marg) & (xp.abs(z) < 0.5 * dep.h + marg)) | (
            (rxy < dep.R + marg) & (dist < dep.rmin + marg))

    # ---------------- per-chain device code (chain-first) ------------------
    def _c(self, k):
        return jnp.asarray(self.C[k])

    @staticmethod
    def _frames(q, t):
        """per-chain q (4, nb), t (3, nb) -> R (nb, 3, 3), t (nb, 3)"""
        return jnp.moveaxis(quat_to_rot(q, jnp), -1, 0), t.T

    @staticmethod
    def _pos(R, t, b, loc):
        return jnp.einsum('...ij,...j->...i', R[b], loc) + t[b]

    def _build_outer(self, Pev):
        """inter-body EV pairs with d < r_i + r_j + outer_skin: beads sorted
           by (cell, body); each bead's candidates are its 14 half-shell
           cells minus its own body's block, enumerated as one flat list"""
        c_ = self._c
        nev, TCAP, OCAP, NBP = self.nev, self.TCAP, self.OCAP, self.NBP
        offs = jnp.asarray(self.offs)
        evb, evr = c_('evb'), c_('evr')

        def ckey(c):
            c = jnp.clip(c + G // 2, 0, G - 1)
            return c[..., 0] + G * (c[..., 1] + G * c[..., 2])
        cidx = jnp.floor(Pev / self.cell).astype(jnp.int32)
        ckb = ckey(cidx) * NBP + evb
        order = jnp.argsort(ckb)
        ks = ckb[order]
        nof = offs.shape[0]
        qk = ckey(cidx[:, None, :] + offs[None]) * NBP          # (N, 14)
        bi = evb[:, None]
        sc = jnp.searchsorted(ks, qk.ravel(), side='left')
        ec = jnp.searchsorted(ks, (qk + NBP).ravel(), side='left')
        so = jnp.searchsorted(ks, (qk + bi).ravel(), side='left')
        eo = jnp.searchsorted(ks, (qk + bi + 1).ravel(), side='left')
        s2 = jnp.stack([sc, eo], -1).ravel()
        L = (jnp.stack([so - sc, ec - eo], -1).ravel()).astype(jnp.int32)
        incl = jnp.cumsum(L)
        total = incl[-1]
        start = incl - L
        p = jnp.arange(TCAP, dtype=jnp.int32)
        sid = jnp.arange(L.shape[0], dtype=jnp.int32)
        marker = jnp.full(TCAP, -1, jnp.int32).at[
            jnp.where(L > 0, start, TCAP)].max(sid, mode='drop')
        seg = jnp.maximum(jax.lax.cummax(marker), 0)
        k = p - start[seg]
        i = seg // (2 * nof)
        o = (seg // 2) % nof
        j = order[jnp.minimum(s2[seg] + k, nev - 1)]
        d2 = jnp.sum((Pev[j] - Pev[i]) ** 2, axis=-1)
        lim = evr[j] + evr[i] + self.outer_skin
        m = (p < total) & ((o > 0) | (j > i)) & (d2 < lim * lim)
        cnt = jnp.sum(m, dtype=jnp.int32)
        sel = jnp.nonzero(m, size=OCAP, fill_value=0)[0]
        pairs = jnp.stack([i[sel], j[sel]], -1).astype(jnp.int32)
        ovf = (total > TCAP) | jnp.any(jnp.abs(cidx) >= G // 2 - 1)
        return pairs, cnt, ovf | (cnt > OCAP)

    def _build_inner(self, Pev, opairs, ocnt):
        evr = self._c('evr')
        i, j = opairs[:, 0], opairs[:, 1]
        d2 = jnp.sum((Pev[i] - Pev[j]) ** 2, -1)
        lim = evr[i] + evr[j] + self.skin
        m = (d2 < lim * lim) & (jnp.arange(self.OCAP) < ocnt)
        cnt = jnp.sum(m, dtype=jnp.int32)
        idx = jnp.nonzero(m, size=self.CAP, fill_value=0)[0].astype(jnp.int32)
        return opairs[idx], cnt, cnt > self.CAP

    def _build_mem(self, R, t):
        if self.dep is None:
            return jnp.zeros(1, jnp.int32), jnp.int32(0), jnp.bool_(False)
        X = self._pos(R, t, self._c('depb'), self._c('depl'))
        m = self._mem_mask(X.T, jnp)
        cnt = jnp.sum(m, dtype=jnp.int32)
        idx = jnp.nonzero(m, size=self.MCAP, fill_value=0)[0].astype(
            jnp.int32)
        return idx, cnt, cnt > self.MCAP

    def _build_em(self, R, t):
        if self.emt is None:
            return jnp.zeros(1, jnp.int32), jnp.int32(0), jnp.bool_(False)
        c_ = self._c
        xm = self._pos(R, t, c_('emb'), c_('eml'))
        v = xm[c_('emj')] - c_('xf')[c_('emi')]
        m = jnp.sum(v * v, -1) < c_('thrm2')
        cnt = jnp.sum(m, dtype=jnp.int32)
        idx = jnp.nonzero(m, size=self.ECAP, fill_value=0)[0].astype(
            jnp.int32)
        return idx, cnt, cnt > self.ECAP

    def _lists(self, q, t, out=None):
        """all lists for one chain at frames q (4, nb), t (3, nb); the outer
           list is reused from `out` = (pairs, count, q0, t0) if given"""
        R, tt = self._frames(q, t)
        Pev = self._pos(R, tt, self._c('evb'), self._c('evl'))
        if out is None:
            op, oc, oo = self._build_outer(Pev)
            oq, ot = q, t
        else:
            op, oc, oq, ot = out
            oo = jnp.bool_(False)
        ip, ic, io = self._build_inner(Pev, op, oc)
        mi, mc, mo = self._build_mem(R, tt)
        ei, ecn, eo = self._build_em(R, tt)
        return dict(out_pairs=op, out_count=oc, out_quat=oq, out_trans=ot,
                    in_pairs=ip, in_count=ic, in_quat=q, in_trans=t,
                    mem_idx=mi, mem_count=mc, em_idx=ei, em_count=ecn,
                    overflow=oo | io | mo | eo)

    def _energy(self, q, t, L):
        """total energy (scalar) of one chain from its lists L"""
        c_ = self._c
        R, tt = self._frames(q, t)
        e = 0.
        # EV over the inner list, packed bead rows
        Tb = jnp.concatenate([R.reshape(-1, 9), tt], -1)       # (nb, 12)
        pr = L['in_pairs']
        ga, gb = c_('beadtab')[pr[:, 0]], c_('beadtab')[pr[:, 1]]
        Ta, Tb_ = Tb[ga[:, 4].astype(jnp.int32)], Tb[gb[:, 4].astype(
            jnp.int32)]

        def xyz(g, T_):
            return jnp.stack([jnp.sum(T_[:, 0:3] * g[:, :3], -1),
                              jnp.sum(T_[:, 3:6] * g[:, :3], -1),
                              jnp.sum(T_[:, 6:9] * g[:, :3], -1)], -1) \
                + T_[:, 9:12]
        d = jnp.sqrt(jnp.sum((xyz(ga, Ta) - xyz(gb, Tb_)) ** 2, -1))
        x = jnp.minimum(d - ga[:, 3] - gb[:, 3], 0.)
        e = e + 0.5 * self.ev_k * jnp.sum(jnp.where(
            jnp.arange(self.CAP) < L['in_count'], x * x, 0.))
        if self.has_op:
            dR = R - c_('Ra')
            dt = tt - c_('ta')
            e = e + jnp.sum(jnp.einsum('bij,bjk,bik->b', dR, c_('M2'), dR)
                            + 2. * jnp.einsum('bi,bij,bj->b', dt, dR,
                                              c_('m1'))
                            + c_('W') * jnp.sum(dt * dt, -1))
        if self.dep is not None:
            mi = L['mem_idx']
            Xm = self._pos(R, tt, c_('depb')[mi], c_('depl')[mi])
            ed = self.dep.energies(None, Xm.T[..., None])[:, 0]
            e = e + jnp.sum(jnp.where(jnp.arange(self.MCAP)
                                      < L['mem_count'], ed, 0.))
        if self.mbm is not None:
            Xb = self._pos(R, tt, c_('mbmb'), c_('mbml'))
            e = e + jnp.sum(self.mbm.energies(None, Xb.T[..., None]))
        if self.bb is not None:
            Pb = self._pos(R, tt, c_('bbb'), c_('bbl'))
            e = e + jnp.sum(self.bb.energies(None, Pb.T[..., None]))
        if self.emt is not None:
            xm = self._pos(R, tt, c_('emb'), c_('eml'))
            e8 = c_('emtab')[L['em_idx']]                       # (E, 16)
            v = xm[e8[:, 11].astype(jnp.int32)] - e8[:, 8:11]
            qf = (e8[:, 0] * v[:, 0] ** 2 + e8[:, 1] * v[:, 1] ** 2
                  + e8[:, 2] * v[:, 2] ** 2 + e8[:, 3] * v[:, 0] * v[:, 1]
                  + e8[:, 4] * v[:, 0] * v[:, 2]
                  + e8[:, 5] * v[:, 1] * v[:, 2])
            ok = (jnp.arange(self.ECAP) < L['em_count']) & \
                (jnp.sum(v * v, -1) < e8[:, 7])
            ov = jnp.where(ok, jnp.exp(e8[:, 6] - 0.5 * qf), 0.)
            e = e + self.emt.w * -jnp.log(2. * jnp.sum(ov) / self.em_denom)
        return jnp.where(L['overflow'], jnp.nan, e)

    def _disp(self, q, t, q0, t0):
        """bound on any bead's displacement between frames (q0, t0) and
           (q, t) of one chain"""
        dot = jnp.abs(jnp.sum(q * q0, axis=0))
        rot = 2. * jnp.sqrt(jnp.maximum(1. - dot * dot, 0.))
        return jnp.max(jnp.sqrt(jnp.sum((t - t0) ** 2, 0))
                       + rot * self._c('ext'))

    def _init_one(self, q, t):
        L = self._lists(q, t)
        return L, self._energy(q, t, L)

    # ---------------- protocol ---------------------------------------------
    LISTS = ('out_pairs', 'out_count', 'out_quat', 'out_trans', 'in_pairs',
             'in_count', 'in_quat', 'in_trans', 'mem_idx', 'mem_count',
             'em_idx', 'em_count', 'overflow')

    def init(self, C, seed=0, kT=None, dofs=None):
        s, dt = self.s, self.dtype()
        d = s.initial_dofs(C) if dofs is None else dofs
        d = {k: jnp.asarray(v, dt) for k, v in d.items()}
        q, t = d['quat'], d['trans']
        same = bool(jnp.all(q == q[..., :1])) and \
            bool(jnp.all(t == t[..., :1]))
        if same:            # one build, broadcast (C builds don't fit)
            L, e = self._jit_init(q[..., 0], t[..., 0])
            L = {k: jnp.broadcast_to(v, (C,) + v.shape) for k, v in L.items()}
            e = jnp.broadcast_to(e, (C,))
        else:
            per = [self._jit_init(q[..., c], t[..., c]) for c in range(C)]
            L = {k: jnp.stack([p[0][k] for p in per]) for k in self.LISTS}
            e = jnp.stack([p[1] for p in per])
        kT = s.kT if kT is None else kT
        return dict(d, **L, energy=e,
                    kT=jnp.broadcast_to(jnp.asarray(kT, dt), (C,)),
                    key=E.chain_keys(seed, C), sweep=jnp.zeros((), jnp.int32),
                    n_acc=jnp.zeros((1, C), jnp.int32),
                    n_inner=jnp.zeros(C, jnp.int32),
                    n_outer=jnp.zeros(C, jnp.int32))

    def _rebuild(self, st, i):
        """rebuild chain i's lists at its current state (outer only if its
           bound since the outer build exceeds (outer_skin - skin) / 2)"""
        q, t = st['quat'][..., i], st['trans'][..., i]
        need_out = self._disp(q, t, st['out_quat'][i], st['out_trans'][i]) \
            > 0.5 * (self.outer_skin - self.skin)
        L = jax.lax.cond(
            need_out, lambda _: self._lists(q, t),
            lambda _: self._lists(q, t, (st['out_pairs'][i],
                                         st['out_count'][i],
                                         st['out_quat'][i],
                                         st['out_trans'][i])), None)
        new = {k: st[k].at[i].set(L[k]) for k in self.LISTS}
        new['overflow'] = st['overflow'].at[i].set(st['overflow'][i]
                                                   | L['overflow'])
        new['n_inner'] = st['n_inner'].at[i].add(1)
        new['n_outer'] = st['n_outer'].at[i].add(need_out.astype(jnp.int32))
        return dict(st, **new)

    def sweep(self, st):
        """one collective step: propose (as the reference), rebuild the lists
           of chains that need it, rescore, Metropolis"""
        n = len(self.s.movers)
        U, Z = E.draws(st['key'], st['sweep'], n, st['quat'].dtype)
        d = {k: st[k] for k in ('quat', 'trans', 'bead', 'nuis')}
        dn = self.ref.propose(d, U, Z)
        qp, tp = dn['quat'], dn['trans']
        need = jax.vmap(self._disp, in_axes=(-1, -1, 0, 0))(
            qp, tp, st['in_quat'], st['in_trans']) > 0.5 * self.skin

        def body(c):
            st, need = c
            i = jnp.argmax(need)
            return self._rebuild(st, i), need.at[i].set(False)
        st, _ = jax.lax.while_loop(lambda c: jnp.any(c[1]), body, (st, need))
        L = {k: st[k] for k in self.LISTS}
        e_new = jax.vmap(self._energy, in_axes=(-1, -1, 0))(qp, tp, L)
        dE = e_new - st['energy']
        acc = U[0, 2] < jnp.exp(-dE / st['kT'])
        return dict(st, quat=jnp.where(acc, qp, st['quat']),
                    trans=jnp.where(acc, tp, st['trans']),
                    energy=jnp.where(acc, e_new, st['energy']),
                    sweep=st['sweep'] + 1,
                    n_acc=st['n_acc'] + acc[None]), acc[None]

    run = E.Engine.run

    def total(self, st):
        return st['energy'] + self.s.constant

    def stats(self, st):
        """list rebuilds per chain (mean) and chains whose lists overflowed"""
        return dict(inner_rebuilds=float(np.mean(st['n_inner'])),
                    outer_rebuilds=float(np.mean(st['n_outer'])),
                    overflowed=int(np.sum(st['overflow'])))

    def dofs(self, st):
        return {k: np.asarray(st[k]) for k in ('quat', 'trans', 'bead',
                                               'nuis')}
