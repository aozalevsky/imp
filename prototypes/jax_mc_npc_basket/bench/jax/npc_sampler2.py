"""Delta-based fused JAX Monte Carlo for PMI systems like NPC_Basket REX2.

Same moves, target and interface as npc_sampler.NPCSampler, but a step never
builds a full candidate state:

* each mover has static tables: the compact rows it moves (with how each
  row's new position is computed), the bodies it rotates, and, for every
  scoring term, the items that touch those rows (connectivity / distance
  pairs, cross-link restraints, Z-axial and membrane particles, excluded-
  volume particles, EM model Gaussians and EM row blocks);
* every term returns its energy change from old vs new positions of only
  those items; the Metropolis test uses the summed change directly (the
  running total score is only reported, and resynced from scratch);
* EM keeps the model x data overlap matrix plus per-block (16 rows) column
  sums, so a move rereads only the affected blocks, and the score is
  rebuilt from block sums (sums of non-negative numbers only: no
  cancellation for densities with tiny overlap);
* accepted moves are written back with small scatters (moved rows, moved
  bodies, moved EM rows), not whole-array selects.

Nuisance moves (NormalMover) rescore everything that depends on nuisances
(cross-links, priors, EM sigma) in full; they are ~1% of steps.
"""
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

import imp_jax_npc as J

# imp_jax_fused.py / imp_jax_patches.py (from earlier IMP-JAX work) sit
# next to this file; IMP_JAX_PRIOR can point to another copy
_PRIOR = os.environ.get('IMP_JAX_PRIOR',
                        os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(_PRIOR) and _PRIOR not in sys.path:
    sys.path.append(_PRIOR)

KIND_BALL, KIND_RB, KIND_TR, KIND_NORMAL = 0, 1, 2, 3
# how a moved row gets its new position
C_MEMBER, C_CENTER, C_FREE, C_BALL, C_PAD = 0, 1, 2, 3, 4
_FOUR3PI = 4.0 / 3.0 * math.pi


class Unsupported(Exception):
    pass


def _derived(r):
    try:
        return r.get_derived_object()
    except NameError:
        return J._EM_REGISTRY[int(r.this)]


def _pad2(rows, fill, width=None):
    width = max([len(r) for r in rows] + [1]) if width is None else width
    out = np.full((len(rows), width), fill, np.int32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _padf(rows, width, shape=(), fill=0.):
    out = np.full((len(rows), width) + shape, fill, np.float64)
    for i, r in enumerate(rows):
        if len(r):
            out[i, :len(r)] = r
    return out


_STATE_CLS = None


def _state_class():
    global _STATE_CLS
    if _STATE_CLS is None:
        import jax

        @jax.tree_util.register_dataclass
        @dataclass
        class NPCState2:
            xyz: jax.Array          # (K+2, 3)
            nuis: jax.Array         # (K+2,)
            quat: jax.Array         # (nb+1, 4)
            ov: jax.Array           # (MB+1, D) EM overlaps (block-padded)
            bs: jax.Array           # (nblk+1, D) EM block column sums
            min2: jax.Array         # (MB+1,) squared nearest-center dist
            e_em: jax.Array         # current EM energy (weighted)
            score: jax.Array        # running total (reporting only)
            imov: jax.Array
            temperature: jax.Array
            rkey: jax.Array
            accepted: jax.Array
            proposed_mover: jax.Array
            accepted_mover: jax.Array
        _STATE_CLS = NPCState2
    return _STATE_CLS


def xl_terms(d, r0, r1, same, s0, s1, psi, length, slope, xp, log=False):
    """Per-contribution CrossLinkMSRestraint term (same math as
       IMP.isd.CrossLinkMSRestraint / imp_jax_patches xl_vec); log=True
       gives its logarithm, which does not underflow for long links."""
    dist = xp.where(same, 36.0 / 35.0 * r0, d)
    dist = xp.maximum(dist, 0.0001)
    half = length / 2.
    xlvol = _FOUR3PI * half ** 3

    def sphere_cap(ra, rb, dd):
        short = xp.minimum(_FOUR3PI * ra ** 3, _FOUR3PI * rb ** 3)
        dsafe = xp.where(dd > 0, dd, 1.0)
        mid = ((math.pi / 12 / dsafe * (ra + rb - dd) * (ra + rb - dd))
               * (dd * dd + 2 * dd * ra - 3 * ra * ra + 2 * dd * rb
                  + 6 * ra * rb - 3 * rb * rb))
        return xp.where(dd <= xp.abs(ra - rb), short,
                        xp.where(dd >= ra + rb, 0., mid))
    voli, volj = _FOUR3PI * s0 ** 3, _FOUR3PI * s1 ** 3
    close = dist < s0 + s1
    fi = xp.where(close, xp.minimum(voli, xlvol),
                  sphere_cap(s0, half, xp.abs(dist - s1 - half)))
    fj = xp.where(close, xp.minimum(volj, xlvol),
                  sphere_cap(s1, half, xp.abs(dist - s0 - half)))
    pofr = fi * fj / voli / volj
    term = psi * (1.0 - pofr) + pofr * (1 - psi)
    if log:
        return xp.log(term) - (slope * dist if slope is not None else 0.)
    if slope is not None:
        term = term * xp.exp(-slope * dist)
    return term


class NPCSampler2:
    def __init__(self, model, rs, movers, temperature=1.0, dtype='float32',
                 cov_mode='rotate', em_extra=2.0, block=16, ablate=(),
                 use_switch=True):
        import jax
        import jax.numpy as jnp
        import IMP
        import IMP.core
        import IMP.isd
        import IMP.jax
        import IMP.npc
        import IMP.pmi
        from IMP.core import _jax_rigid
        import imp_jax_fused
        J.install()

        self.m = m = model
        self.rs = rs
        self.dt = dt = jnp.dtype(dtype)
        self.acc = jnp.float64 if jax.config.jax_enable_x64 else dt
        self.cov_mode = cov_mode
        self.temperature0 = float(temperature)
        ablate = set(ablate)
        CompactArray = imp_jax_fused._compact_array_class()
        pi_ = IMP.ParticleIndex
        scale_key = IMP.isd.Scale.get_scale_key()

        def is_nuisance(p):
            return IMP.isd.Scale.get_is_setup(m, pi_(int(p)))

        # ================= restraints -> items =========================
        hub = []        # (i, j, sphere?, mean, k, weight, mode)
        zax = []        # (p, lower, upper, w/sigma)
        msl = []        # (p, R, r, half_thickness, w/sigma)
        xlw = []        # (weight, [CrossLinkMSRestraint])
        nleaves = []    # (weight, derived) nuisance-only restraints
        ems, evs = [], []

        def walk(r, w):
            d = _derived(r)
            name = type(d).__name__
            if isinstance(d, IMP.isd.LogWrapper):
                rs_ = [x.get_derived_object() for x in d.restraints]
                if not all(isinstance(x, IMP.isd.CrossLinkMSRestraint)
                           for x in rs_):
                    raise Unsupported('LogWrapper of non-crosslinks')
                xlw.append((w * d.get_weight(), rs_))
            elif isinstance(d, IMP.RestraintSet):
                ww = w * d.get_weight()
                for i in range(d.get_number_of_restraints()):
                    walk(d.get_restraint(i), ww)
            elif name == 'GaussianEMRestraint':
                ems.append((w * d.get_weight(), d))
            elif (isinstance(d, IMP.container.PairsRestraint)
                  and d.get_container().get_type_name().startswith(
                      ('ClosePairContainer', 'CloseBipartitePairContainer'))
                  and isinstance(d.get_score_object().get_derived_object(),
                                 IMP.core.SoftSpherePairScore)):
                evs.append((w * d.get_weight(), d))
            elif type(d) in (IMP.core.PairRestraint,
                             IMP.core.DistanceRestraint):
                ps = d.get_score_object()
                try:
                    ps = ps.get_derived_object()
                except Exception:
                    pass
                if isinstance(ps, IMP.core.SphereDistancePairScore):
                    ps = ps._proxied
                i, j = (int(x) for x in d.get_index())
                ww = w * d.get_weight()
                if isinstance(ps, IMP.core
                              .HarmonicUpperBoundSphereDistancePairScore):
                    hub.append((i, j, 1., ps.get_x0(), ps.get_k(), ww, 0))
                    return
                if isinstance(ps, IMP.core.DistancePairScore):
                    uf = ps.get_score_functor().get_unary_function() \
                        .get_derived_object()
                    mode = {IMP.core.HarmonicUpperBound: 0,
                            IMP.core.Harmonic: 1,
                            IMP.core.HarmonicLowerBound: 2}.get(type(uf))
                    if mode is not None:
                        hub.append((i, j, 0., uf.get_mean(), uf.get_k(), ww,
                                    mode))
                        return
                    if isinstance(uf, IMP.core.Linear):
                        if uf.get_slope() != 0.:
                            hub.append((i, j, 0., uf.get_offset(),
                                        uf.get_slope(), ww, 3))
                        return
                raise Unsupported(f'pair score {type(ps).__name__}')
            elif isinstance(d, IMP.npc.ZAxialPositionRestraint):
                p = J.zaxial_params_from_restraint(d)
                ww = w * d.get_weight() / p['sigma']
                for pi in p['pis']:
                    zax.append((int(pi), p['lower'], p['upper'], ww))
            elif isinstance(d, IMP.npc.MembraneSurfaceLocationRestraint):
                p = J.membrane_params_from_restraint(d)
                ww = w * d.get_weight() / p['sigma']
                for pi in p['pis']:
                    msl.append((int(pi), p['R'], p['r'], p['half_thickness'],
                                ww))
            else:
                ins = [int(p.get_index()) for p in
                       IMP.get_input_particles(d.get_inputs())]
                if ins and all(is_nuisance(p) for p in ins):
                    nleaves.append((w, d))
                else:
                    raise Unsupported(f'restraint {type(d).__name__}')
        walk(rs, 1.0)
        if len(ems) > 1:
            raise Unsupported('more than one EM restraint')
        self.n_hub = len(hub)
        self.n_leaves = len(zax) + len(msl) + len(nleaves) + len(xlw)

        # cross-link contributions, grouped per LogWrapper
        xls = []
        for w, rs_ in xlw:
            r0 = rs_[0]
            length = r0.get_length()
            slope = r0.get_slope() if r0.get_has_slope() else None
            for r in rs_:
                if (r.get_is_length_variable() or r.get_length() != length
                        or r.get_has_slope() != (slope is not None)
                        or (slope is not None and r.get_slope() != slope)
                        or r.get_log_prob()):
                    raise Unsupported('cross-link variant')
            cp, cs, cpsi, rcon = [], [], [], []
            for r in rs_:
                lst = []
                for k in range(r.get_number_of_contributions()):
                    lst.append(len(cp))
                    cp.append([int(x) for x in
                               r.get_contribution_particle_indexes(k)])
                    cs.append([int(x) for x in
                               r.get_contribution_sigma_indexes(k)])
                    cpsi.append(int(r.get_contribution_psi_index(k)))
                rcon.append(lst)
            xls.append(dict(w=w, length=length, slope=slope,
                            rw=np.array([r.get_weight() for r in rs_]),
                            cp=np.array(cp, np.int64),
                            cs=np.array(cs, np.int64),
                            cpsi=np.array(cpsi, np.int64), rcon=rcon))

        # ================= movers ======================================
        movs = []
        for mv in movers:
            d = mv.get_derived_object()
            if isinstance(d, IMP.core.SerialMover):
                movs += [x.get_derived_object() for x in d.get_movers()]
            else:
                movs.append(d)
        self.movers = movs
        nmov = len(movs)

        # ================= score states ================================
        nuis_bounds = {}
        for ss in m.get_ordered_score_states():
            try:
                d = ss.get_derived_object()
            except Exception:
                d = ss
            name = type(d).__name__
            if name in ('_RigidBodyNormalizeConstraint',
                        '_RigidBodyPositionConstraint'):
                continue
            if isinstance(d, IMP.isd.NuisanceScoreState):
                n = IMP.isd.Nuisance(m, d.get_index())
                if n.get_has_upper_particle() or n.get_has_lower_particle():
                    raise Unsupported('particle-valued nuisance bounds')
                lo = n.get_lower() if n.get_has_lower_float() else -math.inf
                up = n.get_upper() if n.get_has_upper_float() else math.inf
                nuis_bounds[int(d.get_index())] = (lo, up)
                continue
            if ss.get_is_ignored_by_jax():
                continue
            raise Unsupported(f'score state {name} {ss.get_name()}')

        # ================= rigid bodies ================================
        def body_of(p):
            return int(IMP.core.RigidMember(m, pi_(int(p))).get_rigid_body()
                       .get_particle_index())

        movable, tr_specs = [], []
        for d in movs:
            if isinstance(d, IMP.core.RigidBodyMover):
                movable.append(int(d.get_index()))
            elif isinstance(d, IMP.pmi.TransformMover):
                bodies, xyzs = [], []
                for o in d.get_inputs():
                    p = IMP.Particle.get_from(o)
                    if IMP.core.RigidBody.get_is_setup(m, p.get_index()):
                        bodies.append(int(p.get_index()))
                    else:
                        xyzs.append(int(p.get_index()))
                movable += bodies
                tr_specs.append((bodies, xyzs))
        movable = sorted(set(movable))
        for b in movable:
            if IMP.core.RigidMember.get_is_setup(m, pi_(b)):
                raise Unsupported('mover on a nested rigid body')
        nb = len(movable)
        body_index = {b: i for i, b in enumerate(movable)}
        intcoord = m.get_internal_coordinates_numpy()
        is_rigid = m.get_numpy(_jax_rigid._RB_IS_RIGID_KEY)
        body_members = []
        for b in movable:
            rb = IMP.core.RigidBody(m, pi_(b))
            mem = [int(x) for x in rb.get_member_particle_indexes()]
            for x in rb.get_body_member_particle_indexes():
                if IMP.core.RigidBody(m, x).get_number_of_members():
                    raise Unsupported('nested body with its own members')
                mem.append(int(x))
            if any(is_rigid[x] == 0 for x in mem):
                raise Unsupported('non-rigid members')
            body_members.append(mem)
        bnmem = [IMP.core.RigidBody(m, pi_(b)).get_number_of_members()
                 for b in movable]

        # ================= compact rows ================================
        used = set(movable) | set(nuis_bounds)
        for mem in body_members:
            used |= set(mem)
        for i, j, *_ in hub:
            used |= {i, j}
        used |= set(z[0] for z in zax) | set(z[0] for z in msl)
        for x in xls:
            used |= set(x['cp'].ravel().tolist()) | set(
                x['cs'].ravel().tolist()) | set(x['cpsi'].tolist())
        for _, d in nleaves:
            used |= set(int(p.get_index()) for p in
                        IMP.get_input_particles(d.get_inputs()))
        emp = J.em_params_from_restraint(ems[0][1]) if ems else None
        if emp is not None:
            used |= set(int(x) for x in emp.model_pis) | {emp.sigma_pi}
        evps = [J.ev_params_from_restraint(d) for _, d in evs]
        for p in evps:
            used |= set(int(x) for x in p['pis'])
        for d in movs:
            if isinstance(d, (IMP.core.BallMover, IMP.core.NormalMover)):
                used |= set(int(x) for x in d.get_indexes())
        for bodies, xyzs in tr_specs:
            used |= set(bodies) | set(xyzs)
        rows = np.array(sorted(used), np.int64)
        K = len(rows)
        SINK, NANROW = K, K + 1
        xyz_full, r_full = m.get_spheres_numpy()
        nattr = max(len(xyz_full), len(m.get_numpy(scale_key)),
                    int(rows.max()) + 1)
        table = np.full(nattr, NANROW, np.int32)
        table[rows] = np.arange(K, dtype=np.int32)
        self.rows, self.K, self.table = rows, K, table
        remap = imp_jax_fused._Remap(table, K + 2)

        def compact(full, fill=0.):
            full = np.asarray(full)
            out = np.full((K + 2,) + full.shape[1:], fill, np.float64)
            ok = rows < len(full)
            out[:K][ok] = full[rows[ok]]
            return out
        self._compact = compact
        radius = compact(r_full)
        lower = np.full(K + 2, -np.inf)
        upper = np.full(K + 2, np.inf)
        for p, (lo, up) in nuis_bounds.items():
            lower[table[p]], upper[table[p]] = lo, up

        # ================= EV ==========================================
        evt = []
        for (w, _), p in zip(evs, evps):
            n = len(p['pis'])
            evt.append(dict(
                n=n, w=w, k=p['k'],
                row_pos={int(table[x]): i for i, x in enumerate(p['pis'])},
                rows=np.concatenate([table[p['pis']], [SINK]]),
                r=np.concatenate([radius[table[p['pis']]], [0.]]),
                label=np.concatenate([p['label'], [-10 ** 9]])))
        self.nev = [e['n'] for e in evt]

        # ================= EM (model Gaussians grouped by owner) ========
        BS = block
        if emp is not None:
            M, D = len(emp.model_pis), len(emp.density_pis)
            quat_full = m.get_numpy(_jax_rigid._RB_QUAT_KEY)
            lquat_full = m.get_numpy(_jax_rigid._RB_LQUAT_KEY)
            owner = []
            for p in emp.model_pis:
                own = (body_of(p) if IMP.core.RigidMember.get_is_setup(
                    m, pi_(int(p))) else None)
                owner.append(body_index.get(own, -1)
                             if cov_mode == 'rotate' else -1)
            order = sorted(range(M), key=lambda g: (owner[g] < 0, owner[g],
                                                    g))
            nblk = (M + BS - 1) // BS
            MB = nblk * BS
            DUMMY = MB                      # padding index, outside blocks
            em_rows = np.full(MB + 1, SINK, np.int32)
            em_owner = np.full(MB + 1, -1, np.int32)
            em_cloc = np.zeros((MB + 1, 3, 3))
            em_mass = np.zeros(MB + 1)
            em_rad = np.zeros(MB + 1)
            em_row_pos = {}
            for e, g in enumerate(order):
                p = int(emp.model_pis[g])
                em_rows[e] = table[p]
                em_row_pos[int(table[p])] = e
                em_owner[e] = owner[g]
                em_mass[e], em_rad[e] = emp.model_mass[g], emp.model_radius[g]
                if owner[g] >= 0:
                    Rl = J.quat_to_rot(lquat_full[p], np)
                    em_cloc[e] = J.gaussian_covariance(Rl, emp.model_var[g],
                                                       np)
                elif cov_mode == 'rotate':
                    Rg = J.quat_to_rot(quat_full[p], np)
                    em_cloc[e] = J.gaussian_covariance(Rg, emp.model_var[g],
                                                       np)
                else:
                    em_cloc[e] = np.asarray(IMP.core.Gaussian(
                        m, pi_(p)).get_global_covariance())
            em_w = ems[0][0]
            em_cut = emp.cutoff + em_extra
            em_sigma_row = int(table[emp.sigma_pi])
        else:
            M, D, nblk, MB, DUMMY = 0, 1, 0, 0, 0
            em_rows = np.full(1, SINK, np.int32)
            em_owner = np.full(1, -1, np.int32)
            em_cloc = np.zeros((1, 3, 3))
            em_mass, em_rad = np.zeros(1), np.zeros(1)
            em_row_pos = {}
        self.M, self.D, self.nblk = M, D, nblk

        # ================= move specs (per mover) ======================
        specs = []
        tri = 0
        for d in movs:
            sp = dict(rows=[], code=[], slot=[], loc=[], cw=[], bodies=[])
            if isinstance(d, IMP.core.BallMover):
                idx = [int(x) for x in d.get_indexes()]
                keys = set(k.get_string() for k in d.get_keys())
                if (len(idx) != 1 or keys != set(
                        k.get_string() for k in IMP.core.XYZ.get_xyz_keys())):
                    raise Unsupported('BallMover variant')
                if IMP.core.RigidMember.get_is_setup(m, pi_(idx[0])):
                    raise Unsupported('BallMover on a rigid member')
                sp.update(kind=KIND_BALL, radius=d.get_radius())
                sp['rows'] = [int(table[idx[0]])]
                sp['code'], sp['slot'], sp['loc'] = [C_BALL], [0], [[0, 0, 0]]
                sp['cw'] = [0.]
            elif isinstance(d, (IMP.core.RigidBodyMover,
                                IMP.pmi.TransformMover)):
                if isinstance(d, IMP.core.RigidBodyMover):
                    bodies, xyzs = [int(d.get_index())], []
                    sp['kind'] = KIND_RB
                else:
                    bodies, xyzs = tr_specs[tri]
                    tri += 1
                    sp['kind'] = KIND_TR
                sp.update(tr=d.get_maximum_translation(),
                          rot=d.get_maximum_rotation())
                for s_, b in enumerate(bodies):
                    bi = body_index[b]
                    sp['bodies'].append(bi)
                    sp['rows'].append(int(table[b]))
                    sp['code'].append(C_CENTER)
                    sp['slot'].append(s_)
                    sp['loc'].append([0, 0, 0])
                    # RigidBodyMover rotates about the body center;
                    # TransformMover about the member-weighted centroid
                    sp['cw'].append(1. if sp['kind'] == KIND_RB
                                    else float(bnmem[bi]))
                    for x in body_members[bi]:
                        sp['rows'].append(int(table[x]))
                        sp['code'].append(C_MEMBER)
                        sp['slot'].append(s_)
                        sp['loc'].append(intcoord[x])
                        sp['cw'].append(0.)
                for x in xyzs:
                    sp['rows'].append(int(table[x]))
                    sp['code'].append(C_FREE)
                    sp['slot'].append(0)
                    sp['loc'].append([0, 0, 0])
                    sp['cw'].append(1.)
            elif isinstance(d, IMP.core.NormalMover):
                idx = [int(x) for x in d.get_indexes()]
                keys = list(d.get_keys())
                if len(idx) != 1 or len(keys) != 1 or keys[0] != scale_key:
                    raise Unsupported('NormalMover variant')
                sp.update(kind=KIND_NORMAL, nrow=int(table[idx[0]]),
                          sigma=d.get_sigma())
            else:
                raise Unsupported(f'mover {type(d).__name__}')
            specs.append(sp)
        # rows the movers change (for compact trajectories): coordinates
        # set directly (body centers, beads, free particles), rigid-body
        # members (derived from their body), nuisances
        self.dof_rows = np.array(sorted(
            {r_ for sp in specs if sp['kind'] != KIND_NORMAL
             for r_, c_ in zip(sp['rows'], sp['code']) if c_ != C_MEMBER}),
            np.int64)
        self.member_rows = np.array(sorted(
            {r_ for sp in specs if sp['kind'] != KIND_NORMAL
             for r_, c_ in zip(sp['rows'], sp['code']) if c_ == C_MEMBER}),
            np.int64)
        self.nuis_dof_rows = np.array(sorted(
            {sp['nrow'] for sp in specs if sp['kind'] == KIND_NORMAL}),
            np.int64)

        # ---- affected items per mover --------------------------------
        hub_a = np.array(hub, np.float64).reshape(-1, 7)
        hub_ri = table[hub_a[:, 0].astype(np.int64)] if len(hub) else \
            np.zeros(0, np.int32)
        hub_rj = table[hub_a[:, 1].astype(np.int64)] if len(hub) else \
            np.zeros(0, np.int32)
        row_hub = {}
        for q, (a_, b_) in enumerate(zip(hub_ri, hub_rj)):
            row_hub.setdefault(int(a_), []).append(q)
            row_hub.setdefault(int(b_), []).append(q)
        zax_r = np.array([table[z[0]] for z in zax], np.int32)
        msl_r = np.array([table[z[0]] for z in msl], np.int32)
        row_zax, row_msl = {}, {}
        for q, r_ in enumerate(zax_r):
            row_zax.setdefault(int(r_), []).append(q)
        for q, r_ in enumerate(msl_r):
            row_msl.setdefault(int(r_), []).append(q)
        for x in xls:
            x['crow'] = table[x['cp']].astype(np.int32)       # (nc, 2)
            x['rowres'] = {}
            for ri_, lst in enumerate(x['rcon']):
                for c in lst:
                    for r_ in x['crow'][c]:
                        x['rowres'].setdefault(int(r_), set()).add(ri_)
            x['cmax'] = max(len(l) for l in x['rcon'])
            x['nres'] = len(x['rcon'])

        for sp in specs:
            slot_of = {r_: s_ for s_, r_ in enumerate(sp['rows'])}
            sp['hub'] = sorted(set(q for r_ in sp['rows']
                                   for q in row_hub.get(r_, [])))
            sp['zax'] = sorted(set(q for r_ in sp['rows']
                                   for q in row_zax.get(r_, [])))
            sp['msl'] = sorted(set(q for r_ in sp['rows']
                                   for q in row_msl.get(r_, [])))
            sp['ev'] = [[e['row_pos'][r_] for r_ in sp['rows']
                         if r_ in e['row_pos']] for e in evt]
            sp['ev_slot'] = [[slot_of[int(e['rows'][i])] for i in lst]
                             for e, lst in zip(evt, sp['ev'])]
            em = [em_row_pos[r_] for r_ in sp['rows'] if r_ in em_row_pos]
            sp['em'] = em
            sp['em_slot'] = [slot_of[int(em_rows[e])] for e in em]
            sp['em_bslot'] = [(sp['bodies'].index(em_owner[e])
                               if em_owner[e] >= 0
                               and em_owner[e] in sp['bodies'] else -1)
                              for e in em]
            blocks = sorted(set(e // BS for e in em))
            sp['em_blocks'] = blocks
            pos = {e: k for k, e in enumerate(em)}
            sp['em_bmap'] = [[pos.get(b * BS + k, -1) for k in range(BS)]
                             for b in blocks]
            sp['xl'] = [sorted(set(rr for r_ in sp['rows']
                                   for rr in x['rowres'].get(r_, ())))
                        for x in xls]
            # where each affected item's particles sit among the moved rows
            sp['hub_si'] = [slot_of.get(int(hub_ri[q]), -1) for q in sp['hub']]
            sp['hub_sj'] = [slot_of.get(int(hub_rj[q]), -1) for q in sp['hub']]
            sp['zax_s'] = [slot_of.get(int(zax_r[q]), -1) for q in sp['zax']]
            sp['msl_s'] = [slot_of.get(int(msl_r[q]), -1) for q in sp['msl']]
            sp['xl_s'] = []
            for x, lst in zip(xls, sp['xl']):
                tab = np.full((max(1, len(lst)), x['cmax'], 2), -1, np.int32)
                for a_, rr in enumerate(lst):
                    for c_, con in enumerate(x['rcon'][rr]):
                        for e_ in range(2):
                            tab[a_, c_, e_] = slot_of.get(
                                int(x['crow'][con][e_]), -1)
                sp['xl_s'].append(tab)
        self.specs = specs
        kind = np.array([sp['kind'] for sp in specs], np.int32)
        self.kind = kind
        self.n_kind = np.bincount(kind, minlength=4)

        # ================= device constants =============================
        f = lambda a: jnp.asarray(a, dt)                       # noqa: E731
        i32 = lambda a: jnp.asarray(a, jnp.int32)             # noqa: E731
        G = dict(radius=f(radius), lower=f(lower), upper=f(upper),
                 movers=i32(np.arange(nmov)))
        if len(hub):
            G.update(hub_ri=i32(hub_ri), hub_rj=i32(hub_rj),
                     hub_sph=f(hub_a[:, 2]), hub_x0=f(hub_a[:, 3]),
                     hub_k=f(hub_a[:, 4]), hub_w=f(hub_a[:, 5]),
                     hub_mode=i32(hub_a[:, 6]))
        if zax:
            za = np.array(zax, np.float64)
            G.update(zax_r=i32(zax_r), zax_lo=f(za[:, 1]), zax_up=f(za[:, 2]),
                     zax_w=f(za[:, 3]))
        if msl:
            ma = np.array(msl, np.float64)
            G.update(msl_r=i32(msl_r), msl_R=f(ma[:, 1]), msl_rr=f(ma[:, 2]),
                     msl_h=f(ma[:, 3]), msl_w=f(ma[:, 4]))
        XG = []
        for x in xls:
            rcon = _pad2(x['rcon'], len(x['cp']))           # (nres, cmax)
            XG.append(dict(
                crow=i32(np.concatenate([x['crow'], [[SINK, SINK]]])),
                cs=i32(np.concatenate([table[x['cs']], [[SINK, SINK]]])),
                cpsi=i32(np.concatenate([table[x['cpsi']], [SINK]])),
                same=jnp.asarray(np.concatenate(
                    [x['cp'][:, 0] == x['cp'][:, 1], [False]])),
                rcon=i32(rcon), rw=f(x['rw'])))
        EV = [dict(rows=i32(e['rows']), r=f(e['r']),
                   label=jnp.asarray(e['label'])) for e in evt]
        if emp is not None:
            G.update(em_rows=i32(em_rows), em_owner=i32(em_owner),
                     em_cloc=f(em_cloc), em_mass=f(em_mass),
                     em_rad=f(em_rad), d_xyz=f(emp.density_xyz),
                     d_cov=f(emp.density_cov), d_mass=f(emp.density_mass),
                     d_rad=f(emp.density_radius), d_sdd=f(emp.score_dd))

        class _EMView:
            pass
        if emp is not None:
            _EMView.density_xyz = G['d_xyz']
            _EMView.density_cov = G['d_cov']
            _EMView.density_mass = G['d_mass']
            _EMView.density_radius = G['d_rad']
            _EMView.cutoff = em_cut
        nleaf_fns = [(w, d._get_jax(IMP.jax.FreeSpace).score_func)
                     for w, d in nleaves]

        # ---- per-kind tables (indexed by kind-local mover id) ----------
        present = [k for k in range(4) if self.n_kind[k]]
        kloc = np.zeros(nmov, np.int32)
        T = {}
        for k in present:
            ids = [i for i in range(nmov) if kind[i] == k]
            for j, i in enumerate(ids):
                kloc[i] = j
            S = [specs[i] for i in ids]
            t = {}
            if k == KIND_NORMAL:
                t['nrow'] = i32([s['nrow'] for s in S])
                t['sigma'] = f([s['sigma'] for s in S])
                T[k] = t
                continue
            P = max(len(s['rows']) for s in S)
            t['rows'] = i32(_pad2([s['rows'] for s in S], SINK, P))
            t['code'] = i32(_pad2([s['code'] for s in S], C_PAD, P))
            t['slot'] = i32(_pad2([s['slot'] for s in S], 0, P))
            t['loc'] = f(_padf([s['loc'] for s in S], P, (3,)))
            t['cw'] = f(_padf([s['cw'] for s in S], P))
            if k == KIND_BALL:
                t['radius'] = f([s['radius'] for s in S])
            else:
                kb = max(len(s['bodies']) for s in S)
                t['bodies'] = i32(_pad2([s['bodies'] for s in S], nb, kb))
                t['tr'] = f([s['tr'] for s in S])
                t['rot'] = f([s['rot'] for s in S])
            t['hub'] = i32(_pad2([s['hub'] for s in S], -1))
            t['zax'] = i32(_pad2([s['zax'] for s in S], -1))
            t['msl'] = i32(_pad2([s['msl'] for s in S], -1))
            t['ev'] = [i32(_pad2([s['ev'][q] for s in S], e['n']))
                       for q, e in enumerate(evt)]
            t['ev_slot'] = [i32(_pad2([s['ev_slot'][q] for s in S], 0))
                            for q in range(len(evt))]
            t['em'] = i32(_pad2([s['em'] for s in S], DUMMY))
            t['em_slot'] = i32(_pad2([s['em_slot'] for s in S], 0))
            t['em_bslot'] = i32(_pad2([s['em_bslot'] for s in S], -1))
            t['em_blocks'] = i32(_pad2([s['em_blocks'] for s in S], nblk))
            nbk = t['em_blocks'].shape[1]
            bm = np.full((len(S), nbk, BS), -1, np.int32)
            for j, s in enumerate(S):
                for q, row in enumerate(s['em_bmap']):
                    bm[j, q] = row
            t['em_bmap'] = i32(bm)
            t['xl'] = [i32(_pad2([s['xl'][q] for s in S], -1))
                       for q in range(len(xls))]
            t['hub_si'] = i32(_pad2([s['hub_si'] for s in S], -1,
                                    t['hub'].shape[1]))
            t['hub_sj'] = i32(_pad2([s['hub_sj'] for s in S], -1,
                                    t['hub'].shape[1]))
            t['zax_s'] = i32(_pad2([s['zax_s'] for s in S], -1,
                                   t['zax'].shape[1]))
            t['msl_s'] = i32(_pad2([s['msl_s'] for s in S], -1,
                                   t['msl'].shape[1]))
            t['xl_s'] = []
            for q, x in enumerate(xls):
                A = t['xl'][q].shape[1]
                tab = np.full((len(S), A, x['cmax'], 2), -1, np.int32)
                for j, s_ in enumerate(S):
                    tt_ = s_['xl_s'][q]
                    tab[j, :len(s_['xl'][q])] = tt_[:len(s_['xl'][q])]
                t['xl_s'].append(i32(tab))
            T[k] = t
        G['kloc'] = i32(kloc)
        branch_of_kind = np.zeros(4, np.int32)
        for i, k in enumerate(present):
            branch_of_kind[k] = i
        G['branch'] = i32(branch_of_kind[kind])
        self.pad_sizes = {k: {n: (int(v.shape[1]) if hasattr(v, 'shape')
                                  and v.ndim > 1 else None)
                              for n, v in T[k].items()
                              if not isinstance(v, list)}
                          for k in T}
        self.G, self.T = G, T

        # ================= math helpers =================================
        def qmul(a, b):
            return jnp.stack([
                a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
                - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3],
                a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
                + a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2],
                a[..., 0] * b[..., 2] - a[..., 1] * b[..., 3]
                + a[..., 2] * b[..., 0] + a[..., 3] * b[..., 1],
                a[..., 0] * b[..., 3] + a[..., 1] * b[..., 2]
                - a[..., 2] * b[..., 1] + a[..., 3] * b[..., 0]], -1)

        def in_sphere(u, rad):
            phi = 2. * math.pi * u[0]
            ct = 2. * u[1] - 1.
            st = jnp.sqrt(jnp.maximum(1. - ct * ct, 0.))
            r = rad * jnp.cbrt(u[2])
            return jnp.stack([r * st * jnp.cos(phi), r * st * jnp.sin(phi),
                              r * ct])

        def random_rot(u, max_angle):
            phi = 2. * math.pi * u[0]
            ct = 2. * u[1] - 1.
            st = jnp.sqrt(jnp.maximum(1. - ct * ct, 0.))
            axis = jnp.stack([st * jnp.cos(phi), st * jnp.sin(phi), ct])
            ang = max_angle * (2. * u[2] - 1.)
            return jnp.concatenate([jnp.cos(ang / 2.)[None],
                                    axis * jnp.sin(ang / 2.)])

        def normq(q):
            return q / jnp.linalg.norm(q, axis=-1, keepdims=True)

        def rot(q):
            return J.quat_to_rot(q, jnp)

        eye = jnp.eye(3, dtype=dt)

        # ---- item energies -------------------------------------------
        def hub_e(q, xi, xj):
            dd = jnp.sqrt(jnp.sum((xi - xj) ** 2, axis=-1))
            t = (dd - G['hub_sph'][q] * (G['radius'][G['hub_ri'][q]]
                                         + G['radius'][G['hub_rj'][q]])
                 - G['hub_x0'][q])
            mode = G['hub_mode'][q]
            tc = jnp.where(mode == 0, jnp.maximum(t, 0.),
                           jnp.where(mode == 2, jnp.minimum(t, 0.), t))
            return G['hub_w'][q] * G['hub_k'][q] * jnp.where(
                mode == 3, t, 0.5 * tc * tc)

        def zax_e(q, z):
            lo = jnp.minimum(z - G['zax_lo'][q], 0.)
            up = jnp.maximum(z - G['zax_up'][q], 0.)
            return G['zax_w'][q] * (lo * lo + up * up)

        def msl_e(q, x):
            dd = J.half_torus_distance(x, G['msl_R'][q], G['msl_rr'][q], jnp)
            return G['msl_w'][q] * jnp.where(jnp.abs(dd) > G['msl_h'][q],
                                             dd * dd, 0.)

        def xl_res_e(qx, X, res, xyz_c, nuis):
            """-log(w_r prob_r) for restraints res; xyz_c(rows, end) gives
               the (possibly overridden) positions of contribution end
               0 or 1."""
            x = xls[qx]
            con = X['rcon'][res]                         # (A, cmax)
            valid = con < len(x['cp'])
            crow = X['crow'][con]                        # (A, cmax, 2)
            p0, p1 = xyz_c(crow[..., 0], 0), xyz_c(crow[..., 1], 1)
            d = jnp.sqrt(jnp.sum((p0 - p1) ** 2, axis=-1))
            cs = X['cs'][con]
            logt = xl_terms(d, G['radius'][crow[..., 0]], None,
                            X['same'][con], nuis[cs[..., 0]],
                            nuis[cs[..., 1]], nuis[X['cpsi'][con]],
                            x['length'], x['slope'], jnp, log=True)
            # prob = 1 - prod(1 - t) as -expm1(sum log1p(-t)): in float32,
            # 1 - t rounds to 1 for t < 3e-8 (long links, e.g. after a
            # shuffle), so the naive form gives prob = 0 and score inf, and
            # it loses precision up to t ~ 1e-6. When even this underflows,
            # every t is tiny and prob = sum t.
            l1p = jnp.where(valid, jnp.log1p(-jnp.exp(logt)), 0.)
            prob = -jnp.expm1(jnp.sum(l1p, axis=-1))
            tiny = -jax.nn.logsumexp(jnp.where(valid, logt, -jnp.inf), -1)
            return (jnp.where(prob > 1e-30, -jnp.log(prob), tiny)
                    - jnp.log(X['rw'][res]))

        def em_energy(sdm, slope_sum, scale):
            if emp is None:
                return jnp.zeros((), dt)
            lt = jnp.log(scale * (sdm + 1e-10) / (G['d_sdd'] + 1e-10))
            return em_w * (D * 0.5 * jnp.log(jnp.sum(lt * lt))
                           + emp.cached_term + emp.slope * slope_sum)

        def em_rows(x, cov, mass, rad):
            return J.em_terms(x, cov, mass, rad, _EMView, jnp)

        def slope_of(min2):
            return (jnp.sum(jnp.sqrt(min2[:M])) if emp is not None
                    and emp.slope > 0. else 0.)

        # ================= full evaluation ==============================
        def full(xyz, quat, nuis):
            out = {}
            e = jnp.zeros((), dt)
            if len(hub) and 'cheap' not in ablate:
                q = jnp.arange(len(hub))
                out['hub'] = jnp.sum(hub_e(q, xyz[G['hub_ri']],
                                           xyz[G['hub_rj']]))
                e = e + out['hub']
            if zax and 'cheap' not in ablate:
                q = jnp.arange(len(zax))
                out['zax'] = jnp.sum(zax_e(q, xyz[G['zax_r'], 2]))
                e = e + out['zax']
            if msl and 'cheap' not in ablate:
                q = jnp.arange(len(msl))
                out['msl'] = jnp.sum(msl_e(q, xyz[G['msl_r']]))
                e = e + out['msl']
            out['xl'] = jnp.zeros((), dt)
            if 'cheap' not in ablate:
                for qx, (x, X) in enumerate(zip(xls, XG)):
                    res = jnp.arange(x['nres'])
                    out['xl'] = out['xl'] + x['w'] * jnp.sum(
                        xl_res_e(qx, X, res, lambda r, e: xyz[r], nuis))
            e = e + out['xl']
            out['nleaves'] = jnp.zeros((), dt)
            if nleaf_fns:
                jm = {'xyz': CompactArray(xyz, remap),
                      'r': CompactArray(G['radius'], remap),
                      'nuisance': CompactArray(nuis, remap)}
                for w, fn in nleaf_fns:
                    out['nleaves'] = out['nleaves'] + w * fn(jm)
            e = e + out['nleaves']
            evs_ = []
            for E, ev in zip(EV, evt):
                n = ev['n']
                xa = xyz[E['rows'][:n]]
                ra, la = E['r'][:n], E['label'][:n]
                evs_.append(ev['w'] * 0.5 * jnp.sum(J.ev_pair_energy(
                    xa, ra, la, xa, ra, la, ev['k'], jnp)))
            out['ev'] = sum(evs_) if evs_ else jnp.zeros((), dt)
            e = e + out['ev']
            if emp is not None:
                own = G['em_owner']
                R = jnp.where((own >= 0)[:, None, None],
                              rot(quat[jnp.maximum(own, 0)]), eye)
                cov = jnp.einsum('nij,njk,nlk->nil', R, G['em_cloc'], R)
                ov, min2 = em_rows(xyz[G['em_rows']], cov, G['em_mass'],
                                   G['em_rad'])
                ov = ov.at[M:].set(0.)
                min2 = min2.at[M:].set(0.)
                bs = jnp.concatenate([ov[:MB].reshape(nblk, BS, D).sum(1),
                                      jnp.zeros((1, D), dt)])
                out['em'] = em_energy(bs[:nblk].sum(0), slope_of(min2),
                                      nuis[em_sigma_row])
            else:
                ov = jnp.zeros((1, D), dt)
                bs = jnp.zeros((1, D), dt)
                min2 = jnp.zeros(1, dt)
                out['em'] = jnp.zeros((), dt)
            e = e + out['em']
            out['total'] = e
            return out, ov, bs, min2

        # ================= coordinate-move branches =====================
        def coord_branch(k):
            t = T[k]

            def branch(s, imov, u, z):
                j = G['kloc'][imov]
                rows_ = t['rows'][j]
                code = t['code'][j]
                slot = t['slot'][j]
                old = s.xyz[rows_]                               # (P, 3)
                if k == KIND_BALL:
                    v = in_sphere(u[0:3], t['radius'][j])
                    new = jnp.where((code == C_BALL)[:, None], old + v, old)
                    q_new = None
                else:
                    bodies = t['bodies'][j]                      # (kb,)
                    cw = t['cw'][j]
                    cen = jnp.sum(cw[:, None] * old, 0) / jnp.sum(cw)
                    trans = in_sphere(u[3:6], t['tr'][j])
                    qr = random_rot(u[6:9], t['rot'][j])
                    Rr = rot(qr)

                    def tt(x):
                        return (x - cen) @ Rr.T + cen + trans
                    q_new = normq(qmul(qr[None], s.quat[bodies]))
                    # body centers sit in the moved rows (code C_CENTER)
                    c_old = jnp.zeros((bodies.shape[0], 3), dt).at[
                        jnp.where(code == C_CENTER, slot, bodies.shape[0])
                    ].set(old, mode='drop')
                    c_new = tt(c_old)
                    Rb = rot(q_new)
                    mem = jnp.einsum('pij,pj->pi', Rb[slot],
                                     t['loc'][j]) + c_new[slot]
                    new = jnp.where(
                        (code == C_MEMBER)[:, None], mem,
                        jnp.where((code == C_CENTER)[:, None], c_new[slot],
                                  jnp.where((code == C_FREE)[:, None],
                                            tt(old), old)))

                def pos(r, sl):
                    return jnp.where((sl >= 0)[..., None],
                                     new[jnp.maximum(sl, 0)], s.xyz[r])
                dE = jnp.zeros((), dt)
                # pairs
                if len(hub) and 'cheap' not in ablate:
                    q = t['hub'][j]
                    ok = q >= 0
                    qq = jnp.maximum(q, 0)
                    ri, rj = G['hub_ri'][qq], G['hub_rj'][qq]
                    si, sj = t['hub_si'][j], t['hub_sj'][j]
                    e_new = hub_e(qq, pos(ri, si), pos(rj, sj))
                    e_old = hub_e(qq, s.xyz[ri], s.xyz[rj])
                    dE = dE + jnp.sum(jnp.where(ok, e_new - e_old, 0.))
                if zax and 'cheap' not in ablate:
                    q = t['zax'][j]
                    ok = q >= 0
                    qq = jnp.maximum(q, 0)
                    r_ = G['zax_r'][qq]
                    sl = t['zax_s'][j]
                    dE = dE + jnp.sum(jnp.where(
                        ok, zax_e(qq, pos(r_, sl)[:, 2])
                        - zax_e(qq, s.xyz[r_, 2]), 0.))
                if msl and 'cheap' not in ablate:
                    q = t['msl'][j]
                    ok = q >= 0
                    qq = jnp.maximum(q, 0)
                    r_ = G['msl_r'][qq]
                    sl = t['msl_s'][j]
                    dE = dE + jnp.sum(jnp.where(
                        ok, msl_e(qq, pos(r_, sl)) - msl_e(qq, s.xyz[r_]),
                        0.))
                if 'cheap' not in ablate:
                    for qx, (x, X) in enumerate(zip(xls, XG)):
                        res = t['xl'][qx][j]
                        ok = res >= 0
                        rr = jnp.maximum(res, 0)

                        xs_ = t['xl_s'][qx][j]          # (A, cmax, 2)
                        e_new = xl_res_e(
                            qx, X, rr,
                            lambda r, e, xs_=xs_: pos(r, xs_[..., e]),
                            s.nuis)
                        e_old = xl_res_e(qx, X, rr, lambda r, e: s.xyz[r],
                                         s.nuis)
                        dE = dE + x['w'] * jnp.sum(
                            jnp.where(ok, e_new - e_old, 0.))
                # excluded volume
                if 'ev' not in ablate:
                    for q, (E, ev) in enumerate(zip(EV, evt)):
                        n = ev['n']
                        if n == 0:
                            continue
                        idx = t['ev'][q][j]
                        sl = t['ev_slot'][q][j]
                        valid = idx < n
                        xa = s.xyz[E['rows'][:n]]
                        inP = jnp.zeros(n + 1, bool).at[idx].set(True)[:n]
                        mask = valid[:, None] & ~inP[None]
                        ra, la = E['r'][:n], E['label'][:n]
                        rm, lm = E['r'][idx], E['label'][idx]
                        en = J.ev_pair_energy(new[sl], rm, lm, xa, ra, la,
                                              ev['k'], jnp)
                        eo = J.ev_pair_energy(s.xyz[E['rows'][idx]], rm, lm,
                                              xa, ra, la, ev['k'], jnp)
                        dE = dE + ev['w'] * jnp.sum(
                            jnp.where(mask, en - eo, 0.))
                # EM
                if emp is not None and 'em' not in ablate:
                    ei = t['em'][j]                             # (Pem,)
                    bsl = t['em_bslot'][j]
                    if q_new is not None:
                        Rq = rot(q_new)
                        R = jnp.where((bsl >= 0)[:, None, None],
                                      Rq[jnp.maximum(bsl, 0)], eye)
                    else:
                        R = jnp.broadcast_to(eye, (ei.shape[0], 3, 3))
                    cov = jnp.einsum('nij,njk,nlk->nil', R,
                                     G['em_cloc'][ei], R)
                    ov_n, min2_n = em_rows(new[t['em_slot'][j]], cov,
                                           G['em_mass'][ei], G['em_rad'][ei])
                    blk = t['em_blocks'][j]                     # (nbk,)
                    bmap = t['em_bmap'][j]                      # (nbk, BS)
                    brow = jnp.minimum(blk[:, None] * BS
                                       + jnp.arange(BS)[None], MB)
                    cur = s.ov[brow]                            # (nbk,BS,D)
                    sub = jnp.where((bmap >= 0)[..., None],
                                    ov_n[jnp.maximum(bmap, 0)], cur)
                    bs_new = s.bs.at[blk].set(sub.sum(1))
                    min2_new = s.min2.at[ei].set(min2_n)
                    e_em = em_energy(bs_new[:nblk].sum(0),
                                     slope_of(min2_new),
                                     s.nuis[em_sigma_row])
                    dE = dE + (e_em - s.e_em)
                acc = (dE < 0.) | (u[9] < jnp.exp(-dE / s.temperature))
                xyz = s.xyz.at[rows_].set(jnp.where(acc, new, old))
                quat = s.quat
                if q_new is not None:
                    quat = quat.at[bodies].set(
                        jnp.where(acc, q_new, s.quat[bodies]))
                ov, bs, min2, e_em_s = s.ov, s.bs, s.min2, s.e_em
                if emp is not None and 'em' not in ablate:
                    ov = ov.at[ei].set(jnp.where(acc, ov_n, s.ov[ei]))
                    bs = jnp.where(acc, bs_new, s.bs)
                    min2 = jnp.where(acc, min2_new, s.min2)
                    e_em_s = jnp.where(acc, e_em, s.e_em)
                return finish(s, imov, acc, dE, xyz=xyz, quat=quat, ov=ov,
                              bs=bs, min2=min2, e_em=e_em_s)
            return branch

        def normal_branch(s, imov, u, z):
            t = T[KIND_NORMAL]
            j = G['kloc'][imov]
            row = t['nrow'][j]
            val = jnp.clip(s.nuis[row] + z * t['sigma'][j],
                           G['lower'][row], G['upper'][row])
            nuis = s.nuis.at[row].set(val)
            dE = jnp.zeros((), dt)
            if 'cheap' not in ablate:
                for qx, (x, X) in enumerate(zip(xls, XG)):
                    res = jnp.arange(x['nres'])
                    dE = dE + x['w'] * jnp.sum(
                        xl_res_e(qx, X, res, lambda r, e: s.xyz[r], nuis)
                        - xl_res_e(qx, X, res, lambda r, e: s.xyz[r],
                                   s.nuis))
            if nleaf_fns:
                def leaves(nu):
                    jm = {'xyz': CompactArray(s.xyz, remap),
                          'r': CompactArray(G['radius'], remap),
                          'nuisance': CompactArray(nu, remap)}
                    return sum(w * fn(jm) for w, fn in nleaf_fns)
                dE = dE + leaves(nuis) - leaves(s.nuis)
            e_em = s.e_em
            if emp is not None:
                e_em = em_energy(s.bs[:nblk].sum(0), slope_of(s.min2),
                                 nuis[em_sigma_row])
                dE = dE + (e_em - s.e_em)
            acc = (dE < 0.) | (u[9] < jnp.exp(-dE / s.temperature))
            return finish(s, imov, acc, dE,
                          nuis=s.nuis.at[row].set(
                              jnp.where(acc, val, s.nuis[row])),
                          e_em=jnp.where(acc, e_em, s.e_em))

        NPCState2 = _state_class()

        def finish(s, imov, acc, dE, **kw):
            d = dict(xyz=s.xyz, nuis=s.nuis, quat=s.quat, ov=s.ov, bs=s.bs,
                     min2=s.min2, e_em=s.e_em)
            d.update(kw)
            return NPCState2(
                score=s.score + jnp.where(acc, dE, 0.).astype(s.score.dtype),
                imov=jnp.asarray(imov, jnp.int32),
                temperature=s.temperature, rkey=s.rkey,
                accepted=s.accepted + acc,
                proposed_mover=s.proposed_mover.at[imov].add(1),
                accepted_mover=s.accepted_mover.at[imov].add(
                    acc.astype(jnp.int32)), **d)

        branches = {KIND_BALL: coord_branch(KIND_BALL) if KIND_BALL in T
                    else None,
                    KIND_RB: coord_branch(KIND_RB) if KIND_RB in T else None,
                    KIND_TR: coord_branch(KIND_TR) if KIND_TR in T else None,
                    KIND_NORMAL: normal_branch}
        blist = [branches[k] for k in present]

        def step(s, imov, u, z):
            if len(blist) == 1:
                return blist[0](s, imov, u, z)
            return jax.lax.switch(G['branch'][imov], blist, s, imov, u, z)

        def draws(key, nsteps):
            key, k1, k2 = jax.random.split(key, 3)
            return (key, jax.random.uniform(k1, (nsteps, 10), dtype=dt),
                    jax.random.normal(k2, (nsteps,), dtype=dt))

        def run(s, nsteps):
            s.rkey, U, Z = draws(s.rkey, nsteps)
            i0 = s.imov + 1
            return jax.lax.fori_loop(
                0, nsteps,
                lambda i, s: step(s, (i0 + i) % nmov, U[i], Z[i]), s)

        def run_batched(sb, nsteps):
            sb.rkey, U, Z = jax.vmap(draws, in_axes=(0, None))(sb.rkey,
                                                                 nsteps)
            i0 = sb.imov[0] + 1
            vstep = jax.vmap(step, in_axes=(0, None, 0, 0))
            return jax.lax.fori_loop(
                0, nsteps,
                lambda i, sb: vstep(sb, (i0 + i) % nmov, U[:, i], Z[:, i]),
                sb)

        def resync(s):
            out, ov, bs, min2 = full(s.xyz, s.quat, s.nuis)
            s.ov, s.bs, s.min2 = ov, bs, min2
            s.e_em = out['em']
            s.score = out['total'].astype(s.score.dtype)
            return s

        def terms(s):
            out, ov, bs, min2 = full(s.xyz, s.quat, s.nuis)
            cheap = out['total'] - out['ev'] - out['em']
            return dict(cheap=cheap, em=out['em'], ev=out['ev'],
                        em_running=s.e_em, total=out['total'],
                        total_running=s.score)

        self._full = full
        self._step = step
        self._run_py = run
        self.run = jax.jit(run, static_argnums=1)
        self.run_batched = jax.jit(run_batched, static_argnums=1)
        self.resync = jax.jit(resync)
        self.resync_batched = jax.jit(jax.vmap(resync))
        self.terms = jax.jit(terms)
        self.nb, self.nmov = nb, nmov
        self.movable = movable
        self.quat_key = _jax_rigid._RB_QUAT_KEY
        self.nuis_key = scale_key
        self.em_model_pis = emp.model_pis if emp is not None else []
        self._MB, self._nblk = MB, nblk

    # ------------------------------------------------------------------
    def initial_state(self, seed=0, temperature=None):
        import jax
        import jax.numpy as jnp
        m = self.m
        xyz = self._compact(m.get_spheres_numpy()[0])
        xyz[self.K:] = 1e6
        nuis = self._compact(m.get_numpy(self.nuis_key))
        q = np.zeros((self.nb + 1, 4))
        q[:, 0] = 1.
        if self.nb:
            q[:self.nb] = m.get_numpy(self.quat_key)[np.asarray(self.movable)]
        dt = self.dt
        z = jnp.int32(0)
        s = _state_class()(
            xyz=jnp.asarray(xyz, dt), nuis=jnp.asarray(nuis, dt),
            quat=jnp.asarray(q, dt),
            ov=jnp.zeros((self._MB + 1, self.D), dt),
            bs=jnp.zeros((self._nblk + 1, self.D), dt),
            min2=jnp.zeros(self._MB + 1, dt), e_em=jnp.zeros((), dt),
            score=jnp.zeros((), self.acc), imov=jnp.int32(-1),
            temperature=jnp.asarray(temperature or self.temperature0, dt),
            rkey=jax.random.key(seed), accepted=z,
            proposed_mover=jnp.zeros(self.nmov, jnp.int32),
            accepted_mover=jnp.zeros(self.nmov, jnp.int32))
        return self.resync(s)

    def _fast_update(self):
        """Model.update() with a cached score-state order (Model.update()
           recomputes dependencies each call: ~3 ms here)."""
        import IMP
        sss = self.m.get_score_states()
        ids = tuple(int(x.this) for x in sss)
        if getattr(self, '_upd', (None,))[0] != ids:
            self._upd = (ids, list(IMP.get_update_order(sss)))
        for x in self._upd[1]:
            x.before_evaluate()

    def write_model(self, s, update=True, refresh_covariances=True):
        import jax
        import IMP
        import IMP.core
        m = self.m
        xyz, nuis, quat = jax.device_get((s.xyz, s.nuis, s.quat))
        xyz = np.asarray(xyz, np.float64)
        full = m.get_spheres_numpy()[0]
        ok = self.rows < len(full)
        full[self.rows[ok]] = xyz[:self.K][ok]
        nf = m.get_numpy(self.nuis_key)
        ok = self.rows < len(nf)
        nf[self.rows[ok]] = np.asarray(nuis, np.float64)[:self.K][ok]
        if self.nb:
            qf = m.get_numpy(self.quat_key)
            q = np.asarray(quat, np.float64)[:self.nb]
            qf[np.asarray(self.movable)] = q / np.linalg.norm(
                q, axis=1, keepdims=True)
        if update:
            self._fast_update()
        if refresh_covariances:
            for p in self.em_model_pis:
                IMP.core.Gaussian(m, IMP.ParticleIndex(int(p))) \
                    .update_global_covariance()
