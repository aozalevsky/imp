"""Fused, incremental JAX Monte Carlo for PMI systems like NPC_Basket REX2.

One MC step = one sub-mover of PMI's SerialMover (same order as C++), the
Metropolis test, and the score update. The step works on compact arrays (only
particles that restraints, movers or rigid bodies use), as in the earlier
imp_jax_fused.py prototype, and adds what this system needs:

* movers: BallMover (one particle), RigidBodyMover, IMP.pmi.TransformMover
  (super rigid bodies, about the member-weighted centroid as in C++),
  NormalMover on ISD nuisances (with NuisanceScoreState bounds);
* rigid bodies with nested rigid members (the EM model Gaussians), whose
  orientation follows the parent body;
* incremental scoring of the two expensive terms:
    - excluded volume (PMI ExcludedVolumeSphere): energy change of the moved
      particles against all others, exact; the running total is resynced
      from scratch every `resync_every` frames;
    - GaussianEMRestraint: a cached model x data overlap matrix and nearest-
      center distance per model Gaussian; a move recomputes only the moved
      rows, then the score is rebuilt from column sums (no subtraction, so
      no cancellation);
* all other restraints are scored in full every step: connectivity pairs
  vectorized by score type, cross-links via the vectorized LogWrapper
  (imp_jax_patches xl_vec), the rest through their own _get_jax().

The mover kind is chosen with lax.switch so that a bead move only pays for
one particle (each branch has its own padding). Under vmap (many chains)
the switch becomes a select and all branches run.

EM covariances: by default from the current orientation (cov_mode='rotate').
C++ IMP keeps core::Gaussian's cached global covariance, which is not
refreshed when the parent rigid body rotates; cov_mode='frozen' reproduces
that (covariances fixed at their values when the sampler is created).
Model-data pairs: sphere distance < cutoff + em_extra (default 2 A, what a
freshly built C++ CloseBipartitePairContainer with slack 1 holds).
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


class Unsupported(Exception):
    pass


def _tname(o):
    return type(o).__module__.replace('IMP.', '') + '.' + type(o).__name__


def _derived(r):
    try:
        return r.get_derived_object()
    except NameError:
        return J._EM_REGISTRY[int(r.this)]


def _pad(rows, fill, width=None):
    width = max([len(r) for r in rows] + [1]) if width is None else width
    out = np.full((len(rows), width), fill, np.int32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


_STATE_CLS = None


def _state_class():
    global _STATE_CLS
    if _STATE_CLS is None:
        import jax

        @jax.tree_util.register_dataclass
        @dataclass
        class NPCState:
            xyz: jax.Array          # (K+2, 3)
            nuis: jax.Array         # (K+2,)
            quat: jax.Array         # (nb+1, 4) movable rigid bodies (+dummy)
            e_ev: jax.Array         # running EV energy per EV restraint
            ov: jax.Array           # (M+1, D) masked model x data overlaps
            min2: jax.Array         # (M+1,) squared nearest-center distance
            score: jax.Array
            imov: jax.Array
            temperature: jax.Array
            rkey: jax.Array
            accepted: jax.Array
            proposed_mover: jax.Array
            accepted_mover: jax.Array
        _STATE_CLS = NPCState
    return _STATE_CLS


class NPCSampler:
    def __init__(self, model, rs, movers, temperature=1.0, dtype='float32',
                 cov_mode='rotate', em_extra=2.0, use_switch=True,
                 ablate=()):
        import jax
        import jax.numpy as jnp
        import IMP
        import IMP.core
        import IMP.isd
        import IMP.pmi
        from IMP.core import _jax_rigid
        import imp_jax_fused
        import imp_jax_patches
        imp_jax_patches.apply(['xl_vec'])
        J.install()

        self.m = m = model
        self.rs = rs
        self.dt = dt = jnp.dtype(dtype)
        self.acc = jnp.float64 if jax.config.jax_enable_x64 else dt
        self.cov_mode = cov_mode
        self.use_switch = use_switch
        # timing experiments only: skip terms ('ev', 'em', 'cheap')
        ablate = set(ablate)
        self.temperature0 = float(temperature)
        CompactArray = imp_jax_fused._compact_array_class()
        pi_ = IMP.ParticleIndex

        # ---------------- restraints -> terms ----------------------------
        # pair restraints scored in one vectorized call:
        # (i, j, sphere?, mean, k, weight, mode): 0 harmonic upper bound,
        # 1 harmonic, 2 harmonic lower bound, 3 linear (k = slope);
        # zero-slope Linear restraints (PMI XL '_Linear') are dropped
        hub = []
        leaves = []     # (weight, derived restraint) generic _get_jax
        ems, evs = [], []

        def walk(r, w):
            d = _derived(r)
            name = type(d).__name__
            if isinstance(d, IMP.isd.LogWrapper):
                leaves.append((w, d))
            elif isinstance(d, IMP.RestraintSet):
                ww = w * d.get_weight()
                for i in range(d.get_number_of_restraints()):
                    walk(d.get_restraint(i), ww)
            elif name == 'GaussianEMRestraint':
                ems.append((w * d.get_weight(), d))
            elif (isinstance(d, IMP.container.PairsRestraint)
                  and d.get_container().get_type_name()
                  .startswith(('ClosePairContainer',
                               'CloseBipartitePairContainer'))
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
                        # (d - offset) * slope; k carries the slope
                        if uf.get_slope() != 0.:
                            hub.append((i, j, 0., uf.get_offset(),
                                        uf.get_slope(), ww, 3))
                        return
                leaves.append((w, d))
            else:
                leaves.append((w, d))
        walk(rs, 1.0)
        if len(ems) > 1:
            raise Unsupported('more than one EM restraint')
        self.n_hub, self.n_leaves = len(hub), len(leaves)

        # ---------------- movers ------------------------------------------
        movs = []
        for mv in movers:
            d = mv.get_derived_object()
            if isinstance(d, IMP.core.SerialMover):
                movs += [x.get_derived_object() for x in d.get_movers()]
            else:
                movs.append(d)
        self.movers = movs
        nmov = len(movs)

        # ---------------- score states ------------------------------------
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

        # ---------------- rigid bodies ------------------------------------
        def body_of(pi):
            return int(IMP.core.RigidMember(m, pi_(int(pi))).get_rigid_body()
                       .get_particle_index())

        movable = []          # top-level bodies moved by some mover
        tr_specs = []
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
        # members (non-body and nested-body particles) of each movable body
        body_members = []
        for b in movable:
            rb = IMP.core.RigidBody(m, pi_(b))
            mem = [int(x) for x in rb.get_member_particle_indexes()]
            mem += [int(x) for x in rb.get_body_member_particle_indexes()]
            for x in rb.get_body_member_particle_indexes():
                if IMP.core.RigidBody(m, x).get_number_of_members():
                    raise Unsupported('nested body with its own members')
            body_members.append(mem)
        is_rigid = m.get_numpy(_jax_rigid._RB_IS_RIGID_KEY)
        for mem in body_members:
            if any(is_rigid[x] == 0 for x in mem):
                raise Unsupported('non-rigid members')

        # ---------------- compact rows ------------------------------------
        used = set()
        for _, d in leaves:
            used |= set(int(p.get_index()) for p in
                        IMP.get_input_particles(d.get_inputs()))
        for i, j, *_ in hub:
            used |= {i, j}
        emp = None
        evps = [J.ev_params_from_restraint(d) for _, d in evs]
        for p in evps:
            used |= set(int(x) for x in p['pis'])
        if ems:
            emp = J.em_params_from_restraint(ems[0][1])
            used |= set(int(x) for x in emp.model_pis)
            used.add(emp.sigma_pi)
        used |= set(movable)
        for mem in body_members:
            used |= set(mem)
        used |= set(nuis_bounds)
        for d in movs:
            if isinstance(d, (IMP.core.BallMover, IMP.core.NormalMover)):
                used |= set(int(x) for x in d.get_indexes())
        for bodies, xyzs in tr_specs:
            used |= set(bodies) | set(xyzs)
        rows = np.array(sorted(used), np.int64)
        K = len(rows)
        SINK, NANROW = K, K + 1
        xyz_full, r_full = m.get_spheres_numpy()
        nattr = max(len(xyz_full), len(m.get_numpy(
            IMP.isd.Scale.get_scale_key())), int(rows.max()) + 1)
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
        # padding writes go to SINK; keep it far away from everything
        self._sink_xyz = 1e6

        lower = np.full(K + 2, -np.inf)
        upper = np.full(K + 2, np.inf)
        for p, (lo, up) in nuis_bounds.items():
            lower[table[p]], upper[table[p]] = lo, up

        # body tables (+ one dummy body nb with no members)
        bmem = _pad([table[np.asarray(mem, np.int64)] for mem in body_members]
                    + [[]], SINK)
        mmax = bmem.shape[1]
        bloc = np.zeros((nb + 1, mmax, 3))
        for b, mem in enumerate(body_members):
            bloc[b, :len(mem)] = intcoord[np.asarray(mem, np.int64)]
        brow = np.full(nb + 1, SINK, np.int32)
        brow[:nb] = table[np.asarray(movable, np.int64)] if nb else []
        bnmem = np.zeros(nb + 1)
        for b, p in enumerate(movable):
            bnmem[b] = IMP.core.RigidBody(m, pi_(p)).get_number_of_members()

        # ---------------- EV tables ----------------------------------------
        # one entry per EV restraint; particle lists padded with a dummy
        # (index n) that is masked out
        evt = []
        for (w, _), p in zip(evs, evps):
            n = len(p['pis'])
            evt.append(dict(
                n=n, w=w, k=p['k'],
                pos={int(x): i for i, x in enumerate(p['pis'])},
                rows=np.concatenate([table[p['pis']], [SINK]]),
                r=np.concatenate([radius[table[p['pis']]], [0.]]),
                label=np.concatenate([p['label'], [-10 ** 9]])))
        self.nev = [e['n'] for e in evt]
        # ---------------- EM tables ----------------------------------------
        if emp is not None:
            M = len(emp.model_pis)
            D = len(emp.density_pis)
            em_pos = {int(p): i for i, p in enumerate(emp.model_pis)}
            em_rows = np.concatenate([table[emp.model_pis], [SINK]])
            em_owner = np.full(M + 1, -1, np.int32)
            em_cloc = np.zeros((M + 1, 3, 3))
            quat_full = m.get_numpy(_jax_rigid._RB_QUAT_KEY)
            lquat_full = m.get_numpy(_jax_rigid._RB_LQUAT_KEY)
            for g, p in enumerate(emp.model_pis):
                pi = pi_(int(p))
                frozen = np.asarray(
                    IMP.core.Gaussian(m, pi).get_global_covariance())
                own = (body_of(p) if IMP.core.RigidMember.get_is_setup(m, pi)
                       else None)
                if own in body_index and cov_mode == 'rotate':
                    em_owner[g] = body_index[own]
                    Rl = J.quat_to_rot(lquat_full[int(p)], np)
                    em_cloc[g] = J.gaussian_covariance(Rl, emp.model_var[g],
                                                       np)
                elif cov_mode == 'rotate':
                    Rg = J.quat_to_rot(quat_full[int(p)], np)
                    em_cloc[g] = J.gaussian_covariance(Rg, emp.model_var[g],
                                                       np)
                else:
                    em_cloc[g] = frozen
            em_mass = np.concatenate([emp.model_mass, [0.]])
            em_rad = np.concatenate([emp.model_radius, [0.]])
            em_w = ems[0][0]
            self._emp = emp
            emp.cutoff_eff = emp.cutoff + em_extra
        else:
            M, D, em_pos = 0, 1, {}
            em_rows = np.array([SINK], np.int32)
            em_owner = np.full(1, -1, np.int32)
            em_cloc = np.zeros((1, 3, 3))
            em_mass, em_rad, em_w = np.zeros(1), np.zeros(1), 0.
        self.M, self.D = M, D

        def ev_idx(pis, e):
            return [e['pos'][int(p)] for p in pis if int(p) in e['pos']]

        def em_idx(pis):
            return [em_pos[int(p)] for p in pis if int(p) in em_pos]
        for e in evt:
            e['body'] = _pad([ev_idx(mem, e) for mem in body_members] + [[]],
                             e['n'])
            e['ball'] = np.full(nmov, e['n'], np.int32)
        body_em = _pad([em_idx(mem) for mem in body_members] + [[]], M)

        # ---------------- per-mover tables --------------------------------
        kind = np.zeros(nmov, np.int32)
        ball_row = np.full(nmov, SINK, np.int32)
        ball_rad = np.zeros(nmov)
        ball_em = np.full(nmov, M, np.int32)
        rb_body = np.full(nmov, nb, np.int32)
        mv_tr = np.zeros(nmov)
        mv_rot = np.zeros(nmov)
        tr_id = np.zeros(nmov, np.int32)
        nrm_row = np.full(nmov, SINK, np.int32)
        nrm_sig = np.zeros(nmov)
        nuis_key = IMP.isd.Scale.get_scale_key()
        xyz_keys = set(k.get_string() for k in IMP.core.XYZ.get_xyz_keys())
        it = 0
        for i, d in enumerate(movs):
            if isinstance(d, IMP.core.BallMover):
                idx = [int(x) for x in d.get_indexes()]
                if (len(idx) != 1 or set(k.get_string() for k in
                                         d.get_keys()) != xyz_keys):
                    raise Unsupported('BallMover variant')
                if IMP.core.RigidMember.get_is_setup(m, pi_(idx[0])):
                    raise Unsupported('BallMover on a rigid member')
                kind[i] = KIND_BALL
                ball_row[i], ball_rad[i] = table[idx[0]], d.get_radius()
                for e in evt:
                    if idx[0] in e['pos']:
                        e['ball'][i] = e['pos'][idx[0]]
                if idx[0] in em_pos:
                    ball_em[i] = em_pos[idx[0]]
            elif isinstance(d, IMP.core.RigidBodyMover):
                kind[i] = KIND_RB
                rb_body[i] = body_index[int(d.get_index())]
                mv_tr[i] = d.get_maximum_translation()
                mv_rot[i] = d.get_maximum_rotation()
            elif isinstance(d, IMP.pmi.TransformMover):
                kind[i] = KIND_TR
                tr_id[i] = it
                it += 1
                mv_tr[i] = d.get_maximum_translation()
                mv_rot[i] = d.get_maximum_rotation()
            elif isinstance(d, IMP.core.NormalMover):
                idx = [int(x) for x in d.get_indexes()]
                keys = list(d.get_keys())
                if len(idx) != 1 or len(keys) != 1 or keys[0] != nuis_key:
                    raise Unsupported('NormalMover variant')
                kind[i] = KIND_NORMAL
                nrm_row[i], nrm_sig[i] = table[idx[0]], d.get_sigma()
            else:
                raise Unsupported(f'mover {type(d).__name__}')
        # transform movers: bodies (padded with the dummy body), free
        # particles, centroid weights, moved EV/EM sets (+ EM body slot)
        tr_bodies = _pad([[body_index[b] for b in bs] for bs, _ in tr_specs]
                         or [[]], nb)
        tr_xyz = _pad([table[np.asarray(xs, np.int64)] for _, xs in tr_specs]
                      or [[]], SINK)
        tr_wb = np.where(tr_bodies < nb, bnmem[tr_bodies], 0.)
        tr_wx = np.where(tr_xyz != SINK, 1., 0.)
        tr_em, tr_em_slot = [], []
        for e in evt:
            e['tr'] = []
        for bs, xs in tr_specs or [([], [])]:
            eml, slots = [], []
            for e in evt:
                e['tr'].append(sum((ev_idx(body_members[body_index[b]], e)
                                    for b in bs), []) + ev_idx(xs, e))
            for s_, b in enumerate(bs):
                g = em_idx(body_members[body_index[b]])
                eml += g
                slots += [s_] * len(g)
            g = em_idx(xs)
            eml += g
            slots += [-1] * len(g)
            tr_em.append(eml)
            tr_em_slot.append(slots)
        for e in evt:
            e['tr'] = _pad(e['tr'], e['n'])
        tr_em = _pad(tr_em, M)
        tr_em_slot = _pad(tr_em_slot, -1)
        self.kind = kind
        self.n_kind = np.bincount(kind, minlength=4)

        # ---------------- constants on device -----------------------------
        f = lambda a: jnp.asarray(a, dt)                       # noqa: E731
        i32 = lambda a: jnp.asarray(a, jnp.int32)             # noqa: E731
        hub = np.array(hub, np.float64).reshape(-1, 7)
        C = dict(
            radius=f(radius), lower=f(lower), upper=f(upper),
            bmem=i32(bmem), bloc=f(bloc), brow=i32(brow),
            body_em=i32(body_em),
            em_rows=i32(em_rows), em_owner=i32(em_owner),
            em_cloc=f(em_cloc), em_mass=f(em_mass), em_rad=f(em_rad),
            kind=i32(kind), ball_row=i32(ball_row), ball_rad=f(ball_rad),
            ball_em=i32(ball_em), rb_body=i32(rb_body),
            mv_tr=f(mv_tr), mv_rot=f(mv_rot), tr_id=i32(tr_id),
            nrm_row=i32(nrm_row), nrm_sig=f(nrm_sig),
            tr_bodies=i32(tr_bodies), tr_xyz=i32(tr_xyz), tr_wb=f(tr_wb),
            tr_wx=f(tr_wx), tr_em=i32(tr_em),
            tr_em_slot=i32(tr_em_slot), movers=i32(np.arange(nmov)),
            hub_i=i32(table[hub[:, 0].astype(np.int64)]),
            hub_j=i32(table[hub[:, 1].astype(np.int64)]),
            hub_sph=f(hub[:, 2]), hub_x0=f(hub[:, 3]), hub_k=f(hub[:, 4]),
            hub_w=f(hub[:, 5]), hub_mode=i32(hub[:, 6]))
        if emp is not None:
            C.update(d_xyz=f(emp.density_xyz), d_cov=f(emp.density_cov),
                     d_mass=f(emp.density_mass), d_rad=f(emp.density_radius),
                     d_sdd=f(emp.score_dd))
            em_sigma_row = int(table[emp.sigma_pi])
        self.C = C
        EV = [dict(rows=i32(e['rows']), r=f(e['r']),
                   label=jnp.asarray(e['label']), body=i32(e['body']),
                   ball=i32(e['ball']), tr=i32(e['tr'])) for e in evt]
        import IMP.jax
        leaf_fns = []
        for w, d in leaves:
            ji = d._get_jax(IMP.jax.FreeSpace)
            leaf_fns.append((w, ji.score_func,
                             isinstance(d, IMP.isd.LogWrapper)))

        def view(xyz, nuis):
            return {'xyz': CompactArray(xyz, remap),
                    'r': CompactArray(C['radius'], remap),
                    'nuisance': CompactArray(nuis, remap)}

        # ---------------- math pieces --------------------------------------
        def qmul(a, b):
            return jnp.stack([
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])

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

        def ev_delta(xyz_old, xyz_new, idxs):
            """Change of each EV term when its particles idxs[t] (padded
               with n_t) move from xyz_old to xyz_new; others are unmoved."""
            out = []
            for t, (e, E) in enumerate(zip(evt, EV)):
                n, idx = e['n'], idxs[t]
                if n == 0 or 'ev' in ablate:
                    out.append(jnp.zeros((), self.acc))
                    continue
                xa = xyz_old[E['rows'][:n]]
                rm = E['rows'][idx]
                valid = idx < n
                inP = jnp.zeros(n + 1, bool).at[idx].set(True)[:n]
                mask = valid[:, None] & ~inP[None]
                ra, la = E['r'][:n], E['label'][:n]
                en = J.ev_pair_energy(xyz_new[rm], E['r'][idx],
                                      E['label'][idx], xa, ra, la, e['k'],
                                      jnp)
                eo = J.ev_pair_energy(xyz_old[rm], E['r'][idx],
                                      E['label'][idx], xa, ra, la, e['k'],
                                      jnp)
                out.append(jnp.sum(jnp.where(mask, en - eo, 0.))
                           .astype(self.acc))
            return jnp.stack(out) if out else jnp.zeros(1, self.acc)

        def em_rows_for(xyz, idx, rots):
            """Overlap rows and squared nearest distances for model
               Gaussians idx (padded with M); rots: rotation per entry
               (identity for constant covariance)."""
            x = xyz[C['em_rows'][idx]]
            cl = C['em_cloc'][idx]
            cov = jnp.einsum('nij,njk,nlk->nil', rots, cl, rots)
            ov, min2 = J.em_terms(x, cov, C['em_mass'][idx],
                                  C['em_rad'][idx], _EMView, jnp)
            return ov, min2

        class _EMView:
            pass
        if emp is not None:
            _EMView.density_xyz = C['d_xyz']
            _EMView.density_cov = C['d_cov']
            _EMView.density_mass = C['d_mass']
            _EMView.density_radius = C['d_rad']
            _EMView.cutoff = emp.cutoff_eff
        eye = jnp.eye(3, dtype=dt)

        def em_rot_for(quat, idx):
            own = C['em_owner'][idx]
            R = rot(quat[jnp.maximum(own, 0)])
            return jnp.where((own >= 0)[:, None, None], R, eye)

        def em_energy(ov, min2, nuis):
            if emp is None:
                return jnp.zeros((), dt)
            sdm = jnp.sum(ov[:M], axis=0)
            slope = (jnp.sum(jnp.sqrt(min2[:M])) if emp.slope > 0.
                     else 0.)
            sdd = C['d_sdd']
            lt = jnp.log(nuis[em_sigma_row] * (sdm + 1e-10) / (sdd + 1e-10))
            return em_w * (D * 0.5 * jnp.log(jnp.sum(lt * lt))
                           + emp.cached_term + emp.slope * slope)

        def cheap_energy(xyz, nuis):
            e = jnp.zeros((), dt)
            if 'cheap' in ablate:
                return e
            if len(hub) and 'hub' not in ablate:
                xi, xj = xyz[C['hub_i']], xyz[C['hub_j']]
                dd = jnp.sqrt(jnp.sum((xi - xj) ** 2, axis=-1))
                t = (dd - C['hub_sph'] * (C['radius'][C['hub_i']]
                                          + C['radius'][C['hub_j']])
                     - C['hub_x0'])
                mode = C['hub_mode']
                tc = jnp.where(mode == 0, jnp.maximum(t, 0.),
                               jnp.where(mode == 2, jnp.minimum(t, 0.), t))
                e = e + jnp.sum(C['hub_w'] * C['hub_k'] * jnp.where(
                    mode == 3, t, 0.5 * tc * tc))
            if leaf_fns:
                jm = view(xyz, nuis)
                for w, fn, is_xl in leaf_fns:
                    if ('xl' if is_xl else 'leaves') in ablate:
                        continue
                    e = e + w * fn(jm)
            return e

        def full_terms(xyz, quat, nuis):
            """EV energies, EM caches from scratch"""
            out = []
            for e, E in zip(evt, EV):
                n = e['n']
                xa = xyz[E['rows'][:n]]
                ra, la = E['r'][:n], E['label'][:n]
                out.append((0.5 * jnp.sum(J.ev_pair_energy(
                    xa, ra, la, xa, ra, la, e['k'], jnp))).astype(self.acc))
            e_ev = jnp.stack(out) if out else jnp.zeros(1, self.acc)
            if emp is None:
                return e_ev, jnp.zeros((M + 1, D), dt), jnp.zeros(M + 1, dt)
            idx = jnp.arange(M + 1, dtype=jnp.int32)
            ov, min2 = em_rows_for(xyz, idx, em_rot_for(quat, idx))
            return e_ev, ov, min2

        ev_w = jnp.asarray([e['w'] for e in evt] or [0.], dt)

        def total(e_ev, ov, min2, xyz, nuis):
            return (cheap_energy(xyz, nuis) + em_energy(ov, min2, nuis)
                    + jnp.sum(ev_w * e_ev.astype(dt)))

        # ---------------- move branches ------------------------------------
        def apply_bodies(s_xyz, quat, bodies, q_new, c_new):
            """Set frames of bodies (padded with nb) and their members."""
            mem = C['bmem'][bodies]                     # (nb_, mmax)
            loc = C['bloc'][bodies]                     # (nb_, mmax, 3)
            R = rot(q_new)                              # (nb_, 3, 3)
            x = jnp.einsum('bij,bmj->bmi', R, loc) + c_new[:, None]
            xyz = s_xyz.at[mem.reshape(-1)].set(x.reshape(-1, 3))
            xyz = xyz.at[C['brow'][bodies]].set(c_new)
            quat = quat.at[bodies].set(q_new)
            return xyz, quat

        def finish(s, xyz, quat, nuis, ev_i, em_i, em_R):
            e_ev = s.e_ev + ev_delta(s.xyz, xyz, ev_i)
            if M and 'em' not in ablate:
                ov_r, min2_r = em_rows_for(xyz, em_i, em_R)
                ov = s.ov.at[em_i].set(ov_r)
                min2 = s.min2.at[em_i].set(min2_r)
            else:
                ov, min2 = s.ov, s.min2
            return xyz, quat, nuis, e_ev, ov, min2

        def br_ball(s, imov, u, z):
            row = C['ball_row'][imov]
            xyz = s.xyz.at[row].add(in_sphere(u[0:3], C['ball_rad'][imov]))
            em_i = C['ball_em'][imov][None]
            return finish(s, xyz, s.quat, s.nuis,
                          [E['ball'][imov][None] for E in EV],
                          em_i, em_rot_for(s.quat, em_i))

        def br_rb(s, imov, u, z):
            b = C['rb_body'][imov]
            c_new = s.xyz[C['brow'][b]] + in_sphere(u[3:6], C['mv_tr'][imov])
            q_new = normq(qmul(random_rot(u[6:9], C['mv_rot'][imov]),
                               s.quat[b]))
            xyz, quat = apply_bodies(s.xyz, s.quat, b[None], q_new[None],
                                     c_new[None])
            em_i = C['body_em'][b]
            return finish(s, xyz, quat, s.nuis, [E['body'][b] for E in EV],
                          em_i, em_rot_for(quat, em_i))

        def br_tr(s, imov, u, z):
            t = C['tr_id'][imov]
            bodies = C['tr_bodies'][t]
            xr = C['tr_xyz'][t]
            cb = s.xyz[C['brow'][bodies]]
            cx = s.xyz[xr]
            wb, wx = C['tr_wb'][t], C['tr_wx'][t]
            cen = ((jnp.sum(wb[:, None] * cb, 0)
                    + jnp.sum(wx[:, None] * cx, 0))
                   / (jnp.sum(wb) + jnp.sum(wx)))
            trans = in_sphere(u[3:6], C['mv_tr'][imov])
            qr = random_rot(u[6:9], C['mv_rot'][imov])
            Rr = rot(qr)
            tt = lambda x: (x - cen) @ Rr.T + cen + trans   # noqa: E731
            q_new = normq(jax.vmap(lambda q: qmul(qr, q))(s.quat[bodies]))
            xyz, quat = apply_bodies(s.xyz, s.quat, bodies, q_new, tt(cb))
            xyz = xyz.at[xr].set(jnp.where((wx > 0)[:, None], tt(cx), cx))
            em_i = C['tr_em'][t]
            slot = C['tr_em_slot'][t]
            Rb = rot(quat[bodies])
            em_R = jnp.where((slot >= 0)[:, None, None],
                             Rb[jnp.maximum(slot, 0)], eye)
            # nested Gaussians use their owner's rotation (same as slot)
            em_R = jnp.where((C['em_owner'][em_i] >= 0)[:, None, None],
                             em_R, eye)
            return finish(s, xyz, quat, s.nuis, [E['tr'][t] for E in EV],
                          em_i, em_R)

        def br_normal(s, imov, u, z):
            row = C['nrm_row'][imov]
            val = jnp.clip(s.nuis[row] + z * C['nrm_sig'][imov],
                           C['lower'][row], C['upper'][row])
            return (s.xyz, s.quat, s.nuis.at[row].set(val), s.e_ev, s.ov,
                    s.min2)

        branches = [br_ball, br_rb, br_tr, br_normal]
        present = [k for k in range(4) if self.n_kind[k]]
        branch_of_kind = np.zeros(4, np.int32)
        for i, k in enumerate(present):
            branch_of_kind[k] = i
        C['branch'] = i32(branch_of_kind[kind])
        NPCState = _state_class()

        def step(s, imov, u, z):
            """One MC step with sub-mover imov (shared by all chains when
               vmapped, so lax.switch stays a real branch)."""
            if len(present) == 1:
                cand = branches[present[0]](s, imov, u, z)
            else:
                cand = jax.lax.switch(C['branch'][imov],
                                      [branches[k] for k in present],
                                      s, imov, u, z)
            xyz, quat, nuis, e_ev, ov, min2 = cand
            new = total(e_ev, ov, min2, xyz, nuis)
            acc = (new < s.score) | (u[9] < jnp.exp(
                -(new - s.score) / s.temperature))
            sel = lambda a, o: jnp.where(acc, a, o)       # noqa: E731
            hit = C['movers'] == imov
            return NPCState(
                xyz=sel(xyz, s.xyz), nuis=sel(nuis, s.nuis),
                quat=sel(quat, s.quat), e_ev=sel(e_ev, s.e_ev),
                ov=sel(ov, s.ov), min2=sel(min2, s.min2),
                score=sel(new, s.score), imov=jnp.asarray(imov, jnp.int32),
                temperature=s.temperature, rkey=s.rkey,
                accepted=s.accepted + acc,
                proposed_mover=s.proposed_mover + hit,
                accepted_mover=s.accepted_mover + (hit & acc))

        def draws(key, nsteps):
            key, k1, k2 = jax.random.split(key, 3)
            return (key, jax.random.uniform(k1, (nsteps, 10), dtype=dt),
                    jax.random.normal(k2, (nsteps,), dtype=dt))

        def run(s, nsteps):
            """nsteps MC steps of one chain"""
            s.rkey, U, Z = draws(s.rkey, nsteps)
            i0 = s.imov + 1
            return jax.lax.fori_loop(
                0, nsteps,
                lambda i, s: step(s, (i0 + i) % nmov, U[i], Z[i]), s)

        def run_batched(sb, nsteps):
            """nsteps MC steps of a batch of chains (leading axis). All
               chains are at the same SerialMover position."""
            sb.rkey, U, Z = jax.vmap(draws, in_axes=(0, None))(
                sb.rkey, nsteps)
            i0 = sb.imov[0] + 1
            vstep = jax.vmap(step, in_axes=(0, None, 0, 0))
            return jax.lax.fori_loop(
                0, nsteps,
                lambda i, sb: vstep(sb, (i0 + i) % nmov, U[:, i], Z[:, i]),
                sb)

        def resync(s):
            e_ev, ov, min2 = full_terms(s.xyz, s.quat, s.nuis)
            s.e_ev, s.ov, s.min2 = e_ev, ov, min2
            s.score = total(e_ev, ov, min2, s.xyz, s.nuis)
            return s

        def terms(s):
            """Per-term energies of a state (for checks)"""
            e_ev, ov, min2 = full_terms(s.xyz, s.quat, s.nuis)
            return dict(cheap=cheap_energy(s.xyz, s.nuis),
                        em=em_energy(ov, min2, s.nuis),
                        ev=jnp.sum(ev_w * e_ev),
                        ev_running=jnp.sum(ev_w * s.e_ev),
                        em_running=em_energy(s.ov, s.min2, s.nuis))

        self._step = step
        self._run_py = run
        self.run = jax.jit(run, static_argnums=1)
        self.run_batched = jax.jit(run_batched, static_argnums=1)
        self.resync_batched = jax.jit(jax.vmap(resync))
        self.resync = jax.jit(resync)
        self.terms = jax.jit(terms)
        self._full_terms = full_terms
        self._total = total
        self.nb, self.nmov = nb, nmov
        self.movable = movable
        self.body_members = body_members
        self.quat_key = _jax_rigid._RB_QUAT_KEY
        self.nuis_key = nuis_key
        self.em_model_pis = emp.model_pis if emp is not None else []

    # ------------------------------------------------------------------
    def initial_state(self, seed=0, temperature=None):
        import jax
        import jax.numpy as jnp
        m = self.m
        xyz = self._compact(m.get_spheres_numpy()[0])
        xyz[self.K:] = self._sink_xyz
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
            e_ev=jnp.zeros(max(1, len(self.nev)), self.acc),
            ov=jnp.zeros((self.M + 1, self.D), dt),
            min2=jnp.zeros(self.M + 1, dt), score=jnp.zeros((), dt),
            imov=jnp.int32(-1),
            temperature=jnp.asarray(temperature or self.temperature0, dt),
            rkey=jax.random.key(seed), accepted=z,
            proposed_mover=jnp.zeros(self.nmov, jnp.int32),
            accepted_mover=jnp.zeros(self.nmov, jnp.int32))
        return self.resync(s)

    def write_model(self, s, update=True, refresh_covariances=True):
        """Copy a state into the IMP Model (then Model.update() recomputes
           rigid members from the body frames)."""
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
            m.update()
        if refresh_covariances:
            for p in self.em_model_pis:
                IMP.core.Gaussian(m, IMP.ParticleIndex(int(p))) \
                    .update_global_covariance()
