"""Prototype: a fused, branch-free JAX Monte Carlo step on compact arrays.

Replaces IMP.core.MonteCarlo._get_jax_optimizer() when the system is of the
kind PMI builds for rigid bodies + beads + ISD nuisances, and falls back to
IMP's implementation otherwise. Supported cases:
  * a single SerialMover whose sub-movers are
      - BallMover on the XYZ of one particle that is not a rigid-body member,
      - RigidBodyMover (no nested bodies, no non-rigid members),
      - NormalMover on the ISD nuisance of one particle;
  * score states: _RigidBodyNormalizeConstraint, _RigidBodyPositionConstraint,
    NuisanceScoreState (float bounds only);
  * any restraints that have a JAX implementation (they are evaluated on a
    compact view of the model; see CompactArray);
  * return_best False, no OptimizerStates.

Differences from IMP's generic JAX MonteCarlo, same target distribution:
  1. Compact arrays: only particles that the restraints, movers and
     constraints actually use are kept (about 1k instead of all 39k Model
     particles). IMP particle indexes are remapped at trace time, so IMP's
     restraint _get_jax() functions work unchanged via CompactArray.
  2. No lax.switch/lax.cond in the step. SerialMover's deterministic mover
     order is used to apply every mover type as a masked update (the
     non-selected movers get a zero step into a sink row), and Metropolis
     acceptance is a jnp.where select.
  3. Constraints are applied incrementally: a rigid-body move updates only
     that body's members (rigid members' local coordinates are constant),
     and a nuisance move clamps only that nuisance.
  4. All random numbers for a frame come from one RNG call before the loop.
     (The random stream therefore differs from IMP's; the distribution
     matches.)
"""
import math
from dataclasses import dataclass

import numpy as np

_applied = False
UNROLL = 1          # fori_loop unroll factor for the MC steps of a frame


def _jax_ns():
    """IMP's JAX helpers (Space, get_random_key) live in IMP.jax since the
       2026-10-05 nightly and in IMP._jax_util before that."""
    try:
        import IMP.jax
        return IMP.jax
    except ImportError:
        import IMP._jax_util
        return IMP._jax_util


def apply():
    global _applied
    if _applied:
        return
    import IMP.core
    orig = IMP.core.MonteCarlo._get_jax_optimizer

    def _get_jax_optimizer(self, max_steps, space=None):
        try:
            return _FusedMCOptimizer(self, max_steps, space)
        except _Unsupported as e:
            print(f'imp_jax_fused: falling back to IMP JAX MC ({e})')
            return orig(self, max_steps, space)

    IMP.core.MonteCarlo._get_jax_optimizer = _get_jax_optimizer
    _applied = True


class _Unsupported(Exception):
    pass


# ---------------------------------------------------------------------------
# Compact particle arrays with IMP-index remapping
# ---------------------------------------------------------------------------
class _Remap:
    """Maps IMP ParticleIndex -> compact row. Hashed by identity, so it can
       be pytree aux data."""
    def __init__(self, table, nrows):
        self.table = table        # np.int32[n_model_particles]
        self.nrows = nrows

    def map_static(self, idx):
        try:
            a = np.asarray(idx, dtype=np.int64)
        except (TypeError, ValueError):
            a = np.asarray(int(idx), dtype=np.int64)
        return self.table[a]


def _register_compact_array():
    import jax
    import jax.numpy as jnp
    try:
        from jax.core import Tracer as _Tracer
    except ImportError:
        from jax._src.core import Tracer as _Tracer

    @jax.tree_util.register_pytree_node_class
    class CompactArray:
        """A particle-indexed array that stores only the used rows.
           Indexing with IMP particle indexes (static or traced) is remapped;
           unmapped particles read a NaN sentinel row."""
        def __init__(self, data, remap):
            self.data = data
            self.remap = remap

        def tree_flatten(self):
            return (self.data,), self.remap

        @classmethod
        def tree_unflatten(cls, remap, children):
            return cls(children[0], remap)

        def _map(self, idx):
            if isinstance(idx, _Tracer):
                return jnp.asarray(self.remap.table)[idx]
            try:
                return self.remap.map_static(idx)
            except (TypeError, jax.errors.TracerArrayConversionError,
                    jax.errors.ConcretizationTypeError):
                return jnp.asarray(self.remap.table)[idx]

        def __getitem__(self, idx):
            return self.data[self._map(idx)]

        @property
        def at(self):
            return _At(self)

        @property
        def shape(self):
            return self.data.shape

        @property
        def dtype(self):
            return self.data.dtype

    class _At:
        def __init__(self, arr):
            self.arr = arr

        def __getitem__(self, idx):
            return _AtIdx(self.arr, self.arr._map(idx))

    class _AtIdx:
        def __init__(self, arr, rows):
            self.arr, self.rows = arr, rows

        def _new(self, data):
            return CompactArray(data, self.arr.remap)

        def set(self, v):
            return self._new(self.arr.data.at[self.rows].set(v))

        def add(self, v):
            return self._new(self.arr.data.at[self.rows].add(v))

        def multiply(self, v):
            return self._new(self.arr.data.at[self.rows].multiply(v))

    return CompactArray


_CompactArray = None


def _compact_array_class():
    global _CompactArray
    if _CompactArray is None:
        _CompactArray = _register_compact_array()
    return _CompactArray


# ---------------------------------------------------------------------------
# State of the fused sampler
# ---------------------------------------------------------------------------
def _register_state():
    import jax

    @jax.tree_util.register_dataclass
    @dataclass
    class _FusedState:
        xyz: jax.Array            # (K+2, 3) compact coordinates
        nuis: jax.Array           # (K+2,)   compact nuisances
        quat: jax.Array           # (nb+1, 4) rigid body rotations
        score: jax.Array
        imov: jax.Array           # index of the last SerialMover sub-mover
        temperature: jax.Array
        rkey: jax.Array
        accepted_steps: jax.Array
        rejected_steps: jax.Array
        downward_steps_taken: jax.Array
        upward_steps_taken: jax.Array
        proposed_mover_steps: jax.Array   # (n_movers,)
        accepted_mover_steps: jax.Array   # (n_movers,)

        @property
        def jm(self):
            """JAX Model view (CompactArrays) of this state, for code that
               expects IMP's jm dict (e.g. PMI stat output)."""
            return _CURRENT_VIEW[0](self.xyz, self.nuis)

    return _FusedState


_CURRENT_VIEW = [None]


_FusedStateCls = None


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------
class _FusedMCOptimizer:
    def __init__(self, mc, max_steps, space):
        import jax
        import jax.numpy as jnp
        import IMP
        import IMP.core
        import IMP.isd
        from IMP.core import _jax_rigid
        global _FusedStateCls
        if _FusedStateCls is None:
            _FusedStateCls = _register_state()
        CompactArray = _compact_array_class()

        self.opt = mc
        m = self._m = mc.get_model()
        self._space = space or _jax_ns().FreeSpace
        if mc.get_return_best():
            raise _Unsupported('return_best')
        if len(list(mc.optimizer_states)):
            raise _Unsupported('OptimizerStates')
        movers = [mv.get_derived_object() for mv in mc.movers]
        if len(movers) != 1 or not isinstance(movers[0], IMP.core.SerialMover):
            raise _Unsupported('need exactly one SerialMover')
        self._movers = movers
        subs = [mv.get_derived_object() for mv in movers[0].get_movers()]
        nmov = len(subs)

        # --- rigid bodies ---------------------------------------------------
        rb_pi_objs = list(_jax_rigid._get_rigid_body_indexes(m)) \
            if m.get_has_data(_jax_rigid._RB_LIST_KEY) else []
        if any(True for _ in _jax_rigid._get_nested_rigid_body_indexes(
                m, rb_pi_objs)):
            raise _Unsupported('nested rigid bodies')
        rb_pis = [int(p) for p in rb_pi_objs]
        members = [np.asarray(IMP.core.RigidBody(m, IMP.ParticleIndex(p))
                              .get_member_particle_indexes(), dtype=np.int64)
                   for p in rb_pis]
        if rb_pis:
            is_rigid = m.get_numpy(_jax_rigid._RB_IS_RIGID_KEY)
            if np.any(is_rigid == 0):
                raise _Unsupported('non-rigid members')
        rb_member_set = set(int(x) for mi in members for x in mi)

        # --- score states ---------------------------------------------------
        nuis_bounds = {}
        for ss in m.get_ordered_score_states():
            ss = ss.get_derived_object()
            name = type(ss).__name__
            if name in ('_RigidBodyNormalizeConstraint',
                        '_RigidBodyPositionConstraint'):
                continue
            if isinstance(ss, IMP.isd.NuisanceScoreState):
                n = IMP.isd.Nuisance(m, ss.get_index())
                if n.get_has_upper_particle() or n.get_has_lower_particle():
                    raise _Unsupported('particle-valued nuisance bounds')
                lo = n.get_lower() if n.get_has_lower_float() else -math.inf
                up = n.get_upper() if n.get_has_upper_float() else math.inf
                nuis_bounds[int(ss.get_index())] = (lo, up)
                continue
            raise _Unsupported(f'score state {name}')

        # --- movers ---------------------------------------------------------
        xyz_keys = frozenset(IMP.core.XYZ.get_xyz_keys())
        nuis_key = IMP.isd.Scale.get_scale_key()
        kinds = []
        for sub in subs:
            if isinstance(sub, IMP.core.BallMover):
                idx = [int(i) for i in sub.get_indexes()]
                if frozenset(sub.get_keys()) != xyz_keys or len(idx) != 1 \
                        or idx[0] in rb_member_set:
                    raise _Unsupported('BallMover variant')
                kinds.append(('ball', idx[0], sub.get_radius()))
            elif isinstance(sub, IMP.core.RigidBodyMover):
                kinds.append(('rb', rb_pis.index(int(sub.get_index())),
                              sub.get_maximum_translation(),
                              sub.get_maximum_rotation()))
            elif isinstance(sub, IMP.core.NormalMover):
                idx = [int(i) for i in sub.get_indexes()]
                keys = list(sub.get_keys())
                if len(idx) != 1 or len(keys) != 1 or keys[0] != nuis_key:
                    raise _Unsupported('NormalMover variant')
                kinds.append(('normal', idx[0], sub.get_sigma()))
            else:
                raise _Unsupported(f'mover {type(sub).__name__}')

        # --- restraints -----------------------------------------------------
        sf = mc.get_scoring_function().get_derived_object()
        rji = sf._get_jax(space=self._space)
        rscore = rji.score_func
        rsingle = sf.get_single_restraint()
        used = set(int(p.get_index()) for p in IMP.get_input_particles(
            rsingle.get_inputs()))
        extra_keys = [k for k in rji._keys if k != nuis_key]

        # --- compact index space ---------------------------------------------
        used |= set(rb_pis) | rb_member_set | set(nuis_bounds)
        used |= set(k[1] for k in kinds if k[0] in ('ball', 'normal'))
        rows = np.array(sorted(used), dtype=np.int64)
        K = len(rows)
        SINK, NAN_ROW = K, K + 1
        nmodel = m.get_spheres_numpy()[0].shape[0]
        nattr = max(nmodel, len(m.get_numpy(nuis_key)))
        table = np.full(nattr, NAN_ROW, dtype=np.int32)
        table[rows] = np.arange(K, dtype=np.int32)
        remap = _Remap(table, K + 2)
        self._rows = rows
        self._remap = remap
        # ISD nuisance particles (bounded or moved), for per-frame recording
        self._nuis_pis = sorted(set(nuis_bounds) |
                                set(k[1] for k in kinds if k[0] == 'normal'))
        self._nuis_rows = table[np.asarray(self._nuis_pis, dtype=np.int64)] \
            if self._nuis_pis else np.zeros(0, np.int32)
        self._nuis_names = [m.get_particle_name(IMP.ParticleIndex(pi))
                            for pi in self._nuis_pis]

        def compact(full, fill=np.nan):
            # IMP attribute arrays can have different lengths (e.g. xyz stops
            # before the nuisance particles), so mask rows past the end
            full = np.asarray(full)
            out = np.full((K + 2,) + full.shape[1:], fill, dtype=np.float32)
            ok = rows < len(full)
            out[:K][ok] = full[rows[ok]]
            return out

        xyz_full, r_full = m.get_spheres_numpy()
        radius = compact(r_full)
        nuis_full = m.get_numpy(nuis_key)
        lower = np.full(K + 2, -np.inf, np.float32)
        upper = np.full(K + 2, np.inf, np.float32)
        for pi, (lo, up) in nuis_bounds.items():
            lower[table[pi]], upper[table[pi]] = lo, up
        extra = {k.get_string(): compact(m.get_numpy(k)) for k in extra_keys}

        nb = len(rb_pis)
        mmax = max([len(mi) for mi in members] + [1])
        members_pad = np.full((nb + 1, mmax), SINK, np.int32)
        local_pad = np.zeros((nb + 1, mmax, 3), np.float32)
        intcoord = m.get_internal_coordinates_numpy() if nb else None
        for b, mi in enumerate(members):
            members_pad[b, :len(mi)] = table[mi]
            local_pad[b, :len(mi)] = intcoord[mi]
        rb_row = np.full(nb + 1, SINK, np.int32)
        rb_row[:nb] = table[np.asarray(rb_pis, dtype=np.int64)]
        # dense per-row tables: owning body of each rigid member (nb + 1 for
        # rows that are not rigid members; body nb is the no-op dummy) and
        # its constant local coordinates
        owner = np.full(K + 2, nb + 1, np.int32)
        local_all = np.zeros((K + 2, 3), np.float32)
        for b, mi in enumerate(members):
            owner[table[mi]] = b
            local_all[table[mi]] = intcoord[mi]

        # per-mover parameter tables (non-selected mover types are no-ops)
        ball_row = np.full(nmov, SINK, np.int32)
        ball_rad = np.zeros(nmov, np.float32)
        nrm_row = np.full(nmov, SINK, np.int32)
        nrm_sig = np.zeros(nmov, np.float32)
        rb_body = np.full(nmov, nb, np.int32)
        rb_tr = np.zeros(nmov, np.float32)
        rb_rot = np.zeros(nmov, np.float32)
        for i, k in enumerate(kinds):
            if k[0] == 'ball':
                ball_row[i], ball_rad[i] = table[k[1]], k[2]
            elif k[0] == 'normal':
                nrm_row[i], nrm_sig[i] = table[k[1]], k[2]
            else:
                rb_body[i], rb_tr[i], rb_rot[i] = k[1], k[2], k[3]

        # constants on the default device
        c = {n: jnp.asarray(v) for n, v in dict(
            radius=radius, lower=lower, upper=upper, members_pad=members_pad,
            local_pad=local_pad, rb_row=rb_row, ball_row=ball_row,
            owner=owner, local_all=local_all,
            rows=np.arange(K + 2, dtype=np.int32),
            movers=np.arange(nmov, dtype=np.int32),
            bodies=np.arange(nb + 1, dtype=np.int32),
            ball_rad=ball_rad, nrm_row=nrm_row, nrm_sig=nrm_sig,
            rb_body=rb_body, rb_tr=rb_tr, rb_rot=rb_rot).items()}
        c_extra = {n: jnp.asarray(v) for n, v in extra.items()}

        def view(xyz, nuis):
            """JAX Model dict for IMP's restraint functions"""
            jm = {'xyz': CompactArray(xyz, remap),
                  'r': CompactArray(c['radius'], remap),
                  'nuisance': CompactArray(nuis, remap)}
            for n, v in c_extra.items():
                jm[n] = CompactArray(v, remap)
            return jm

        def score_of(xyz, nuis):
            return rscore(view(xyz, nuis))

        def q2m(q):
            v0, v1, v2, v3 = q[0], q[1], q[2], q[3]
            return jnp.array(
                [[v0*v0 + v1*v1 - v2*v2 - v3*v3, 2.*(v1*v2 + v0*v3),
                  2.*(v1*v3 - v0*v2)],
                 [2.*(v1*v2 - v0*v3), v0*v0 - v1*v1 + v2*v2 - v3*v3,
                  2.*(v2*v3 + v0*v1)],
                 [2.*(v1*v3 + v0*v2), 2.*(v2*v3 - v0*v1),
                  v0*v0 - v1*v1 - v2*v2 + v3*v3]])

        def qmul(q1, q2):
            return jnp.array(
                [q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                 q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                 q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                 q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]])

        def in_sphere(u, radius):
            # same construction as IMP.algebra get_random_vector_in_3d_sphere
            phi = 2. * math.pi * u[0]
            theta = jnp.arccos(2. * u[1] - 1.)
            r = radius * jnp.cbrt(u[2])
            return jnp.array([r * jnp.sin(theta) * jnp.cos(phi),
                              r * jnp.sin(theta) * jnp.sin(phi),
                              r * jnp.cos(theta)])

        def step(s, u, z):
            """One MC step. u: 10 uniforms, z: 1 standard normal.
               All updates are dense masked selects over the ~1k compact
               rows (no scatter/switch/cond), so XLA can fuse them."""
            imov = (s.imov + 1) % nmov
            xyz, nuis, quat = s.xyz, s.nuis, s.quat
            rows = c['rows']
            # BallMover (zero radius / sink row if another mover type)
            v = in_sphere(u[0:3], c['ball_rad'][imov])
            xyz_p = xyz + (rows == c['ball_row'][imov])[:, None] * v
            # NormalMover on a nuisance, then its NuisanceScoreState clamp
            nr = c['nrm_row'][imov]
            nval = jnp.clip(nuis[nr] + z * c['nrm_sig'][imov],
                            c['lower'][nr], c['upper'][nr])
            nuis_p = jnp.where(rows == nr, nval, nuis)
            # RigidBodyMover, then normalize + update only this body's members
            b = c['rb_body'][imov]
            rr = c['rb_row'][b]
            t_new = xyz[rr] + in_sphere(u[3:6], c['rb_tr'][imov])
            phi = 2. * math.pi * u[6]
            theta = jnp.arccos(2. * u[7] - 1.)
            axis = jnp.array([jnp.sin(theta) * jnp.cos(phi),
                              jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])
            ang = c['rb_rot'][imov] * (2. * u[8] - 1.)
            qrot = jnp.concatenate([jnp.cos(ang / 2.)[None],
                                    axis * jnp.sin(ang / 2.)])
            q_new = qmul(qrot, quat[b])
            qn = jnp.linalg.norm(q_new)
            q_new = jnp.where(qn < 1e-3, jnp.array([1., 0., 0., 0.]),
                              q_new / qn)
            quat_p = jnp.where((c['bodies'] == b)[:, None], q_new, quat)
            R = q2m(q_new)
            la = c['local_all']
            # v @ R written out, so it fuses instead of becoming a GEMM
            glob = (la[:, 0:1] * R[0] + la[:, 1:2] * R[1]
                    + la[:, 2:3] * R[2] + t_new)
            xyz_p = jnp.where((c['owner'] == b)[:, None], glob, xyz_p)
            xyz_p = jnp.where((rows == rr)[:, None], t_new, xyz_p)
            # score + Metropolis (branch-free)
            new = score_of(xyz_p, nuis_p)
            down = new < s.score
            acc = down | (u[9] < jnp.exp(-(new - s.score) / s.temperature))
            sel = lambda a, o: jnp.where(acc, a, o)  # noqa: E731
            return _FusedStateCls(
                xyz=sel(xyz_p, xyz), nuis=sel(nuis_p, nuis),
                quat=sel(quat_p, quat), score=sel(new, s.score), imov=imov,
                temperature=s.temperature, rkey=s.rkey,
                accepted_steps=s.accepted_steps + acc,
                rejected_steps=s.rejected_steps + (~acc),
                downward_steps_taken=s.downward_steps_taken + down,
                upward_steps_taken=s.upward_steps_taken + (acc & ~down),
                proposed_mover_steps=s.proposed_mover_steps
                + (c['movers'] == imov),
                accepted_mover_steps=s.accepted_mover_steps
                + ((c['movers'] == imov) & acc))

        nsteps = max_steps

        def frame(s):
            key, k1, k2 = jax.random.split(s.rkey, 3)
            U = jax.random.uniform(k1, (nsteps, 10))
            Z = jax.random.normal(k2, (nsteps,))
            s.rkey = key
            return jax.lax.fori_loop(0, nsteps,
                                     lambda i, s: step(s, U[i], Z[i]), s,
                                     unroll=UNROLL)

        self.inner_steps = nsteps
        self.n_loops = 1
        self._imp_opt_states = []
        self._frame_py = frame
        self.apply_func = jax.jit(frame)
        self._score_of = jax.jit(score_of)
        self._x0 = dict(xyz=compact(xyz_full), nuis=compact(nuis_full),
                        quat=None)
        if nb:
            self._x0['quat'] = np.concatenate(
                [m.get_numpy(_jax_rigid._RB_QUAT_KEY)[np.asarray(rb_pis)],
                 [[1., 0., 0., 0.]]]).astype(np.float32)
        else:
            self._x0['quat'] = np.array([[1., 0., 0., 0.]], np.float32)
        self._nmov, self._rb_pis, self._nb = nmov, rb_pis, nb
        self._jm_view = view
        _CURRENT_VIEW[0] = view

    # -- interface used by IMP.pmi.samplers.MonteCarlo ------------------------
    def get_initial_state(self, key=None):
        import jax.numpy as jnp
        import IMP.isd
        m = self._m
        x = self._x0
        K = len(self._rows)
        xyz = x['xyz'].copy()
        xyz_full = m.get_spheres_numpy()[0]
        ok = self._rows < len(xyz_full)
        xyz[:K][ok] = xyz_full[self._rows[ok]]
        nuis = x['nuis'].copy()
        nuis_full = m.get_numpy(IMP.isd.Scale.get_scale_key())
        ok = self._rows < len(nuis_full)
        nuis[:K][ok] = nuis_full[self._rows[ok]]
        xyz, nuis, quat = (jnp.asarray(xyz), jnp.asarray(nuis),
                           jnp.asarray(x['quat']))
        z = jnp.int32(0)
        st = _FusedStateCls(
            xyz=xyz, nuis=nuis, quat=quat,
            score=self._score_of(xyz, nuis), imov=jnp.int32(-1),
            temperature=jnp.float32(self.opt.get_kt()),
            rkey=key if key is not None else _jax_ns().get_random_key(),
            accepted_steps=z, rejected_steps=z, downward_steps_taken=z,
            upward_steps_taken=z,
            proposed_mover_steps=jnp.zeros(self._nmov, jnp.int32),
            accepted_mover_steps=jnp.zeros(self._nmov, jnp.int32))
        self._prev_stats = None
        return st

    def _write_model(self, xyz, nuis, quat):
        """Copy one chain's compact coordinates, nuisances and rigid-body
           rotations (host arrays) into the IMP Model."""
        from IMP.isd import Scale
        from IMP.core import _jax_rigid
        m = self._m
        K = len(self._rows)
        xyz_full = m.get_spheres_numpy()[0]
        ok = self._rows < len(xyz_full)
        xyz_full[self._rows[ok]] = xyz[:K][ok]
        nuis_full = m.get_numpy(Scale.get_scale_key())
        ok = self._rows < len(nuis_full)
        nuis_full[self._rows[ok]] = nuis[:K][ok]
        if self._nb:
            m.get_numpy(_jax_rigid._RB_QUAT_KEY)[
                np.asarray(self._rb_pis)] = quat[:self._nb]

    def _apply_stats(self, score, d_down, d_up, d_rej, d_prop_i, d_acc_i):
        """Report one frame's MC statistics (deltas) to IMP's MonteCarlo,
           SerialMover and sub-movers, as C++ MonteCarlo would."""
        self.opt.set_number_of_downward_steps(int(d_down))
        self.opt.set_number_of_upward_steps(int(d_up))
        self.opt.set_number_of_rejected_steps(int(d_rej))
        self.opt.set_best_accepted_energy(float(score))
        self.opt.set_last_accepted_energy(float(score))
        serial = self._movers[0]
        serial.add_to_statistics(int(d_down + d_up + d_rej), int(d_rej))
        for j, sub in enumerate(serial.get_movers()):
            sub.add_to_statistics(int(d_prop_i[j]),
                                  int(d_prop_i[j] - d_acc_i[j]))

    def optimize(self, st):
        import jax
        import jax.numpy as jnp
        # PMI's set_kt() may store a Python float; keep the dtype stable so
        # the jitted frame is not retraced
        st.temperature = jnp.float32(st.temperature)
        st = self.apply_func(st)
        host = jax.device_get((st.xyz, st.nuis, st.quat, st.score,
                               st.accepted_steps, st.rejected_steps,
                               st.downward_steps_taken,
                               st.upward_steps_taken,
                               st.proposed_mover_steps,
                               st.accepted_mover_steps))
        xyz, nuis, quat, score, acc, rej, down, up, prop_i, acc_i = host
        self._write_model(xyz, nuis, quat)
        prev = self._prev_stats or (0, 0, 0, 0, np.zeros_like(prop_i),
                                    np.zeros_like(acc_i))
        p_acc, p_rej, p_down, p_up, p_prop, p_acci = prev
        self._prev_stats = (acc, rej, down, up, prop_i, acc_i)
        self._apply_stats(score, down - p_down, up - p_up, rej - p_rej,
                          prop_i - p_prop, acc_i - p_acci)
        self._last_host_score = float(score)
        return self._last_host_score, st
