"""From an IMP model to flat tables: what moves, how, and what is scored.

extract(source) walks the movers first (to learn which particles move and
how), then the restraint tree. Each restraint goes to the term class
registered for it (jax_mc.terms). A restraint whose inputs never move
becomes a constant. Anything else raises Unsupported.

Layout (all float64 numpy on the host):

* rows: every particle a term reads. Each row is one of
    STATIC  fixed global coordinates
    MEMBER  a member of a moving rigid body (local coordinates); the body's
            own particle is its first member, at local (0, 0, 0)
            (row_is_center marks it)
    BEAD    a particle moved by a BallMover or a TransformMover (global
            coordinates are degrees of freedom); a NonRigidMember of a
            static body is allowed
  Rows are ordered static first, then body by body (center, members), then
  beads in mover order, so that each mover's rows are contiguous and
  positions() is a plain concatenation.
* nuisances: every Scale particle a term reads (value, bounds).
* degrees of freedom, chain-last: quat (4, nb, C), trans (3, nb, C),
  bead (3, nbead, C), nuis (N, C). positions() gives X (3, rows, C).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import imp_hooks

STATIC, MEMBER, BEAD = 0, 2, 3


class Unsupported(Exception):
    pass


@dataclass
class Mover:
    kind: str                   # 'rb', 'ball', 'transform', 'normal'
    imp: object
    bodies: list = field(default_factory=list)    # body indexes
    beads: list = field(default_factory=list)     # bead indexes
    nuis: int = -1                                 # nuisance index
    max_trans: float = 0.
    max_rot: float = 0.
    radius: float = 0.
    sigma: float = 0.
    frame: object = None        # ball in a static parent frame (3x3) or None
    rows: list = field(default_factory=list)       # rows it moves (finalize)


def quat_to_rot(q, xp=np):
    """(4, ...) unit quaternions (w, x, y, z) -> (3, 3, ...) rotations, in
       the form IMP::algebra::Rotation3D uses (w*w + x*x - y*y - z*z, ...),
       so member positions match IMP's exactly. (Gaussian covariances use
       Eigen's form instead, as core::Gaussian does: terms.geom.)"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.stack([
        xp.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z),
                  2 * (x * z + w * y)]),
        xp.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z,
                  2 * (y * z - w * x)]),
        xp.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                  w * w - x * x - y * y + z * z])])


class System:
    """Tables for one IMP system; build with extract()"""

    def __init__(self, source, cov_mode='frozen'):
        import IMP
        self.source = source
        self.name = source.name
        self.m = source.model
        self.mode = source.mode
        self.kT = source.kT
        self.cov_mode = cov_mode
        self._pi = IMP.ParticleIndex
        self._row_of = {}           # particle index -> row (before layout)
        self._nuis_of = {}
        self.terms = []
        self.constant = 0.
        self.constant_restraints = []
        self.skipped = []           # (reason, restraint)

    # ---------------- registration (used by extractors) ----------------
    def row(self, p):
        p = int(p)
        if p not in self._row_of:
            self._row_of[p] = len(self._row_of)
        return p

    def nuisance(self, p):
        p = int(p)
        if p not in self._nuis_of:
            self._nuis_of[p] = len(self._nuis_of)
        return p

    def moves(self, p):
        """Does particle p (or its nuisance value) ever move?"""
        p = int(p)
        return (p in self.body_index or p in self.bead_index
                or p in self.moved_nuis or p in self._member_body)

    # ---------------- movers -------------------------------------------
    def _movers(self):
        import IMP
        import IMP.core
        import IMP.isd
        try:
            import IMP.pmi
            transform = IMP.pmi.TransformMover
        except (ImportError, AttributeError):
            transform = ()
        m, pi_ = self.m, self._pi
        flat = []

        def walk(mv):
            d = mv.get_derived_object() if hasattr(
                mv, 'get_derived_object') else mv
            if isinstance(d, IMP.core.SerialMover):
                for x in d.get_movers():
                    walk(x)
            else:
                flat.append(d)
        for mv in self.source.movers:
            walk(mv)
        bodies, beads, nuis = [], [], []
        movers = []
        xyz_keys = set(k.get_string() for k in IMP.core.XYZ.get_xyz_keys())
        scale_key = IMP.isd.Scale.get_scale_key()

        def add(lst, p):
            if p not in lst:
                lst.append(p)
            return lst.index(p)

        for d in flat:
            if isinstance(d, IMP.core.RigidBodyMover):
                b = int(d.get_index())
                movers.append(Mover('rb', d, bodies=[add(bodies, b)],
                                    max_trans=d.get_maximum_translation(),
                                    max_rot=d.get_maximum_rotation()))
            elif transform and isinstance(d, transform):
                bs, xs = [], []
                for o in d.get_inputs():
                    p = int(IMP.Particle.get_from(o).get_index())
                    if IMP.core.RigidBody.get_is_setup(m, pi_(p)):
                        bs.append(add(bodies, p))
                    else:
                        xs.append(add(beads, p))
                movers.append(Mover('transform', d, bodies=bs, beads=xs,
                                    max_trans=d.get_maximum_translation(),
                                    max_rot=d.get_maximum_rotation()))
            elif isinstance(d, IMP.core.BallMover):
                idx = [int(x) for x in d.get_indexes()]
                keys = set(k.get_string() for k in d.get_keys())
                frame = None
                if len(idx) == 1 and keys == {'local_x', 'local_y',
                                              'local_z'}:
                    # internal coordinates of a non-rigid member: the step
                    # is a ball in the parent body's frame (checked to be
                    # static below)
                    pp = pi_(idx[0])
                    if not IMP.core.NonRigidMember.get_is_setup(m, pp):
                        raise Unsupported('BallMover on local coordinates '
                                          'of a non-member')
                    rot = IMP.core.NonRigidMember(m, pp).get_rigid_body() \
                        .get_reference_frame().get_transformation_to() \
                        .get_rotation()
                    frame = np.array([rot.get_rotation_matrix_row(i)
                                      for i in range(3)])
                elif len(idx) != 1 or keys != xyz_keys:
                    raise Unsupported('BallMover on several particles or '
                                      'not on x, y, z')
                movers.append(Mover('ball', d, beads=[add(beads, idx[0])],
                                    radius=d.get_radius(), frame=frame))
            elif isinstance(d, IMP.core.NormalMover):
                idx = [int(x) for x in d.get_indexes()]
                keys = list(d.get_keys())
                if len(idx) != 1 or len(keys) != 1 or keys[0] != scale_key:
                    raise Unsupported('NormalMover not on one Scale')
                movers.append(Mover('normal', d, nuis=add(nuis, idx[0]),
                                    sigma=d.get_sigma()))
            else:
                raise Unsupported(f'mover {type(d).__name__}')
        self.movers = movers
        self.body_pi = np.array(bodies, np.int64)
        self.bead_pi = np.array(beads, np.int64)
        self.body_index = {p: i for i, p in enumerate(bodies)}
        self.bead_index = {p: i for i, p in enumerate(beads)}
        self.moved_nuis = set(nuis)
        self._moved_nuis_order = nuis

        # rigid bodies: frames, members
        member_body = {}
        for b in bodies:
            if IMP.core.RigidMember.get_is_setup(m, pi_(b)):
                raise Unsupported('mover on a nested rigid body')
            rb = IMP.core.RigidBody(m, pi_(b))
            for x in rb.get_member_particle_indexes():
                member_body[int(x)] = b
            for x in rb.get_body_member_particle_indexes():
                if IMP.core.RigidBody(m, x).get_number_of_members():
                    raise Unsupported('nested body with members')
                member_body[int(x)] = b
        for p in beads:
            if p in member_body:
                pp = pi_(p)
                if IMP.core.RigidMember.get_is_setup(m, pp):
                    raise Unsupported('BallMover on a rigid member')
                raise Unsupported('bead of a moving body')
            if IMP.core.RigidMember.get_is_setup(m, pi_(p)):
                raise Unsupported('BallMover on a rigid member')
        self._member_body = member_body
        q, t = [], []
        for b in bodies:
            tr = IMP.core.RigidBody(m, pi_(b)).get_reference_frame() \
                .get_transformation_to()
            q.append(tr.get_rotation().get_quaternion())
            t.append(tr.get_translation())
        self.quat0 = np.array(q, np.float64).reshape(-1, 4)
        self.trans0 = np.array(t, np.float64).reshape(-1, 3)
        self.bead0 = np.array([IMP.core.XYZ(m, pi_(int(p))).get_coordinates()
                               for p in beads], np.float64).reshape(-1, 3)

    # ---------------- score states -------------------------------------
    def _score_states(self):
        import IMP.isd
        m = self.m
        bounds = {}
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
                bounds[int(d.get_index())] = (lo, up)
                continue
            if ss.get_is_ignored_by_jax():
                continue
            raise Unsupported(f'score state {name} {ss.get_name()}')
        self.nuis_bounds = bounds

    # ---------------- layout -------------------------------------------
    def _layout(self):
        """Final row order, row tables, nuisance tables, mover rows"""
        import IMP
        import IMP.core
        import IMP.isd
        m, pi_ = self.m, self._pi
        for b in self.body_pi:
            self.row(b)
        for p in self.bead_pi:
            self.row(p)
        used = list(self._row_of)

        def key(p):
            if p in self.body_index:
                return (1, self.body_index[p], 0, p)
            if p in self._member_body:
                return (1, self.body_index[self._member_body[p]], 1, p)
            if p in self.bead_index:
                return (2, self.bead_index[p], 0, p)
            return (0, 0, 0, p)
        rows = sorted(used, key=key)
        self.row_pi = np.array(rows, np.int64)
        self.row_of = {p: i for i, p in enumerate(rows)}
        n = len(rows)
        kind = np.zeros(n, np.int64)
        ref = np.zeros(n, np.int64)
        local = np.zeros((n, 3))
        xyz0 = np.zeros((n, 3))
        radius = np.zeros(n)
        is_center = np.zeros(n, bool)
        for i, p in enumerate(rows):
            pp = pi_(p)
            if IMP.core.XYZ.get_is_setup(m, pp):
                xyz0[i] = IMP.core.XYZ(m, pp).get_coordinates()
            if IMP.core.XYZR.get_is_setup(m, pp):
                radius[i] = IMP.core.XYZR(m, pp).get_radius()
            if p in self.body_index:
                kind[i], ref[i] = MEMBER, self.body_index[p]
                is_center[i] = True
            elif p in self._member_body:
                kind[i] = MEMBER
                ref[i] = self.body_index[self._member_body[p]]
                if IMP.core.RigidMember.get_is_setup(m, pp):
                    local[i] = IMP.core.RigidMember(m, pp) \
                        .get_internal_coordinates()
                else:
                    local[i] = IMP.core.NonRigidMember(m, pp) \
                        .get_internal_coordinates()
            elif p in self.bead_index:
                kind[i], ref[i] = BEAD, self.bead_index[p]
        self.row_kind, self.row_ref = kind, ref
        self.row_is_center = is_center
        # the concatenation in positions() relies on this order
        assert np.all(np.diff(kind[kind != STATIC]) >= 0)
        assert np.all(kind[:np.sum(kind == STATIC)] == STATIC)
        bd = ref[kind == BEAD]
        assert np.array_equal(bd, np.arange(len(bd)))
        assert np.all(np.diff(ref[kind == MEMBER]) >= 0)
        self.row_local, self.row_xyz0, self.row_radius = local, xyz0, radius
        self.n_rows = n
        # nuisances: moved ones first (mover order), then the rest
        for p in self._moved_nuis_order:
            self.nuisance(p)
        nus = sorted(self._nuis_of, key=lambda p: (
            p not in self.moved_nuis,
            self._moved_nuis_order.index(p) if p in self.moved_nuis else 0,
            p))
        self.nuis_pi = np.array(nus, np.int64)
        self.nuis_of = {p: i for i, p in enumerate(nus)}
        self.nuis0 = np.array([IMP.isd.Scale(m, pi_(p)).get_scale()
                               for p in nus], np.float64)
        self.nuis_lower = np.array([self.nuis_bounds.get(p, (-math.inf,))[0]
                                    for p in nus])
        self.nuis_upper = np.array([self.nuis_bounds.get(
            p, (None, math.inf))[1] for p in nus])
        self.nb, self.nbead = len(self.body_pi), len(self.bead_pi)
        self.n_nuis = len(nus)
        # rows each mover moves
        rows_of_body = {}
        for i in range(n):
            if kind[i] == MEMBER:
                rows_of_body.setdefault(int(ref[i]), []).append(i)
        for mv in self.movers:
            r = []
            for b in mv.bodies:
                r += rows_of_body.get(b, [])
            r += [self.row_of[int(self.bead_pi[k])] for k in mv.beads]
            mv.rows = sorted(r)
        self.static_rows = np.nonzero(kind == STATIC)[0]

    def rows(self, pis):
        return np.array([self.row_of[int(p)] for p in pis], np.int64)

    def nuis_rows(self, pis):
        return np.array([self.nuis_of[int(p)] for p in pis], np.int64)

    # ---------------- degrees of freedom -------------------------------
    def initial_dofs(self, C=1):
        def rep(a):
            a = np.asarray(a, np.float64).T[..., None]
            return np.repeat(a, C, axis=-1)
        return dict(quat=rep(self.quat0), trans=rep(self.trans0),
                    bead=rep(self.bead0),
                    nuis=np.repeat(self.nuis0[:, None], C, axis=1))

    def positions(self, dofs, xp=np):
        """X (3, rows, C) from chain-last degrees of freedom: static rows,
           then members (R_b local + t_b), then beads, concatenated"""
        quat, trans, bead = dofs['quat'], dofs['trans'], dofs['bead']
        C = dofs['nuis'].shape[-1] if self.n_nuis else \
            (quat.shape[-1] if self.nb else bead.shape[-1])
        kind = self.row_kind
        n_st = int(np.sum(kind == STATIC))
        parts = [xp.broadcast_to(xp.asarray(self.row_xyz0[:n_st].T)[..., None],
                                 (3, n_st, C))]
        mem = np.nonzero(kind == MEMBER)[0]
        if len(mem):
            R = quat_to_rot(quat, xp)[:, :, self.row_ref[mem]]  # (3,3,n,C)
            loc = xp.asarray(self.row_local[mem].T)[..., None]  # (3, n, 1)
            parts.append(R[:, 0] * loc[0] + R[:, 1] * loc[1]
                         + R[:, 2] * loc[2] + trans[:, self.row_ref[mem]])
        if self.nbead:
            parts.append(bead)
        return xp.concatenate(parts, axis=1)

    def get_imp_dofs(self):
        """Current IMP model state as C=1 chain-last dofs"""
        import IMP.core
        import IMP.isd
        m, pi_ = self.m, self._pi
        q, t = [], []
        for b in self.body_pi:
            tr = IMP.core.RigidBody(m, pi_(int(b))).get_reference_frame() \
                .get_transformation_to()
            q.append(tr.get_rotation().get_quaternion())
            t.append(tr.get_translation())
        bead = [IMP.core.XYZ(m, pi_(int(p))).get_coordinates()
                for p in self.bead_pi]
        nuis = [IMP.isd.Scale(m, pi_(int(p))).get_scale()
                for p in self.nuis_pi]
        f = lambda a, k: np.asarray(a, np.float64).reshape(-1, k).T[  # noqa
            ..., None]
        return dict(quat=f(q, 4), trans=f(t, 3), bead=f(bead, 3),
                    nuis=np.asarray(nuis, np.float64).reshape(-1, 1))

    def set_imp_dofs(self, dofs, c=0):
        """Write chain c of chain-last dofs into the IMP model"""
        import IMP
        import IMP.algebra
        import IMP.core
        import IMP.isd
        m, pi_ = self.m, self._pi
        V = IMP.algebra.Vector3D
        for i, b in enumerate(self.body_pi):
            qq = [float(x) for x in dofs['quat'][:, i, c]]
            tt = [float(x) for x in dofs['trans'][:, i, c]]
            IMP.core.RigidBody(m, pi_(int(b))).set_reference_frame(
                IMP.algebra.ReferenceFrame3D(IMP.algebra.Transformation3D(
                    IMP.algebra.Rotation3D(*qq), V(*tt))))
        for i, p in enumerate(self.bead_pi):
            x = V(*[float(v) for v in dofs['bead'][:, i, c]])
            pp = pi_(int(p))
            if IMP.core.NonRigidMember.get_is_setup(m, pp):
                nr = IMP.core.NonRigidMember(m, pp)
                fr = nr.get_rigid_body().get_reference_frame()
                nr.set_internal_coordinates(fr.get_local_coordinates(x))
            else:
                IMP.core.XYZ(m, pp).set_coordinates(x)
        for i, p in enumerate(self.nuis_pi):
            IMP.isd.Scale(m, pi_(int(p))).set_scale(
                float(dofs['nuis'][i, c]))
        m.update()

    def random_dofs(self, C, seed=0, scale=3.0):
        """C states near the initial one: bodies rotated and translated,
           beads displaced, moved nuisances changed, each by up to `scale`
           times their mover's step (for checks)"""
        rng = np.random.default_rng(seed)
        d = self.initial_dofs(C)
        for mv in self.movers:
            for b in mv.bodies:
                for c in range(1, C):
                    ax = rng.normal(size=3)
                    ax /= np.linalg.norm(ax)
                    ang = scale * mv.max_rot * rng.uniform(-1, 1)
                    qr = np.concatenate([[math.cos(ang / 2)],
                                         ax * math.sin(ang / 2)])
                    d['quat'][:, b, c] = _qmul(qr, d['quat'][:, b, c])
                    d['trans'][:, b, c] += scale * mv.max_trans * \
                        rng.uniform(-1, 1, 3)
            for k in mv.beads:
                step = mv.radius if mv.kind == 'ball' else mv.max_trans
                d['bead'][:, k, 1:] += scale * step * rng.uniform(
                    -1, 1, (3, C - 1))
            if mv.kind == 'normal':
                i = self.nuis_of[int(self._moved_nuis_order[mv.nuis])]
                lo, up = self.nuis_lower[i], self.nuis_upper[i]
                v = d['nuis'][i, 1:] + scale * mv.sigma * rng.normal(
                    size=C - 1)
                d['nuis'][i, 1:] = np.clip(v, lo, up)
        return d


def _qmul(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


# ---------------------------------------------------------------------------
def extract(source, cov_mode='frozen', imp_jax_fallback=True):
    """System for a jax_mc.sources.Source"""
    from . import terms as T
    s = System(source, cov_mode)
    s._movers()
    s._score_states()
    builders = {}           # term class -> instance (one per class/key)

    def walk(r, w):
        import IMP
        d = imp_hooks.derived(r)
        if w * d.get_weight() == 0.:
            s.skipped.append(('weight 0', d))
            return
        # terms first: some restraint sets are terms (isd.LogWrapper)
        for cls in T.REGISTRY:
            key = cls.claims(d, s)
            if key is not None:
                k = (cls, key)
                if k not in builders:
                    builders[k] = cls(key)
                builders[k].add(s, w, d)
                return
        if isinstance(d, IMP.RestraintSet):
            ww = w * d.get_weight()
            for i in range(d.get_number_of_restraints()):
                walk(d.get_restraint(i), ww)
            return
        ins = imp_hooks.input_particles(d)
        if not any(s.moves(p) for p in ins):
            s.constant += w * d.evaluate(False)
            s.constant_restraints.append((w, d))
            return
        if imp_jax_fallback and T.ImpJax.claims_fallback(d, s):
            k = (T.ImpJax, 'fallback')
            if k not in builders:
                builders[k] = T.ImpJax('fallback')
            builders[k].add(s, w, d)
            return
        raise Unsupported(f'restraint {type(d).__name__} '
                          f'"{d.get_name()}"')

    for r in source.restraints:
        walk(r, 1.0)
    s._layout()
    s.terms = [b for b in builders.values()]
    for t in s.terms:
        t.finalize(s)
    return s
