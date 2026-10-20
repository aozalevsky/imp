"""JAX (or numpy) scoring for the imp_tutorial RNA Pol II stalk system, as a
function of the sampled degrees of freedom:

* quat (2, 4), trans (2, 3): frames of the Rpb4 and Rpb7 rigid bodies (the
  only bodies with a RigidBodyMover);
* bead (47, 3): global positions of the flexible beads. All of them are
  nonrigid members of rigid bodies that never move, so their BallMovers
  (which move internal coordinates) are, in distribution, isotropic moves
  of the global position.

Everything else is static. Terms (C++ -> here):
* 12 PMI ConnectivityRestraints (64 core.PairRestraints,
  SphereDistancePairScore(HarmonicUpperBound(optdist, 10)));
* PMI ExcludedVolumeSphere: container.PairsRestraint(SoftSpherePairScore(1))
  over a ClosePairContainer with RigidClosePairsFinder: all pairs except
  those of two rigid members of one body (zero for non-overlapping pairs,
  so dense = exact);
* 2 PMI CrossLinkingMassSpectrometryRestraints: isd.LogWrapper of 153 / 108
  isd.CrossLinkMSRestraints (sigma, psi fixed: no nuisance movers), score
  -sum log p; their Linear PairRestraints (slope 0) and nuisance priors are
  constant and kept as an offset. The forward model is the one IMP
  currently computes (salilab/imp#1120); if salilab/imp#1123 is merged,
  xl_neglogp() needs the same change;
* PMI GaussianEMRestraint (weight 80): isd.GaussianEMRestraint
      -log(2 MD / (MM + DD)) + slope sum |x_m - x_d|
  with MD over model-data pairs and MM over model-model pairs within the
  close-pair cutoff (+ em_slack, 0 by default), MM including the self
  terms. Model
  covariances: 'rotate' (from the current orientation) or 'frozen' (the
  core::Gaussian cache that C++ actually uses, salilab/imp#1119).
"""
import math

import numpy as np

_F4PI3 = 4.0 / 3.0 * math.pi
_GAUSS_NORM = 1. / (2. * math.pi) ** 1.5


# ---------------------------------------------------------------------------
# array helpers (jax.numpy or numpy via `xp`)
# ---------------------------------------------------------------------------
def quat_to_rot(q, xp):
    """Rotation matrices (..., 3, 3) from (w, x, y, z) quaternions, as
       Eigen::Quaterniond::toRotationMatrix (used by core::Gaussian)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2. * x, 2. * y, 2. * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return xp.stack([
        xp.stack([1. - (tyy + tzz), txy - twz, txz + twy], -1),
        xp.stack([txy + twz, 1. - (txx + tzz), tyz - twx], -1),
        xp.stack([txz - twy, tyz + twx, 1. - (txx + tyy)], -1)], -2)


def gaussian_covariance(rot, var, xp):
    """R diag(var) R^T for (..., 3, 3) rotations and (..., 3) variances"""
    return xp.einsum('...ij,...j,...kj->...ik', rot, var, rot)


def gmm_overlap(xa, ca, ma, xb, cb, mb, xp):
    """IMP::isd::score_gaussian_overlap for all pairs (a_i, b_j).
       x: (n, 3) centers, c: (n, 3, 3) covariances, m: (n,) masses.
       Returns (na, nb)."""
    s = ca[:, None] + cb[None]
    v = xb[None] - xa[:, None]
    a, b, c = s[..., 0, 0], s[..., 0, 1], s[..., 0, 2]
    d, e, f = s[..., 1, 1], s[..., 1, 2], s[..., 2, 2]
    # cofactors of the symmetric 3x3 matrix
    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - c * d
    det = a * c00 + b * c01 + c * c02
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    quad = (c00 * vx * vx + c11 * vy * vy + c22 * vz * vz
            + 2. * (c01 * vx * vy + c02 * vx * vz + c12 * vy * vz)) / det
    return (ma[:, None] * mb[None] * _GAUSS_NORM / xp.sqrt(det)
            * xp.exp(-0.5 * quad))


class TutSystem:
    """Host-side (numpy, float64) description of the tutorial system"""

    def __init__(self, S, cov_mode='rotate', em_slack=0.):
        import IMP
        import IMP.atom
        import IMP.container
        import IMP.core
        import IMP.isd
        from build_system import REGISTRY
        m = self.m = S.m
        self.S = S
        self.cov_mode = cov_mode
        pi_ = IMP.ParticleIndex
        m.update()
        # restraints are picked by name below: refuse anything else
        for r in S.rs.get_restraints():
            n = r.get_name()
            if not (n in ('ConnectivityRestraint', 'ExcludedVolumeSphere',
                          'GaussianEMRestraint') or n.startswith(
                              'CrossLinkingMassSpectrometryRestraint')):
                raise NotImplementedError('restraint ' + n)

        # ---------------- movers, in SerialMover order --------------------
        movers, bodies, beads = [], [], []
        for mv in S.movers:
            tn = type(mv).__name__
            if tn == 'RigidBodyMover':
                mv = IMP.core.RigidBodyMover.get_from(mv)
                (rbp,) = [IMP.Particle.get_from(o) for o in mv.get_outputs()]
                bodies.append(IMP.core.RigidBody(rbp))
                movers.append(('rb', len(bodies) - 1,
                               mv.get_maximum_translation(),
                               mv.get_maximum_rotation()))
            elif tn == 'BallMover':
                mv = IMP.core.BallMover.get_from(mv)
                (p,) = [IMP.Particle.get_from(o) for o in mv.get_outputs()]
                pi = p.get_index()
                if not IMP.core.NonRigidMember.get_is_setup(m, pi):
                    raise NotImplementedError('free bead')
                beads.append(int(pi))
                movers.append(('bead', len(beads) - 1, mv.get_radius(), 0.))
            else:
                raise NotImplementedError(tn)
        self.movers = movers
        self.nb, self.nbead = len(bodies), len(beads)
        self.bodies = bodies
        self.body_pi = np.array([int(b.get_particle_index())
                                 for b in bodies])
        self.bead_pi = np.array(beads)
        moving_body = {int(b.get_particle_index()): i
                       for i, b in enumerate(bodies)}
        for p in beads:
            owner = IMP.core.NonRigidMember(m, pi_(p)).get_rigid_body()
            if int(owner.get_particle_index()) in moving_body:
                raise NotImplementedError('bead of a moving body')
        rf = [b.get_reference_frame().get_transformation_to() for b in bodies]
        self.quat0 = np.array([t.get_rotation().get_quaternion() for t in rf])
        self.trans0 = np.array([t.get_translation() for t in rf])
        self.bead0 = np.array([IMP.core.XYZ(m, pi_(p)).get_coordinates()
                               for p in beads])

        # ---------------- particle rows -----------------------------------
        # kind: 0 static, 1 member of moving body `ref`, 2 bead `ref`
        rows, rowpos = [], {}
        bead_index = {p: i for i, p in enumerate(beads)}

        def row(p):
            p = int(p)
            if p not in rowpos:
                rowpos[p] = len(rows)
                rows.append(p)
            return rowpos[p]

        def body_of(p):
            p = pi_(int(p))
            if IMP.core.RigidMember.get_is_setup(m, p):
                return int(IMP.core.RigidMember(m, p).get_rigid_body()
                           .get_particle_index())
            return None

        # connectivity
        cp, cmean, ck = [], [], []
        for r in S.rs.get_restraints():
            if r.get_name() != 'ConnectivityRestraint':
                continue
            for c in IMP.RestraintSet.get_from(r).get_restraints():
                c = IMP.core.PairRestraint.get_from(c)
                a, b = [IMP.Particle.get_from(o).get_index()
                        for o in c.get_inputs()]
                ps = c.get_score_object().get_derived_object()
                tn = type(ps).__name__
                # IMP.core.SphereDistancePairScore(HarmonicUpperBound) gives
                # the specialized HarmonicUpperBoundSphereDistancePairScore
                if tn == 'HarmonicUpperBoundSphereDistancePairScore':
                    mean, k = ps.get_x0(), ps.get_k()
                elif tn == 'SphereDistancePairScore':
                    uf = ps.get_score_functor().get_unary_function() \
                        .get_derived_object()
                    if type(uf).__name__ != 'HarmonicUpperBound':
                        raise NotImplementedError(type(uf).__name__)
                    mean, k = uf.get_mean(), uf.get_k()
                else:
                    raise NotImplementedError(tn)
                if c.get_weight() != 1.0 or r.get_weight() != 1.0:
                    raise NotImplementedError('connectivity weight')
                cp.append((row(a), row(b)))
                cmean.append(mean)
                ck.append(k)
        self.conn_pairs = np.array(cp)
        self.conn_mean = np.array(cmean)
        self.conn_k = np.array(ck)

        # excluded volume
        cpc = S.ev.cpc
        if cpc.get_number_of_pair_filters():
            raise NotImplementedError('pair filters')
        evr = IMP.container.PairsRestraint.get_from(
            S.ev.rs.get_restraints()[0])
        self.ev_k = evr.get_score_object().get_derived_object().get_k()
        evp = [int(p) for p in cpc.get_all_possible_indexes()]
        self.ev_rows = np.array([row(p) for p in evp])
        lab = []
        for i, p in enumerate(evp):
            b = body_of(p)
            lab.append(b if b is not None else -1 - i)
        self.ev_label = np.array(lab)

        # crosslinks
        xr, xsame, xsi, xsj, xpsi = [], [], [], [], []
        length = slope = None
        self.xl_groups = []
        for xl in (S.xl1, S.xl2):
            if xl.linear.evaluate(1.0) != 0.0 or xl.linear.evaluate(7.) != 0.:
                raise NotImplementedError('XL linear term')
            (lwr,) = xl.rs.get_restraints()
            lw = IMP.isd.LogWrapper.get_from(lwr)
            if lw.get_weight() != 1.0 or xl.rs.get_weight() != 1.0:
                raise NotImplementedError('XL weight')
            start = len(xr)
            for r in lw.restraints:
                d = IMP.isd.CrossLinkMSRestraint.get_from(r)
                if (d.get_number_of_contributions() != 1
                        or d.get_log_prob() or d.get_is_length_variable()
                        or not d.get_has_slope()):
                    raise NotImplementedError('XL variant')
                if length is None:
                    length, slope = d.get_length(), d.get_slope()
                if (d.get_length(), d.get_slope()) != (length, slope):
                    raise NotImplementedError('XL parameters differ')
                a, b = d.get_contribution_particle_indexes(0)
                si, sj = d.get_contribution_sigma_indexes(0)
                xr.append((row(a), row(b)))
                xsame.append(int(a) == int(b))
                sc = IMP.isd.Scale
                xsi.append(sc(m, si).get_scale())
                xsj.append(sc(m, sj).get_scale())
                xpsi.append(sc(m, d.get_contribution_psi_index(0))
                            .get_scale())
            self.xl_groups.append((start, len(xr)))
        self.xl_rows = np.array(xr)
        self.xl_same = np.array(xsame)
        self.xl_si = np.array(xsi)
        self.xl_sj = np.array(xsj)
        self.xl_psi = np.array(xpsi)
        self.xl_length, self.xl_slope = length, slope
        # constant XL parts: priors and the (slope 0) linear terms
        const = 0.
        for xl in (S.xl1, S.xl2):
            for r in (xl.rspsi, xl.rssig, xl.rslin):
                const += r.evaluate(False)      # includes the weight
        self.xl_const = const

        # EM
        (_, args, _), = [v for v in REGISTRY.values() if v[0] == 'isd_gem']
        model_ps = [int(p) for p in args[1]]
        dens_ps = [int(p) for p in args[2]]
        model_cut, dens_cut, em_slope = args[4], args[5], args[6]
        if args[8] or args[9]:
            raise NotImplementedError('EM backbone_slope / local')
        self.em_w = S.gemt.rs.get_weight()
        G = IMP.core.Gaussian
        mass = lambda p: IMP.atom.Mass(m, pi_(p)).get_mass()  # noqa: E731
        rad = lambda p: IMP.core.XYZR(m, pi_(p)).get_radius()  # noqa: E731
        xyz = lambda p: np.array(IMP.core.XYZ(m, pi_(p))  # noqa: E731
                                 .get_coordinates())
        self.em_rows = np.array([row(p) for p in model_ps])
        self.em_mass = np.array([mass(p) for p in model_ps])
        self.em_rad = np.array([rad(p) for p in model_ps])
        self.em_var = np.array([G(m, pi_(p)).get_variances()
                                for p in model_ps])
        # covariance in the owning body's frame (rotate mode) and the
        # covariance C++ has cached (frozen mode)
        cl, cf, own = [], [], []
        for p in model_ps:
            g = G(m, pi_(p))
            cf.append(np.asarray(g.get_global_covariance()))
            b = moving_body[body_of(p)]
            own.append(b)
            rot = IMP.core.RigidMember(m, pi_(p)) \
                .get_internal_transformation().get_rotation()
            R = np.array([rot.get_rotation_matrix_row(i) for i in range(3)])
            cl.append(gaussian_covariance(R, np.asarray(g.get_variances()),
                                          np))
        self.em_cov_local = np.array(cl)
        self.em_cov_frozen = np.array(cf)
        self.em_owner = np.array(own)
        self.d_xyz = np.array([xyz(p) for p in dens_ps])
        self.d_cov = np.array([np.asarray(G(m, pi_(p))
                                          .get_global_covariance())
                               for p in dens_ps])
        self.d_mass = np.array([mass(p) for p in dens_ps])
        self.d_rad = np.array([rad(p) for p in dens_ps])
        self.em_slope = em_slope
        # close pairs: sphere distance < cutoff (+ em_slack). The C++
        # close-pair containers hold every such pair plus, depending on
        # when the list was last rebuilt, some pairs up to cutoff + 2 slack
        # (slack 1 A); see em_masks_from_cpp() for an exact comparison.
        self.em_mm_cut = model_cut + em_slack
        self.em_md_cut = dens_cut + em_slack
        self.em_md_mask = self.em_mm_mask = None
        dd = gmm_overlap(self.d_xyz, self.d_cov, self.d_mass,
                         self.d_xyz, self.d_cov, self.d_mass, np)
        self.em_dd = dd.sum()
        # self terms use whatever covariance C++ had at setup; they depend
        # only on det(cov), i.e. not on the orientation
        cs = self.em_cov_frozen
        self.em_self_mm = sum(gmm_overlap(
            xyz(p)[None], cs[i:i + 1], self.em_mass[i:i + 1],
            xyz(p)[None], cs[i:i + 1], self.em_mass[i:i + 1], np)[0, 0]
            for i, p in enumerate(model_ps))
        self.em_model_pi = np.array(model_ps)
        self.em_dens_pi = np.array(dens_ps)

        # ---------------- row tables -------------------------------------
        self.rows = np.array(rows)
        n = self.n = len(rows)
        kind = np.zeros(n, np.int64)
        ref = np.zeros(n, np.int64)
        local = np.zeros((n, 3))
        static = np.zeros((n, 3))
        radius = np.zeros(n)
        for i, p in enumerate(rows):
            pp = pi_(p)
            static[i] = IMP.core.XYZ(m, pp).get_coordinates()
            radius[i] = (IMP.core.XYZR(m, pp).get_radius()
                         if IMP.core.XYZR.get_is_setup(m, pp) else 0.)
            b = body_of(p)
            if b is not None and b in moving_body:
                kind[i], ref[i] = 1, moving_body[b]
                local[i] = IMP.core.RigidMember(m, pp) \
                    .get_internal_coordinates()
            elif p in bead_index:
                kind[i], ref[i] = 2, bead_index[p]
        self.row_kind, self.row_ref = kind, ref
        self.row_local, self.row_static, self.row_r = local, static, radius

        # the static scores (pairs/terms that involve no sampled particle)
        # are part of every term below; nothing is dropped

    def em_masks_from_cpp(self):
        """Use the pairs in the C++ EM close-pair containers right now
           (call after the C++ restraint was evaluated)"""
        import IMP.container
        gem = self.S.gemt.gaussianEM_restraint
        im = {int(p): i for i, p in enumerate(self.em_model_pi)}
        idn = {int(p): i for i, p in enumerate(self.em_dens_pi)}
        nm, nd = len(im), len(idn)
        md = np.zeros((nm, nd), bool)
        mm = np.zeros((nm, nm), bool)
        for o in gem.get_inputs():
            tn = o.get_type_name()
            if tn.startswith('CloseBipartitePairContainer'):
                c = IMP.container.CloseBipartitePairContainer.get_from(o)
                for a, b in c.get_indexes():
                    md[im[int(a)], idn[int(b)]] = True
            elif tn.startswith('ClosePairContainer'):
                c = IMP.container.ClosePairContainer.get_from(o)
                for a, b in c.get_indexes():
                    i, j = sorted((im[int(a)], im[int(b)]))
                    mm[i, j] = True
        self.em_md_mask, self.em_mm_mask = md, mm

    # ------------------------------------------------------------------
    def positions(self, quat, trans, bead, xp=np):
        R = quat_to_rot(quat, xp)
        kind = xp.asarray(self.row_kind)
        ref = xp.asarray(self.row_ref)
        mem = xp.einsum('nij,nj->ni', R[ref * (kind == 1)],
                        xp.asarray(self.row_local)) \
            + trans[ref * (kind == 1)]
        bd = bead[ref * (kind == 2)]
        return xp.where((kind == 1)[:, None], mem,
                        xp.where((kind == 2)[:, None], bd,
                                 xp.asarray(self.row_static)))

    def em_covariances(self, quat, xp=np):
        if self.cov_mode == 'frozen':
            return xp.asarray(self.em_cov_frozen)
        R = quat_to_rot(quat, xp)[xp.asarray(self.em_owner)]
        return xp.einsum('nij,njk,nlk->nil', R,
                         xp.asarray(self.em_cov_local), R)


# ---------------------------------------------------------------------------
# term math
# ---------------------------------------------------------------------------
def conn_energy(xa, ra, xb, rb, mean, k, xp):
    d = xp.sqrt(xp.sum((xa - xb) ** 2, axis=-1)) - ra - rb
    x = xp.maximum(d - mean, 0.)
    return 0.5 * k * x * x


def sphere_cap(r1, r2, d, xp, f32=False):
    """isd::CrossLinkMSRestraint::sphere_cap. Its arguments are floats in
       C++, so the comparisons, (r1 + r2 - d) and the second polynomial are
       float arithmetic; f32=True reproduces that (numpy only)."""
    if f32:
        f = np.float32
        r1, r2, d = (np.asarray(r1, f), np.broadcast_to(np.asarray(r2, f),
                     np.shape(d)), np.asarray(d, f))
        big, small = np.maximum(r1, r2), np.minimum(r1, r2)
        short = np.minimum(_F4PI3 * r1.astype(float) ** 3,
                           _F4PI3 * r2.astype(float) ** 3)
        a = (r1 + r2 - d).astype(float)
        dsafe = np.where(d > 0, d, f(1)).astype(float)
        poly = (d * d + f(2) * d * r1 - f(3) * r1 * r1 + f(2) * d * r2
                + f(6) * r1 * r2 - f(3) * r2 * r2).astype(float)
        mid = math.pi / 12 / dsafe * a * a * poly
        return np.where(d <= big - small, short,
                        np.where(d >= r1 + r2, 0., mid))
    big = xp.maximum(r1, r2)
    small = xp.minimum(r1, r2)
    short = xp.minimum(_F4PI3 * r1 ** 3, _F4PI3 * r2 ** 3)
    dsafe = xp.where(d > 0, d, 1.0)
    mid = ((math.pi / 12 / dsafe * (r1 + r2 - d) * (r1 + r2 - d))
           * (d * d + 2 * d * r1 - 3 * r1 * r1 + 2 * d * r2
              + 6 * r1 * r2 - 3 * r2 * r2))
    return xp.where(d <= big - small, short,
                    xp.where(d >= r1 + r2, 0., mid))


def xl_neglogp(dist, si, sj, psi, length, slope, xp, f32=False):
    """-log p of single-contribution CrossLinkMSRestraints"""
    dist = xp.maximum(dist, 0.0001)
    half = length / 2
    voli, volj = _F4PI3 * si ** 3, _F4PI3 * sj ** 3
    xlvol = _F4PI3 * half ** 3
    close = dist < si + sj
    fi = xp.where(close, xp.minimum(voli, xlvol),
                  sphere_cap(si, half, xp.abs(dist - sj - half), xp, f32))
    fj = xp.where(close, xp.minimum(volj, xlvol),
                  sphere_cap(sj, half, xp.abs(dist - si - half), xp, f32))
    pofr = fi * fj / voli / volj
    term = (psi * (1.0 - pofr) + pofr * (1 - psi)) * xp.exp(-slope * dist)
    return -xp.log(term)


def ev_energy(X, R, lab, k, xp):
    """SoftSpherePairScore over all pairs i < j with different labels"""
    d = xp.sqrt(xp.sum((X[:, None] - X[None]) ** 2, axis=-1) + 1e-300)
    x = xp.minimum(d - R[:, None] - R[None], 0.)
    e = 0.5 * k * x * x
    n = X.shape[0]
    iu = np.triu(np.ones((n, n), bool), 1)
    keep = xp.asarray(iu & (lab[:, None] != lab[None]))
    return xp.sum(xp.where(keep, e, 0.))


def em_parts(s, xm, cm, xp):
    """(MD, MM cross terms, slope sum) of the isd.GaussianEMRestraint for
       model Gaussians at xm with covariances cm"""
    dx = xp.asarray(s.d_xyz)
    ov = gmm_overlap(xm, cm, xp.asarray(s.em_mass), dx,
                     xp.asarray(s.d_cov), xp.asarray(s.d_mass), xp)
    dist = xp.sqrt(xp.sum((xm[:, None] - dx[None]) ** 2, axis=-1))
    sph = dist - xp.asarray(s.em_rad)[:, None] - xp.asarray(s.d_rad)[None]
    mdm = (sph < s.em_md_cut if s.em_md_mask is None
           else xp.asarray(s.em_md_mask))
    md = xp.sum(xp.where(mdm, ov, 0.))
    mm = gmm_overlap(xm, cm, xp.asarray(s.em_mass), xm, cm,
                     xp.asarray(s.em_mass), xp)
    dmm = xp.sqrt(xp.sum((xm[:, None] - xm[None]) ** 2, axis=-1) + 1e-300)
    sphm = dmm - xp.asarray(s.em_rad)[:, None] - xp.asarray(s.em_rad)[None]
    n = xm.shape[0]
    iu = xp.asarray(np.triu(np.ones((n, n), bool), 1))
    mmm = (iu & (sphm < s.em_mm_cut) if s.em_mm_mask is None
           else xp.asarray(s.em_mm_mask))
    mmx = xp.sum(xp.where(mmm, mm, 0.))
    return md, mmx, xp.sum(dist)


def energy_terms(s, quat, trans, bead, xp=np, f32_caps=False):
    """Per-term energies (C++ restraint-set names) for the sampled DOFs"""
    X = s.positions(quat, trans, bead, xp)
    r = xp.asarray(s.row_r)
    out = {}
    cp = s.conn_pairs
    out['connectivity'] = xp.sum(conn_energy(
        X[cp[:, 0]], r[cp[:, 0]], X[cp[:, 1]], r[cp[:, 1]],
        xp.asarray(s.conn_mean), xp.asarray(s.conn_k), xp))
    er = s.ev_rows
    out['ev'] = ev_energy(X[er], r[er], s.ev_label, s.ev_k, xp)
    xr = s.xl_rows
    dist = xp.where(xp.asarray(s.xl_same), 36.0 / 35.0 * r[xr[:, 0]],
                    xp.sqrt(xp.sum((X[xr[:, 0]] - X[xr[:, 1]]) ** 2,
                                   axis=-1)))
    nl = xl_neglogp(dist, xp.asarray(s.xl_si), xp.asarray(s.xl_sj),
                    xp.asarray(s.xl_psi), s.xl_length, s.xl_slope, xp,
                    f32_caps)
    out['xl'] = xp.sum(nl) + s.xl_const
    xm = X[s.em_rows]
    cm = s.em_covariances(quat, xp)
    md, mmx, dsum = em_parts(s, xm, cm, xp)
    denom = s.em_self_mm + 2. * mmx + s.em_dd
    out['em'] = s.em_w * (-xp.log(2. * md / denom) + s.em_slope * dsum)
    out['total'] = out['connectivity'] + out['ev'] + out['xl'] + out['em']
    return out


def get_imp_dofs(s):
    """(quat, trans, bead) of the current IMP model"""
    import IMP
    import IMP.core
    q, t = [], []
    for b in s.bodies:
        tr = b.get_reference_frame().get_transformation_to()
        q.append(tr.get_rotation().get_quaternion())
        t.append(tr.get_translation())
    bead = [IMP.core.XYZ(s.m, IMP.ParticleIndex(int(p))).get_coordinates()
            for p in s.bead_pi]
    return np.array(q), np.array(t), np.array(bead)


def set_imp_dofs(s, quat, trans, bead):
    """Write sampled DOFs into the IMP model (bead global positions become
       internal coordinates of their fixed owner bodies)"""
    import IMP
    import IMP.algebra
    import IMP.core
    m = s.m
    for b, qq, tt in zip(s.bodies, quat, trans):
        b.set_reference_frame(IMP.algebra.ReferenceFrame3D(
            IMP.algebra.Transformation3D(
                IMP.algebra.Rotation3D(*[float(x) for x in qq]),
                IMP.algebra.Vector3D(*[float(x) for x in tt]))))
    for p, x in zip(s.bead_pi, bead):
        nr = IMP.core.NonRigidMember(m, IMP.ParticleIndex(int(p)))
        tr = nr.get_rigid_body().get_reference_frame()
        nr.set_internal_coordinates(tr.get_local_coordinates(
            IMP.algebra.Vector3D(*[float(v) for v in x])))
    m.update()


def cpp_terms(S):
    """The same per-term split from the C++ restraint set"""
    out = {'connectivity': 0., 'ev': 0., 'xl': 0., 'em': 0.}
    for r in S.rs.get_restraints():
        n = r.get_name()
        v = r.evaluate(False)            # includes the weight
        if n == 'ConnectivityRestraint':
            out['connectivity'] += v
        elif n == 'ExcludedVolumeSphere':
            out['ev'] += v
        elif n.startswith('CrossLinkingMassSpectrometryRestraint'):
            out['xl'] += v
        elif n == 'GaussianEMRestraint':
            out['em'] += v
        else:
            raise NotImplementedError(n)
    out['total'] = sum(out.values())
    return out
