"""JAX implementations of the NPC_Basket restraints that IMP's JAX path
lacks, written as IMP-style `_get_jax()` methods (`install()` attaches them):

* IMP.bayesianem.GaussianEMRestraint: model GMM vs data GMM, dense over
  all model x data pairs with the model-data cutoff as a mask, plus the
  slope term. Same math as GaussianEMRestraint::unprotected_evaluate.
* IMP.npc.ZAxialPositionRestraint, IMP.npc.MembraneSurfaceLocationRestraint.
* IMP.container.PairsRestraint over a ClosePairContainer with a
  SoftSpherePairScore (PMI's ExcludedVolumeSphere): dense over all pairs of
  the container's particles, minus the pairs that RigidClosePairsFinder
  skips (both in the same rigid body). The score is zero for non-overlapping
  pairs, so this equals the C++ score whatever the close-pair list holds.
  (Stock IMP uses all pairs, including those within one rigid body.)

The math lives in plain array functions (`em_*`, `ev_*`, `zaxial_*`,
`membrane_*`) so the fused sampler (npc_sampler.py) can reuse it on its own
compact arrays.
"""
import math

import os

import numpy as np

# 1 / (2 pi)^(3/2), as in IMP::isd::score_gaussian_overlap
_GAUSS_NORM = 0.06349363593424097


# ---------------------------------------------------------------------------
# array math (jax.numpy or numpy via `xp`)
# ---------------------------------------------------------------------------
def quat_to_rot(q, xp):
    """Rotation matrices (..., 3, 3) from (w, x, y, z) quaternions, as
       Eigen::Quaterniond::toRotationMatrix (used by core::Gaussian)."""
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


class EMParams:
    """Constants of one GaussianEMRestraint (host numpy, float64)"""
    def __init__(self, m, model_pis, density_pis, sigma_pi, slope,
                 cutoff):
        import IMP.core
        import IMP.atom
        self.model_pis = np.asarray(model_pis, np.int64)
        self.density_pis = np.asarray(density_pis, np.int64)
        self.sigma_pi = int(sigma_pi)
        self.slope = float(slope)
        self.cutoff = float(cutoff)
        G = IMP.core.Gaussian
        pi = IMP.ParticleIndex
        self.model_var = np.array([G(m, pi(int(p))).get_variances()
                                   for p in self.model_pis])
        self.model_mass = np.array([IMP.atom.Mass(m, pi(int(p))).get_mass()
                                    for p in self.model_pis])
        self.model_radius = np.array([IMP.core.XYZR(m, pi(int(p)))
                                      .get_radius() for p in self.model_pis])
        self.density_xyz = np.array([IMP.core.XYZ(m, pi(int(p)))
                                     .get_coordinates()
                                     for p in self.density_pis])
        self.density_cov = np.array([np.asarray(
            G(m, pi(int(p))).get_global_covariance())
            for p in self.density_pis])
        self.density_mass = np.array([IMP.atom.Mass(m, pi(int(p)))
                                      .get_mass() for p in self.density_pis])
        self.density_radius = np.array([IMP.core.XYZR(m, pi(int(p)))
                                        .get_radius()
                                        for p in self.density_pis])
        # per-density sum of data-data overlaps and the constant part of
        # the score (GaussianEMRestraint::compute_initial_scores)
        dd = gmm_overlap(self.density_xyz, self.density_cov,
                         self.density_mass, self.density_xyz,
                         self.density_cov, self.density_mass, np)
        # C++ accumulates map_score_dd_[i1] += overlap(i1, i2)
        self.score_dd = dd.sum(axis=1)
        nd = len(self.density_pis)
        self.cached_term = (np.sum(np.log(self.score_dd))
                            + (3.0 + nd) / 2.0 * math.log(2.0)
                            + 0.5 * math.log(math.pi)
                            - math.lgamma(nd / 2.0))


def em_terms(xm, covm, mm, rm, p, xp):
    """Per model Gaussian: overlap with every data Gaussian (masked by the
       close-pair cutoff on sphere distance) (n, D), and the squared
       distance to the nearest data center (n,)."""
    dt = xm.dtype
    xd = xp.asarray(p.density_xyz, dtype=dt)
    ov = gmm_overlap(xm, covm, mm, xd, xp.asarray(p.density_cov, dtype=dt),
                     xp.asarray(p.density_mass, dtype=dt), xp)
    diff = xm[:, None] - xd[None]
    d2 = xp.sum(diff * diff, axis=-1)
    sphere = (xp.sqrt(d2) - rm[:, None]
              - xp.asarray(p.density_radius, dtype=dt)[None])
    ov = xp.where(sphere < p.cutoff, ov, 0.)
    return ov, xp.min(d2, axis=1)


def em_score_from_sums(score_dm, min_d2_sum_sqrt, scale, p, xp):
    """Final GaussianEMRestraint score from the per-density model overlap
       sums and the summed nearest-center distances."""
    dt = score_dm.dtype
    sdd = xp.asarray(p.score_dd, dtype=dt)
    lt = xp.log(scale * (score_dm + 1e-10) / (sdd + 1e-10))
    nd = len(p.density_pis)
    return (nd * 0.5 * xp.log(xp.sum(lt * lt)) + p.cached_term
            + p.slope * min_d2_sum_sqrt)


def em_score(xm, covm, scale, p, xp):
    dt = xm.dtype
    ov, min_d2 = em_terms(xm, covm, xp.asarray(p.model_mass, dtype=dt),
                          xp.asarray(p.model_radius, dtype=dt), p, xp)
    slope_sum = xp.sum(xp.sqrt(min_d2)) if p.slope > 0. else 0.
    return em_score_from_sums(xp.sum(ov, axis=0), slope_sum, scale, p, xp)


def ev_pair_energy(xa, ra, ba, xb, rb, bb, k, xp):
    """SoftSpherePairScore for all pairs (a_i, b_j), zero for pairs with
       equal labels (same rigid body, or same side of a bipartite set).
       (na, nb)"""
    diff = xa[:, None] - xb[None]
    d = xp.sqrt(xp.sum(diff * diff, axis=-1))
    x = xp.minimum(d - ra[:, None] - rb[None], 0.)
    return xp.where(ba[:, None] != bb[None], 0.5 * k * x * x, 0.)


def zaxial_score(z, lower, upper, sigma, xp):
    lo = xp.minimum(z - lower, 0.)
    up = xp.maximum(z - upper, 0.)
    return xp.sum(lo * lo + up * up) / sigma


def half_torus_distance(xyz, R, r, xp):
    """npc's half_torus_distance (signed distance from the pore membrane)"""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dxy = xp.sqrt(x * x + y * y)
    safe = xp.where(dxy > 1e-9, dxy, 1.)
    dtx = xp.where(dxy > 1e-9, x - x / safe * R, x - R)
    dty = xp.where(dxy > 1e-9, y - y / safe * R, y)
    inside = xp.sqrt(z * z + dtx * dtx + dty * dty) - r
    outside = xp.abs(z) - r
    return xp.where(dxy < R, inside, outside)


def membrane_score(xyz, R, r, half_thickness, sigma, xp):
    d = half_torus_distance(xyz, R, r, xp)
    return xp.sum(xp.where(xp.abs(d) > half_thickness, d * d, 0.)) / sigma


# ---------------------------------------------------------------------------
# reading parameters out of the C++ objects
# ---------------------------------------------------------------------------
def _info_floats(ri):
    return {ri.get_float_key(i): ri.get_float_value(i)
            for i in range(ri.get_number_of_float())}


def em_params_from_restraint(r):
    """EMParams from a C++ GaussianEMRestraint (no Python getters exist for
       the particle lists: inputs are model ps, density ps, sigma, container;
       debug() fills get_indexes() with the density ps)."""
    import IMP
    import IMP.isd
    m = r.get_model()
    r.debug()
    dens = [int(p) for p in r.get_indexes()]
    dset = set(dens)
    model, sigma = [], None
    for pi in _restraint_particle_indexes(r):
        if pi in dset:
            continue
        if IMP.isd.Scale.get_is_setup(m, IMP.ParticleIndex(int(pi))):
            sigma = int(pi)
        else:
            model.append(int(pi))
    # density ps in constructor order are not recoverable, but the score
    # does not depend on their order except via std::map (sorted index)
    info = _info_floats(r.get_static_info())
    return EMParams(m, model, sorted(dens), sigma, r.get_slope(),
                    info['density cutoff'])


def _restraint_particle_indexes(r):
    import IMP
    out = []
    for o in r.get_inputs():
        try:
            out.append(int(IMP.Particle.get_from(o).get_index()))
        except (ValueError, TypeError):
            pass
    return np.asarray(out, np.int64)


def zaxial_params_from_restraint(r):
    info = _info_floats(r.get_static_info())
    # consider_radius is not in the static info; PMI's default is False
    return dict(pis=_restraint_particle_indexes(r),
                lower=info['lower bound'], upper=info['upper bound'],
                sigma=info['sigma'])


def membrane_params_from_restraint(r):
    info = _info_floats(r.get_static_info())
    return dict(pis=_restraint_particle_indexes(r),
                R=info['major radius'], r=info['minor radius'],
                half_thickness=info['thickness'], sigma=info['sigma'])


def ev_params_from_restraint(r):
    """For a PairsRestraint with a SoftSpherePairScore over
       * a ClosePairContainer (distance 0, no pair filters, the default
         RigidClosePairsFinder): all pairs except those within one rigid
         body (label = rigid body, unique for non-members);
       * a CloseBipartitePairContainer (distance 0, no pair filters): all
         pairs between the two sets (label = 0 / 1 for set A / B).
       A pair contributes iff the labels differ. Zero distance means only
       overlapping pairs score, so the result does not depend on the
       close-pair list (slack) at all."""
    import IMP
    import IMP.core
    m = r.get_model()
    cpc = r.get_container().get_derived_object()
    if cpc.get_number_of_pair_filters():
        raise NotImplementedError('pair filters')
    k = r.get_score_object().get_derived_object().get_k()
    if cpc.get_type_name().startswith('CloseBipartitePairContainer'):
        pairs = np.asarray([[int(a), int(b)] for a, b in
                            cpc.get_range_indexes()], np.int64)
        A, B = np.unique(pairs[:, 0]), np.unique(pairs[:, 1])
        if len(np.intersect1d(A, B)):
            raise NotImplementedError('overlapping bipartite sets')
        pis = np.concatenate([A, B])
        label = np.concatenate([np.zeros(len(A), np.int64),
                                np.ones(len(B), np.int64)])
        return dict(pis=pis, label=label, k=k, bipartite=True)
    pis = np.asarray([int(p) for p in cpc.get_all_possible_indexes()],
                     np.int64)
    label = np.empty(len(pis), np.int64)
    for i, p in enumerate(pis):
        pi = IMP.ParticleIndex(int(p))
        if IMP.core.RigidMember.get_is_setup(m, pi):
            label[i] = int(IMP.core.RigidMember(m, pi).get_rigid_body()
                           .get_particle_index())
        else:
            label[i] = -1 - i          # unique: never excluded
    return dict(pis=pis, label=label, k=k, bipartite=False)


# ---------------------------------------------------------------------------
# IMP-style _get_jax() methods
# ---------------------------------------------------------------------------
_EM_REGISTRY = {}


def _em_get_jax(self, space):
    import jax.numpy as jnp
    import IMP.isd
    from IMP.core import _jax_rigid
    p = em_params_from_restraint(self)
    m = self.get_model()
    rb_index = {int(pi): i for i, pi in
                enumerate(_jax_rigid._get_rigid_body_indexes(m))}
    mrb = np.asarray([rb_index[int(pi)] for pi in p.model_pis], np.int32)
    model_pis = jnp.asarray(p.model_pis)
    var = jnp.asarray(p.model_var)

    def score(jm):
        xm = jm['xyz'][model_pis]
        q = jm['rigid_bodies'].quaternion[mrb]
        cov = gaussian_covariance(quat_to_rot(q, jnp), var.astype(xm.dtype),
                                  jnp)
        scale = jm['nuisance'][p.sigma_pi]
        return em_score(xm, cov, scale, p, jnp)
    return self._wrap_jax(score, keys=['rigid_bodies',
                                       IMP.isd.Scale.get_scale_key()])


def _zaxial_get_jax(self, space):
    import jax.numpy as jnp
    p = zaxial_params_from_restraint(self)
    pis = jnp.asarray(p['pis'])

    def score(jm):
        return zaxial_score(jm['xyz'][pis][:, 2], p['lower'], p['upper'],
                            p['sigma'], jnp)
    return self._wrap_jax(score)


def _membrane_get_jax(self, space):
    import jax.numpy as jnp
    p = membrane_params_from_restraint(self)
    pis = jnp.asarray(p['pis'])

    def score(jm):
        return membrane_score(jm['xyz'][pis], p['R'], p['r'],
                              p['half_thickness'], p['sigma'], jnp)
    return self._wrap_jax(score)


def install():
    """Attach the _get_jax() implementations to the IMP classes."""
    import jax
    # full float32 matrix products on GPUs, not TF32 (salilab/imp#1129);
    # costs ~0.5% here. Set JAX_DEFAULT_MATMUL_PRECISION to override.
    if 'JAX_DEFAULT_MATMUL_PRECISION' not in os.environ:
        jax.config.update('jax_default_matmul_precision', 'highest')
    import IMP.bayesianem
    import IMP.container
    import IMP.core
    import IMP.npc

    IMP.npc.ZAxialPositionRestraint._get_jax = _zaxial_get_jax
    IMP.npc.MembraneSurfaceLocationRestraint._get_jax = _membrane_get_jax

    # bayesianem's SWIG wrapper has no _object_cast_to_GaussianEMRestraint,
    # so Restraint.get_derived_object() raises NameError for it. Keep the
    # Python proxies created by the constructor and look them up by pointer.
    cls = IMP.bayesianem.GaussianEMRestraint
    if not getattr(cls, '_jax_installed', False):
        orig_init = cls.__init__

        def __init__(self, *args, **kw):
            orig_init(self, *args, **kw)
            _EM_REGISTRY[int(self.this)] = self
        cls.__init__ = __init__
        IMP.bayesianem._object_cast_to_GaussianEMRestraint = \
            lambda o: _EM_REGISTRY[int(o.this)]
        cls._get_jax = _em_get_jax
        cls._jax_installed = True

    orig_pairs = IMP.container.PairsRestraint._get_jax
    if not getattr(orig_pairs, '_npc', False):
        def _pairs_get_jax(self, space):
            import jax.numpy as jnp
            c = self.get_container()
            score = self.get_score_object().get_derived_object()
            if not (c.get_type_name().startswith(
                    ('ClosePairContainer', 'CloseBipartitePairContainer'))
                    and isinstance(score, IMP.core.SoftSpherePairScore)):
                return orig_pairs(self, space)
            p = ev_params_from_restraint(self)
            pis = jnp.asarray(p['pis'])
            label = jnp.asarray(p['label'])

            def evscore(jm):
                x, r = jm['xyz'][pis], jm['r'][pis]
                e = ev_pair_energy(x, r, label, x, r, label, p['k'], jnp)
                return 0.5 * jnp.sum(e)
            return self._wrap_jax(evscore)
        _pairs_get_jax._npc = True
        IMP.container.PairsRestraint._get_jax = _pairs_get_jax


def register_existing(objs):
    """Register already-constructed GaussianEMRestraint proxies (if install()
       ran after they were created)."""
    for o in objs:
        _EM_REGISTRY[int(o.this)] = o
