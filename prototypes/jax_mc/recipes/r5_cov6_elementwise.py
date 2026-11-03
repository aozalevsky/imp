"""Recipe R5: Closed-form 6-component covariances and elementwise 3x3 products.

Specialization level:
  Level 1 (Core geometry math in `jax_mc.terms.geom` and tuple-unpacked
  primitives used by Level-3 templates).

Preconditions (asserted):
  1. Symmetric 3x3 covariance matrices are passed as 6 upper-triangle
     components `(c00, c01, c02, c11, c12, c22)` with positive diagonals.
  2. Quaternions have 4 components `(w, x, y, z)` on axis 0.
"""
import numpy as np
import jax.numpy as jnp

from jax_mc import engine as E
from jax_mc.terms import geom

SYM = geom.SYM
GAUSS_NORM = geom.GAUSS_NORM


def quat_to_rot_imp(q, xp=jnp):
    """(4, ...) unit quaternions -> (3, 3, ...) rotation matrix in
       IMP::algebra::Rotation3D form (w^2 + x^2 - y^2 - z^2, ...)."""
    assert q.shape[0] == 4, f'R5 requires (4, ...) quaternion, got {q.shape}'
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.stack([
        xp.stack([w * w + x * x - y * y - z * z, 2. * (x * y - w * z),
                  2. * (x * z + w * y)]),
        xp.stack([2. * (x * y + w * z), w * w - x * x + y * y - z * z,
                  2. * (y * z - w * x)]),
        xp.stack([2. * (x * z - w * y), 2. * (y * z + w * x),
                  w * w - x * x - y * y + z * z])])


def quat_to_rot_eigen9(q):
    """(4, ...) unit quaternions -> 9-tuple of rotation matrix entries in
       Eigen's toRotationMatrix form (1 - 2(y^2 + z^2), ...), as used by
       IMP::core::Gaussian."""
    assert q.shape[0] == 4, f'R5 requires (4, ...) quaternion, got {q.shape}'
    w, x, y, z = q[0], q[1], q[2], q[3]
    tx, ty, tz = 2. * x, 2. * y, 2. * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return (
        1. - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1. - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1. - (txx + tyy),
    )


def rot_cov6(R9, c6):
    """6 upper-triangle entries (c00, c01, c02, c11, c12, c22) of R C R^T for
       symmetric 6-component covariance c6 and 9-tuple rotation R9."""
    assert len(R9) == 9 and len(c6) == 6, (
        f'R5 rot_cov6 requires 9-tuple R9 and 6-component c6, got '
        f'{len(R9)}, {len(c6)}')
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R9
    k00, k01, k02, k11, k12, k22 = c6[0], c6[1], c6[2], c6[3], c6[4], c6[5]
    t00 = r00 * k00 + r01 * k01 + r02 * k02
    t01 = r00 * k01 + r01 * k11 + r02 * k12
    t02 = r00 * k02 + r01 * k12 + r02 * k22
    t10 = r10 * k00 + r11 * k01 + r12 * k02
    t11 = r10 * k01 + r11 * k11 + r12 * k12
    t12 = r10 * k02 + r11 * k12 + r12 * k22
    t20 = r20 * k00 + r21 * k01 + r22 * k02
    t21 = r20 * k01 + r21 * k11 + r22 * k12
    t22 = r20 * k02 + r21 * k12 + r22 * k22
    return (
        t00 * r00 + t01 * r01 + t02 * r02,
        t00 * r10 + t01 * r11 + t02 * r12,
        t00 * r20 + t01 * r21 + t02 * r22,
        t10 * r10 + t11 * r11 + t12 * r12,
        t10 * r20 + t11 * r21 + t12 * r22,
        t20 * r20 + t21 * r21 + t22 * r22,
    )


def overlap_cov6(xa, ca6, ma, xb, cb6, mb, xp=jnp):
    """IMP::isd::score_gaussian_overlap from 6 upper-triangle covariance
       entries, purely elementwise (no matmul / dot_general / TF32)."""
    assert len(ca6) == 6 and len(cb6) == 6
    a = ca6[0] + cb6[0]
    b = ca6[1] + cb6[1]
    c = ca6[2] + cb6[2]
    d = ca6[3] + cb6[3]
    e = ca6[4] + cb6[4]
    f = ca6[5] + cb6[5]
    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - c * d
    det = a * c00 + b * c01 + c * c02
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    vx, vy, vz = xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]
    quad = (c00 * vx * vx + c11 * vy * vy + c22 * vz * vz
            + 2. * (c01 * vx * vy + c02 * vx * vz + c12 * vy * vz)) / det
    return ma * mb * GAUSS_NORM / xp.sqrt(det) * xp.exp(-0.5 * quad)


def rotate_cf(R, local, t):
    """Elementwise 3x3 rotation R (3, 3, C) applied to local points (3, m)
       plus translation t (3, C) -> (3, m, C) without dot_general."""
    assert R.shape[:2] == (3, 3) and local.shape[0] == 3 and t.shape[0] == 3
    return (R[:, 0, None] * local[0][None, :, None]
            + R[:, 1, None] * local[1][None, :, None]
            + R[:, 2, None] * local[2][None, :, None]) + t[:, None]


def check_preconditions(s):
    """Assert R5 preconditions on System s."""
    for t in s.terms:
        if hasattr(t, 'cloc') and hasattr(t, 'cglob'):
            assert t.cloc.shape[0] == 6 and t.cglob.shape[0] == 6
            assert np.all(t.cloc[[0, 3, 5]] > 0.)
            assert np.all(t.cglob[[0, 3, 5]] > 0.)


def factory(s):
    """R5 is built into jax_mc.terms.geom and engine; verify preconditions and
       return the reference Engine."""
    check_preconditions(s)
    return E.Engine(s)
