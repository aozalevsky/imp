"""Shared math for terms, chain-last: vectors are (3, ..., C) arrays,
symmetric 3x3 matrices are 6 components (xx, xy, xz, yy, yz, zz) on axis 0.
Everything is elementwise (no dot_general), so float32 on GPU never goes
through TF32."""
import math

import jax.numpy as jnp

FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi
GAUSS_NORM = 0.06349363593424097      # 1 / (2 pi)^(3/2), as in IMP::isd

SYM = ((0, 1, 2), (1, 3, 4), (2, 4, 5))   # 6-component index of (i, j)


def norm(v):
    return jnp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def quat_to_rot(q):
    """(4, ...) unit quaternions (w, x, y, z) -> (3, 3, ...), as Eigen's
       toRotationMatrix (what IMP::algebra::Rotation3D and core::Gaussian
       use)"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    tx, ty, tz = 2. * x, 2. * y, 2. * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return jnp.stack([
        jnp.stack([1. - (tyy + tzz), txy - twz, txz + twy]),
        jnp.stack([txy + twz, 1. - (txx + tzz), tyz - twx]),
        jnp.stack([txz - twy, tyz + twx, 1. - (txx + tyy)])])


def rotate_cov6(R, c6):
    """R C R^T for rotations R (3, 3, ...) and covariances c6 (6, ...)"""
    M = [[sum(R[i, k] * c6[SYM[k][j]] for k in range(3)) for j in range(3)]
         for i in range(3)]
    out = []
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        out.append(sum(M[i][k] * R[j, k] for k in range(3)))
    return jnp.stack(out)


def cov6_from_matrix(c):
    """(..., 3, 3) numpy/jax -> (6, ...)"""
    return jnp.stack([c[..., 0, 0], c[..., 0, 1], c[..., 0, 2],
                      c[..., 1, 1], c[..., 1, 2], c[..., 2, 2]])


def gaussian_overlap(xa, ca, ma, xb, cb, mb, xp=jnp):
    """IMP::isd::score_gaussian_overlap, broadcasting: x (3, ...),
       c (6, ...), m (...). xp=numpy for host-side float64 constants."""
    s = [ca[k] + cb[k] for k in range(6)]
    a, b, c, d, e, f = s
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


def sphere_cap(r1, r2, d, cpp_float=False):
    """isd::CrossLinkMSRestraint::sphere_cap. In C++ its arguments are
       floats, so the comparisons, (r1 + r2 - d) and the second polynomial
       are single-precision arithmetic; cpp_float=True reproduces that (only
       for exact float64 checks against C++, ~1e-8 relative)."""
    if cpp_float:
        f, g = jnp.float32, d.dtype
        r1, r2, d = (jnp.asarray(r1).astype(f),
                     jnp.broadcast_to(jnp.asarray(r2), d.shape).astype(f),
                     d.astype(f))
        big, small = jnp.maximum(r1, r2), jnp.minimum(r1, r2)
        short = jnp.minimum(FOUR_THIRDS_PI * r1.astype(g) ** 3,
                            FOUR_THIRDS_PI * r2.astype(g) ** 3)
        a = (r1 + r2 - d).astype(g)
        dsafe = jnp.where(d > 0, d, f(1)).astype(g)
        poly = (d * d + f(2) * d * r1 - f(3) * r1 * r1 + f(2) * d * r2
                + f(6) * r1 * r2 - f(3) * r2 * r2).astype(g)
        mid = math.pi / 12 / dsafe * a * a * poly
        return jnp.where(d <= big - small, short,
                         jnp.where(d >= r1 + r2, 0., mid))
    big = jnp.maximum(r1, r2)
    small = jnp.minimum(r1, r2)
    short = jnp.minimum(FOUR_THIRDS_PI * r1 ** 3, FOUR_THIRDS_PI * r2 ** 3)
    dsafe = jnp.where(d > 0, d, 1.0)
    mid = ((math.pi / 12 / dsafe * (r1 + r2 - d) * (r1 + r2 - d))
           * (d * d + 2 * d * r1 - 3 * r1 * r1 + 2 * d * r2
              + 6 * r1 * r2 - 3 * r2 * r2))
    return jnp.where(d <= big - small, short,
                     jnp.where(d >= r1 + r2, 0., mid))


def xl_term(dist, s0, s1, psi, length, slope, cpp_float=False):
    """One CrossLinkMSRestraint contribution's probability term"""
    dist = jnp.maximum(dist, 0.0001)
    half = length / 2.
    xlvol = FOUR_THIRDS_PI * half ** 3
    voli, volj = FOUR_THIRDS_PI * s0 ** 3, FOUR_THIRDS_PI * s1 ** 3
    close = dist < s0 + s1
    fi = jnp.where(close, jnp.minimum(voli, xlvol),
                   sphere_cap(s0, half, jnp.abs(dist - s1 - half),
                              cpp_float))
    fj = jnp.where(close, jnp.minimum(volj, xlvol),
                   sphere_cap(s1, half, jnp.abs(dist - s0 - half),
                              cpp_float))
    pofr = fi * fj / voli / volj
    term = psi * (1.0 - pofr) + pofr * (1 - psi)
    if slope is not None:
        term = term * jnp.exp(-slope * dist)
    return term
