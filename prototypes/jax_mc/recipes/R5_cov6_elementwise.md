# R5: Closed-form 6-component covariances and elementwise 3×3 products

## What it does
Represents symmetric $3 \times 3$ Gaussian covariance matrices by their 6
upper-triangle entries `(c00, c01, c02, c11, c12, c22)` and evaluates:
1. Quaternion-to-rotation conversions as explicit scalar expressions:
   - `quat_to_rot_imp`: `IMP::algebra::Rotation3D` form (`w*w + x*x - y*y - z*z`)
     for rigid-body member coordinates.
   - `quat_to_rot_eigen9`: Eigen `toRotationMatrix` form (`1 - 2(y*y + z*z)`)
     as a 9-tuple of arrays for Gaussian covariance rotations.
2. Covariance conjugation $R C R^T$ (`rot_cov6`) in closed form on the 6
   upper-triangle components.
3. Gaussian overlap (`overlap_cov6`) via explicit $3 \times 3$ cofactors,
   determinant, and quadratic form on the 6 upper-triangle components.
4. Rigid-body point rotation $R \cdot x_{\text{local}} + t$ (`rotate_cf`) via
   elementwise products and sums rather than `jnp.einsum` / `lax.dot_general`.

Avoiding `dot_general` prevents XLA GPU from routing `float32` $3 \times 3$
products through TensorFloat-32 (TF32, 10-bit mantissa; `salilab/imp#1129`),
and passing 9-tuples / 6-tuples avoids intermediate `(3, 3, ...)` stack/slice
fusion barriers in XLA.

- **Specialization level**: **Level 1 (Core geometry math)** in
  [`jax_mc/terms/geom.py`](../jax_mc/terms/geom.py) and tuple-unpacked helpers
  in [`recipes/r5_cov6_elementwise.py`](r5_cov6_elementwise.py).

## Preconditions (enforced by `assert`)
1. Symmetric $3 \times 3$ covariances are 6-component upper-triangle tuples/arrays
   `(c00, c01, c02, c11, c12, c22)` with positive diagonal entries
   (`assert len(c6) == 6`, `assert np.all(t.cloc[[0, 3, 5]] > 0.)`).
2. Quaternions have 4 components `(w, x, y, z)` on axis 0
   (`assert q.shape[0] == 4`).

## Measured gain
- **RNA Polymerase II (`polii`) on 1 NVIDIA L40 GPU (`float32`)**:
  - Eliminates TF32 precision loss on GPU, reducing single-state total energy
    error `|f32 - f64|` from `2.1 kT` to `2.6e-4 kT` (`8,000×` more accurate).
  - Unpacking `R9` and `c6` into 9-tuples and 6-tuples reduces register
    pressure and shared-memory spills in the rigid-body EM kernels (`+3%` to
    `+5%` speedup on GPU and CPU).

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing Metropolis decisions against
  `Engine(s)` in `float64`).
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r5_cov6_elementwise:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r5_cov6_elementwise:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r5_cov6_elementwise.py`](r5_cov6_elementwise.py):

```python
def rot_cov6(R9, c6):
    assert len(R9) == 9 and len(c6) == 6
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
    assert len(ca6) == 6 and len(cb6) == 6
    a, b, c = ca6[0] + cb6[0], ca6[1] + cb6[1], ca6[2] + cb6[2]
    d, e, f = ca6[3] + cb6[3], ca6[4] + cb6[4], ca6[5] + cb6[5]
    c00, c01, c02 = d * f - e * e, c * e - b * f, b * e - c * d
    det = a * c00 + b * c01 + c * c02
    c11, c12, c22 = a * f - c * c, b * c - a * e, a * d - b * b
    vx, vy, vz = xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]
    quad = (c00 * vx * vx + c11 * vy * vy + c22 * vz * vz
            + 2. * (c01 * vx * vy + c02 * vx * vz + c12 * vy * vz)) / det
    return ma * mb * GAUSS_NORM / xp.sqrt(det) * xp.exp(-0.5 * quad)
```
