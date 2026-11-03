# R4: Per-body EM cache and invariant intra-body overlap (`rotate` mode)

## What it does
Replaces the running-total additive/subtractive EM cache (`cache + new - old`)
with per-rigid-body cached components:
1. **Per-body model–data overlap sums** `mds[b]` and **distance sums** `sl[b]`
   for `b in 0 .. nb - 1`.
2. **Per-body-pair inter-body model–model overlaps** `inter[b1, b2]` for
   `0 <= b1 < b2 < nb`.
3. **Invariant intra-body model–model overlap (`intra`)**: when model Gaussian
   covariances rotate with their rigid body (`s.cov_mode == 'rotate'`), a rigid
   transformation $(R, t)$ rotates both the inter-center displacement
   $x_i - x_j \mapsto R(x_i - x_j)$ and the combined covariance
   $\Sigma_i + \Sigma_j \mapsto R(\Sigma_i + \Sigma_j)R^T$. Consequently, both
   the sphere-distance cutoff and the Gaussian overlap quadratic form
   $(x_i - x_j)^T (\Sigma_i + \Sigma_j)^{-1} (x_i - x_j)$ are strictly
   invariant under rigid-body moves. R4 computes `intra` once outside the sweep
   loop and never recomputes intra-body pairs during rigid-body steps.
4. **No floating-point subtraction in the cache**: when body `b` moves, row `b`
   of `mds` and `sl` and the inter-body pairs touching `b` are replaced and
   summed across bodies (`mds1 + mds2`), avoiding `cache + new - old`
   cancellation drift in `float32`.

- **Specialization level**:
  - **Level 1 (Term subclass)**: [`BodyCacheEMIsd(EMIsd)`](r4_em_body_cache.py)
    supports any number of rigid bodies `nb >= 1` in `Engine(s)`.
  - **Level 3 (Template)**: `_em_parts` and `_em_rb` in
    [`FewBodiesBeads`](../templates/few_bodies_beads/sampler.py).

## Preconditions (enforced by `assert`)
1. `s.cov_mode == 'rotate'` for the invariant intra-body overlap cache
   (`assert s.cov_mode == 'rotate'`). Under `frozen` covariances, rigid-body
   rotations change $(x_i - x_j)$ without rotating $\Sigma_i + \Sigma_j$, so
   intra-body overlaps are not invariant.
2. All model EM Gaussians belong to moving rigid bodies (`assert np.all(s.row_kind[emt.rows] == S.MEMBER)` and `assert np.all(emt.owner >= 0)`), with no EM Gaussians on flexible beads.
3. Density Gaussians are static (`assert emt.dx0 is not None`).
4. No `TransformMover` moves multiple bodies simultaneously
   (`assert not any(mv.kind == 'transform' for mv in s.movers)`).
5. Supports any number of rigid bodies (`assert s.nb >= 1`).

## Measured gain
- **RNA Polymerase II (`polii`, 10 model Gaussians = 4 on Rpb4 + 6 on Rpb7,
  50 density Gaussians)**:
  - Reduces model–data Gaussian pair evaluations per rigid-body step from
    `10 × 50 = 500` to `4 × 50 = 200` (Rpb4) or `6 × 50 = 300` (Rpb7) (`2×`
    fewer MD pairs), and model–model pairs from `45` upper-triangle pairs to
    `4 × 6 = 24` inter-body pairs (`0` intra-body pairs per step in `rotate`
    mode).
  - On **1 CPU core (`float64`, 1 chain)**: combined with R1 and R5–R7,
    improves throughput by `+23%` (`49.9k -> 61.4k` MC steps/s).
  - On **1 NVIDIA L40 GPU (`float32`)**: contributes `+5%` to `+7%` total sweep
    throughput by shrinking the 2 rigid-body step kernels.

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing Metropolis decisions against
  `Engine(s)` in `float64` across `<= 2,048` and `> 4,096` chains at `kT = 1`
  and `kT = 10` with `--cov rotate`).
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r4_em_body_cache:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r4_em_body_cache:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r4_em_body_cache.py`](r4_em_body_cache.py):

```python
def check_preconditions(s, emt, require_rotate=True):
    if require_rotate:
        assert s.cov_mode == 'rotate', (
            f'R4 requires cov_mode == "rotate" for invariant intra-body '
            f'overlaps, got "{s.cov_mode}"')
    assert np.all(s.row_kind[emt.rows] == S.MEMBER)
    if s.cov_mode == 'rotate':
        assert np.all(emt.owner >= 0)
    assert emt.dx0 is not None
    assert not any(mv.kind == 'transform' for mv in s.movers)
    assert s.nb >= 1
```
