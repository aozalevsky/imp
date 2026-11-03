# R7: Contract-preserving sliced RNG

## What it does
Follows the `jax_mc` random-number contract (`engine.draws`), which draws
`U = uniform(k, (n_movers, 3))` and `Z = normal(fold_in(k, 1), (n_movers, 6))`
per chain `c` and sweep `s`, but slices out the unused rotation columns
(`Z[:, 3:6]` and `U[:, 1]` for `BallMover`s, which neither rotate nor use
`U[:, 1]`) **inside** the per-chain function before `jnp.moveaxis(..., 0, -1)`
transposes the draws to chain-last layout `(..., C)` (`sliced_draws`).

For RNA Polymerase II (`47` `BallMover`s and `2` `RigidBodyMover`s), `BallMover`s
use only `3` normals (`Z[:, 0:3]`) and `2` uniforms (`U[:, 0]` for `cbrt` and
`U[:, 2]` for Metropolis acceptance), while only the `2` `RigidBodyMover`s use
the rotation axis `Z[rot_movers, 3:6]` and rotation angle `U[rot_movers, 1]`.
Slicing inside `vmap` before the chain-last transpose cuts the transposed RNG
buffer volume from `49 × (6 + 3) = 441` floats/chain/sweep to
`49 × (3 + 2) + 2 × (3 + 1) = 253` floats/chain/sweep (`42.6%` fewer RNG words
materialized in global memory; `48%` fewer normal words).

- **Specialization level**: **Level 2 / Level 3** in
  [`SlicedRNGEngine(Engine)`](r7_sliced_rng.py) and
  [`FewBodiesBeads`](../templates/few_bodies_beads/sampler.py).

## Preconditions (enforced by `assert`)
1. All movers in `s.movers` belong to `{'ball', 'rb', 'transform', 'normal'}`
   (`assert kinds <= {'ball', 'rb', 'transform', 'normal'}`).
2. Per-chain keys and sweep indices follow the `jax_mc` random-number contract
   (`chain_keys`, `fold_in(key_c, sweep)`).

## Measured gain
- **RNA Polymerase II (`polii`)**:
  - Reduces global-memory RNG writes per sweep by `42.6%` (`441 -> 253`
    floats/chain/sweep).
  - On **1 NVIDIA L40 GPU (`float32`, 65,536 chains)**: saves `49.3 MB` of
    global-memory traffic per sweep and improves high-chain-count throughput by
    `+3%` to `+4%`.

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing Metropolis decisions against
  `Engine(s)` in `float64`; every draw consumed is bitwise-identical to
  `engine.draws`).
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r7_sliced_rng:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r7_sliced_rng:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r7_sliced_rng.py`](r7_sliced_rng.py):

```python
def sliced_draws(key, sweep, n_movers, rot_movers, dtype):
    rot_idx = np.asarray(rot_movers, np.int32)

    def one(k):
        k = jax.random.fold_in(k, sweep)
        u = jax.random.uniform(k, (n_movers, 3), dtype)
        z = jax.random.normal(jax.random.fold_in(k, 1), (n_movers, 6), dtype)
        u02 = jnp.stack([u[:, 0], u[:, 2]], axis=1)
        return z[:, :3], u02, z[rot_idx, 3:6], u[rot_idx, 1]

    nrm3, uni02, rb_ax, rb_u1 = jax.vmap(one)(key)
    return (jnp.moveaxis(nrm3, 0, -1),
            jnp.moveaxis(uni02, 0, -1),
            jnp.moveaxis(rb_ax, 0, -1),
            jnp.moveaxis(rb_u1, 0, -1))
```
