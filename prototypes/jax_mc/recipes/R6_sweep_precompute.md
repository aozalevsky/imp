# R6: Sweep-wide precompute of bead proposals' static terms

## What it does
Because each flexible bead is moved at most once per sweep by a `BallMover` (and
no other mover modifies bead coordinates), every bead's old position at the
moment its mover runs equals its position `X0[:, bead_off:bead_off + nbead]` at
the start of the sweep, and its proposed position `xn_all` depends only on `X0`
and the sweep's random numbers `(nrm, uni)`.

`precompute_bead_static_ev` computes all `nbead` proposed bead positions and
evaluates their static EV grid energies `es_n_all` of shape `(nbead, C)` in a
single batched `(nbead, C, K)` kernel at the start of the sweep (behind
`jax.lax.optimization_barrier`), removing all 3D static-grid index arithmetic
and table gathers from the sequential bead steps.

- **Specialization level**: **Level 3 (Sweep-level precompute)** in
  [`recipes/r6_sweep_precompute.py`](r6_sweep_precompute.py) and
  [`FewBodiesBeads`](../templates/few_bodies_beads/sampler.py) (for
  `C <= 32,768` chains where `es_n_all` fits in L2 cache).

## Preconditions (enforced by `assert`)
1. Non-empty static particle set (`assert len(s.static_rows) > 0`).
2. Exactly one `BallMover` per bead per sweep, in contiguous bead row order
   `[bead_off, bead_off + nbead)`
   (`assert len(bead_steps) == s.nbead`,
   `assert np.array_equal(bead_rows, np.arange(bead_off, bead_off + s.nbead))`).
3. No non-ball mover touches any bead row
   (`assert not len(mv.beads)` for `mv.kind != 'ball'`).

## Measured gain
- **RNA Polymerase II (`polii`)**:
  - Replaces `47` separate `(1, C, 21)` static EV grid lookups per sweep with
    one batched `(47, C, 21)` lookup.
  - On **1 NVIDIA L40 GPU (`float32`)**: improves throughput by `+6%` to `+11%`
    for `C <= 32,768` chains by keeping register pressure low inside the
    blocked bead kernels.
  - On **1 CPU core (`float64`, 1 chain)**: part of the `+23%` speedup from
    R1 + R4–R7 (`49.9k -> 61.4k` MC steps/s) by vectorizing the grid lookup
    across all 47 beads.

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing Metropolis decisions against
  `Engine(s)` in `float64` across `<= 2,048` and `> 4,096` chains at `kT = 1`
  and `kT = 10`).
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r6_sweep_precompute:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r6_sweep_precompute:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r6_sweep_precompute.py`](r6_sweep_precompute.py):

```python
def precompute_bead_static_ev(X0, bead_off, nbead, bead_steps, bead_radii,
                              r_beads, nrm, uni, grid):
    dt = X0.dtype
    xo_all = X0[:, bead_off:bead_off + nbead]
    db = jnp.moveaxis(nrm[bead_steps, :3], 1, 0)
    ub = uni[bead_steps, 0]
    norm_db = jnp.sqrt(
        db[0] * db[0] + db[1] * db[1] + db[2] * db[2])
    step_b = jnp.asarray(bead_radii, dt)[:, None] * jnp.cbrt(ub)
    xn_all = xo_all + (db / norm_db[None]) * step_b[None]
    return jax.lax.optimization_barrier(grid_ev(grid, xn_all, r_beads, dt))
```
