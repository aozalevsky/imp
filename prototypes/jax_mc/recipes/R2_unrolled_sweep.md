# R2: Unrolled sweep, in-place `dynamic_update_slice` behind `optimization_barrier`, and flat tuple carry

## What it does
Replaces the generic `lax.scan` + `lax.switch` mover dispatch and dynamic
index scatter (`X.at[:, rows].set(...)`) with:
1. A Python-unrolled sweep over `s.movers` where each mover accesses its own
   contiguous row slice `[r0, r0 + nr)` in `X`.
2. In-place `jax.lax.dynamic_update_slice_in_dim` with the slice offset `r0`
   hidden behind `jax.lax.optimization_barrier(jnp.int32(r0))` (`barrier_slice`,
   `barrier_update`, `barrier_select_update`). Without the barrier, XLA's
   `HloSimplifier` converts `dynamic-slice` with a constant offset into a
   static `kSlice`, which causes XLA GPU's
   `CanEmitFusedDynamicUpdateSliceInPlace` check to reject in-place buffer
   aliasing and emit a full-buffer `kCopy` of `X` before every mover step.
3. A flat tuple loop carry in `jax.lax.fori_loop` where per-body arrays
   (`quat`, `trans`, `es_body`), per-bead caches (`es_bead`), and acceptance
   counters (`n_acc`) are unpacked into individual `(..., C)` tuple elements,
   eliminating all intermediate `dynamic-update-slice` kernels on state tables
   inside the sweep.

- **Specialization level**: **Level 3 (Sweep override / custom state carry)**
  in [`UnrolledEngine(Engine)`](r2_unrolled_sweep.py) and
  [`FewBodiesBeads`](../templates/few_bodies_beads/sampler.py).

## Preconditions (enforced by `assert`)
1. `s.mode == 'serial'` (`assert s.mode == 'serial'`).
2. Few movers so compile time stays short (`assert 0 < len(s.movers) <= max_movers`,
   49 movers in Pol II).
3. Every coordinate mover moves a contiguous slice of rows
   (`assert len(r) > 0 and np.array_equal(r, np.arange(r[0], r[0] + len(r)))`).

## Measured gain
- **RNA Polymerase II (`polii`) on 1 NVIDIA L40 GPU (`float32`)**:
  - Removes `~100–280` full-buffer `kCopy` kernels per sweep (`0` copies of
    `X` in the compiled HLO loop, verified with `jax_mc.bench --hlo`).
  - Unpacking the loop carry into a flat tuple eliminates `98` (`2 × 49`)
    per-step DUS launches on `es_bead` and `n_acc` per sweep (`+13%` at
    `8,192` chains and `+16%` at `16,384` chains).

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing decisions against `Engine(s)` in
  `float64`, and bitwise-identical proposals and energies).
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r2_unrolled_sweep:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r2_unrolled_sweep:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r2_unrolled_sweep.py`](r2_unrolled_sweep.py):

```python
def barrier_slice(X, start, length, axis=1):
    od = jax.lax.optimization_barrier(jnp.int32(start))
    return jax.lax.dynamic_slice_in_dim(X, od, int(length), axis=axis)


def barrier_update(X, new_slice, start, axis=1):
    od = jax.lax.optimization_barrier(jnp.int32(start))
    return jax.lax.dynamic_update_slice_in_dim(X, new_slice, od, axis=axis)


def barrier_select_update(X, acc, Xn, start, length, axis=1):
    od = jax.lax.optimization_barrier(jnp.int32(start))
    Xo = jax.lax.dynamic_slice_in_dim(X, od, int(length), axis=axis)
    return jax.lax.dynamic_update_slice_in_dim(
        X, jnp.where(acc, Xn, Xo), od, axis=axis)
```
