# Recipes (R1–R7): From the reference engine to a specialized sampler

Every recipe in `recipes/` is a self-contained, testable optimization step
away from `jax_mc.engine.Engine` with:
- explicit preconditions enforced by `assert`;
- a `factory(s)` entry point for `python -m jax_mc.check decisions`;
- a documentation page (`R1_*.md` .. `R7_*.md`) stating what it does, its
  specialization level (1: Term subclass, 2: Engine branch/draws, 3: custom
  sweep), its preconditions, its measured gain, its exactness level, and its
  reference function.

## Summary of Recipes R1–R7

| Recipe | Page & Module | Level | What it does | Key preconditions (`assert`) |
|---|---|---|---|---|
| **R1** | [R1_static_ev_grid.md](R1_static_ev_grid.md) / [`r1_static_ev_grid.py`](r1_static_ev_grid.py) | 1 & 3 | Static/movable split; 3D cell grid (`uint16` table + packed `float4` `(x,y,z,r)`) for static EV | `0 < n_static < n`; `n_static <= 65535`; disjoint static/moving exclusion labels |
| **R2** | [R2_unrolled_sweep.md](R2_unrolled_sweep.md) / [`r2_unrolled_sweep.py`](r2_unrolled_sweep.py) | 3 | Unrolled `SerialMover` sweep; in-place `dynamic_update_slice_in_dim` behind `optimization_barrier`; flat tuple carry | `s.mode == 'serial'`; `<= 128` movers; contiguous rows per mover |
| **R3** | [R3_blocked_beads.md](R3_blocked_beads.md) / [`r3_blocked_beads.py`](r3_blocked_beads.py) | 3 | Blocked bead steps (`_bead_block_batched` for `C <= 4,096`, `_bead_block_chained` for `C > 4,096`) with within-block corrections | 1 `BallMover` per bead in contiguous row order; no EM on beads; `cmax == 1` on bead cross-links |
| **R4** | [R4_em_body_cache.md](R4_em_body_cache.md) / [`r4_em_body_cache.py`](r4_em_body_cache.py) | 1 & 3 | Per-body EM cache (MD sums, inter-body sums, and invariant intra-body overlap computed once) | `s.cov_mode == 'rotate'`; all model EM Gaussians on moving rigid bodies; static density; any `nb >= 1` |
| **R5** | [R5_cov6_elementwise.md](R5_cov6_elementwise.md) / [`r5_cov6_elementwise.py`](r5_cov6_elementwise.py) | 1 & 3 | Closed-form 6-component covariance rotation/overlap and elementwise 3×3 rotations (no `dot_general` / TF32) | 6-component symmetric covariances with positive diagonals; 4-component quaternions |
| **R6** | [R6_sweep_precompute.md](R6_sweep_precompute.md) / [`r6_sweep_precompute.py`](r6_sweep_precompute.py) | 3 | Sweep-wide precompute of all bead proposals' static EV grid lookups (`es_n_all`) from `X0` | 1 `BallMover` per bead per sweep in contiguous row order; non-empty static set |
| **R7** | [R7_sliced_rng.md](R7_sliced_rng.md) / [`r7_sliced_rng.py`](r7_sliced_rng.py) | 2 & 3 | Contract-preserving sliced RNG draws (`sliced_draws` slices unused rotation columns before chain-last transpose) | Movers in `{'ball', 'rb', 'transform', 'normal'}`; follows `engine.draws` contract |

## Running All Exactness Checks

Run [`recipes/check_all.py`](check_all.py) on a GPU node to verify Level D
(`0` differing Metropolis decisions against `jax_mc.engine.Engine` in `float64`
at both `<= 2,048` and `> 4,096` chains, `kT = 1` and `kT = 10`) for every
recipe R1–R7 and [`templates.few_bodies_beads.sampler:factory`](../templates/few_bodies_beads/sampler.py):

```bash
python -m recipes.check_all
```

It also runs the template at 4,096, 16,384, 32,768 and 65,536 chains, where
its code changes path. `tests/check_all.sh` calls it. R5 has no check of its
own: its functions are checked through the template.
