# Few-bodies-and-beads template

**Fits:** `SerialMover` systems with a few moving rigid bodies, many flexible
beads moved by `BallMover`s in contiguous bead order, a large static particle
background, and soft-sphere excluded volume, harmonic upper-bound connectivity,
unambiguous cross-links, and an ISD Gaussian EM restraint. Example: the RNA
Polymerase II stalk tutorial (`polii`: 2 moving rigid bodies, 47 flexible
beads, 49 movers per sweep, 369 static EV particles, 10 model × 50 density EM
Gaussians).

**Asserted limits worth knowing before you copy it:** exactly 2 moving
rigid bodies (the rigid-body step's EM cache pairs each body with the other
one, `b2 = 1 - b` in `_em_rb`; R4's term subclass handles any number); no NormalMovers; one
SoftSphereEV, one Pairs (harmonic upper bound) and one EMIsd term, plus
cross-links and nuisance-only priors. Anything else raises `Unsupported`
or an assert in `FewBodiesBeads.__init__`.

**Tuned for one L40.** The sweep switches code paths by chain count: bead
block sizes 12 / 34 / 9 / 6 / 4 up to 2,048 / 4,096 / 16,384 / 32,768 / more
chains, sweep-entry positions reused up to 8,192, the R6 precompute up to
32,768 (`_es_thresh`), barriers at 2,048 and 16,384. Every path is checked
(`python -m recipes.check_all`); re-tune the numbers in `_sweep_unpacked`
for another GPU and re-run the check.

**Starts from:** `sampler_soa.py` (`TutSamplerSoA`), moved onto `jax_mc`'s
`System`, `terms`, and `engine` random-number contract (`chain_keys`, `draws`).

**Recipes** (preconditions asserted in `FewBodiesBeads.__init__`):

| recipe | what it does | precondition |
|---|---|---|
| R1 | Static/movable split; static EV scored via a precomputed 3D cell grid (`uint16` cell table, packed `(x, y, z, r)` point table); static-vs-dynamic EV cached per bead (`es_bead`) and per rigid body (`es_body`) | Non-empty static EV set (`n_se > 0`); `n_se <= 65535` (`uint16`); disjoint exclusion labels between static and moving EV rows |
| R2 | Unrolled sweep in `SerialMover` order with static row offsets, in-place `dynamic_update_slice_in_dim` behind `optimization_barrier`, and a flat tuple loop carry (`0` copies of `X` in the compiled loop) | `s.mode == 'serial'`; contiguous rows per mover |
| R3 | Blocked bead steps (`_bead_block_batched` for `C <= 4,096`, `_bead_block_chained` for `C > 4,096`) with exact within-block EV, connectivity, and cross-link corrections (enabled on GPU, disabled on CPU) | One `BallMover` per bead per sweep in contiguous bead row order; no EM Gaussians on beads; unambiguous cross-links (`cmax == 1`) |
| R4 | Per-body EM cache (`_em_parts`, `_em_rb`): caches per-body model-data overlap and distance sums; under `cov_mode == 'rotate'`, intra-body model-model overlap is invariant under rigid-body moves and is computed once outside the sweep loop | All model EM Gaussians on moving rigid bodies; static density Gaussians (`dx0 is not None`); `cov_mode == 'rotate'` for the invariant intra-body overlap cache (falls back to per-step intra-body recompute under `frozen`) |
| R5 | Closed-form 6-component symmetric covariance rotations (`rot_cov6`), overlap (`overlap_cov6`), and elementwise 3×3 rotations (`rotate_cf`), avoiding `dot_general` / TF32 | 6-component symmetric covariances with positive diagonals; 4-component unit quaternions |
| R6 | Sweep-wide precomputation of bead proposals' static EV grid lookups (`es_n_all`) from the sweep-entry positions `X0` before the mover loop | One `BallMover` per bead per sweep in contiguous bead row order; non-empty static EV set |
| R7 | Contract-preserving sliced RNG draws (`sliced_draws`): generates per-chain draws matching `engine.draws` bit-for-bit, slicing out unused rotation columns before the chain-last transpose | Movers in `{'ball', 'rb'}` following the `jax_mc` random-number contract |

**Check** against the reference engine, in float64 (both chain-count regimes):

```bash
python -m jax_mc.check decisions polii \
    --sampler templates.few_bodies_beads.sampler:factory \
    --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

python -m jax_mc.check decisions polii \
    --sampler templates.few_bodies_beads.sampler:factory \
    --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
```

**Benchmark** (`polii`, 1 NVIDIA L40 GPU, `float32`, 200 sweeps):

```bash
python -m jax_mc.bench polii \
    --sampler templates.few_bodies_beads.sampler:factory \
    --chains 1024,4096,8192,16384,32768,65536 --sweeps 200 --hlo
```

| chains | `--cov rotate` (MC steps/s) | `--cov frozen` (MC steps/s) |
|---:|---:|---:|
| 1,024 | 201.44 M | 205.90 M |
| 4,096 | 591.24 M | 587.65 M |
| 8,192 | 742.02 M | 759.96 M |
| 16,384 | 1,158.79 M (1.16 bn) | 1,161.32 M (1.16 bn) |
| 32,768 | **1,200.38 M (1.20 bn)** | 1,185.65 M (1.19 bn) |
| 65,536 | 1,150.22 M (1.15 bn) | **1,202.79 M (1.20 bn)** |
