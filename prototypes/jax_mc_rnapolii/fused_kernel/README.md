# A whole sweep as one GPU kernel (proof of concept)

**An experiment, not meant to be maintained.** `TutSamplerSoA` runs a sweep
of the tutorial's 49 moves as about 340 XLA kernels. Here the whole sweep is
one hand-written kernel, in Pallas, JAX's kernel language, which compiles
through Triton on NVIDIA GPUs. It shows how much faster the same sampler can
get. The kernel itself is about 350 lines (`_sweep_kernel` in
`pallas_sweep.py`), written for this system's layout, and Pallas's GPU
lowering is experimental and changes between JAX versions.

## Result

Complete tutorial runs on four L40s: 16,384 chains per GPU, 20,000 frames,
DOF snapshots every 200 frames, scores every 10. MC steps/s including the
output (IMP nightly 20261022, JAX 0.11.2):

| | sampling time | MC steps/s |
|---|---|---|
| `run_soa.py` (XLA) | 311 s | 2.07 billion |
| fused kernel, `--no-em-cache` | 177 s | 3.63 billion |
| fused kernel, `--no-em-cache --rescore recorded` | 166 s | 3.87 billion |
| fused kernel, `--rescore recorded` (EM cache on) | 153 s | **4.19 billion** |

* The first two rows have the same settings: the kernel is 1.76 times as
  fast. `run_soa.py`'s complete runs with this output at 32,768 chains per
  GPU reach 2.58 billion.
* The last row ran on a second node of the same type. There, the row
  above it gave 3.73 billion, so the EM cache adds 12%.
* `run_soa.py --rescore recorded` does the full rescore only on the frames
  whose scores are recorded (every 10th here); the others keep the
  sampler's tracked energies. The median gap between tracked and rescored
  energy is then 4e-4 (1.2e-4 with a rescore every frame). The largest gap,
  1.4-1.7 (1.1-1.2), is at the first rescore after the start; after that
  no gap exceeds 0.25.
* The EM cache: a rigid-body move recomputes only the moved body's
  model-data sums and the model-model pairs across the two bodies. The
  other body's sums are kept per chain, and pairs within one body do not
  change when it moves rigidly.

**Same results as `TutSamplerSoA`:**

* One sweep of 4,096 chains, from the same state and random numbers, with
  and without the EM cache (`test_fused.py --check`):
  * all 192,512 bead moves and 8,192 rigid-body moves make the same
    accept/reject decision;
  * positions, quaternions and translations are bit-identical;
  * EM agrees to 6e-5, total energies to 6e-2 (float32 sums in another
    order, on totals of ~2.6e5).
* Complete runs with the same seeds follow the same trajectories at first:
  96% of chains of the fused and the XLA run are within 0.5 of each other's
  score at frame 1,999. At nine checkpoints from frame 9 to 19,999, the
  mean scores differ by at most 0.22 standard errors (the runs share seeds,
  so they are correlated).
* Each run gives the same number of good-scoring frames (score at most
  591.4, the best 0.1% of a larger run) to within 1%. The same holds
  between the runs with and without the EM cache and `--rescore recorded`.
* RMF files made from the snapshots, rescored by C++ (2 chains x 3
  frames per run): within 2.9e-4, with EM scored on the pairs within its
  cutoff (salilab/imp#1121).

**Sampler only**, one L40, MC steps/s (`test_fused.py --bench
--no-em-cache`):

| chains | XLA (`TutSamplerSoA`) | fused kernel | speedup |
|---|---|---|---|
| 1,024 | 64.5M | 161M | 2.5x |
| 4,096 | 183M | 609M | 3.3x |
| 16,384 | 530M | 1,072M | 2.0x |
| 32,768 | 654M | 976M | 1.5x |
| 65,536 | 607M | 990M | 1.6x |

* The XLA column runs on the kernel's state, padded to 128 rows. Without
  the padding XLA did 706M at 32,768 chains, so the speedup there is 1.4x.
* Runs of 1,000-4,000 chains gain the most: XLA was limited there by its
  kernel launches.
* From 16,384 chains on, the kernel's time grows linearly with the chains,
  45-50 ns per chain per sweep.

## Where a frame's time goes

16,384 chains, one L40, ms per frame of 10 sweeps, without the EM cache
and with a rescore every frame (`breakdown.py --no-em-cache`):

| part | ms |
|---|---|
| 47 bead moves | 3.24 |
| 2 rigid-body moves, without EM | 2.36 |
| EM term in those moves | 1.57 |
| full rescore | 0.64 |
| random numbers (XLA) | 0.35 |
| recording scores, drift and the 10 best | 0.22 |

The two rigid-body moves cost more than all 47 bead moves.

## How the kernel works (`pallas_sweep.py`, `TutSamplerFused`)

* One `pallas_call` per sweep. Each GPU program takes 16 chains (2 warps).
* The dynamic rows (105 here, padded to 128) stay in registers for the
  dense EV sums and are updated after each accepted bead move.
* The moved bead and its bond and cross-link partners are read from the
  state in memory; every move writes its result back, followed by a
  barrier. Static partners are a padding row (position 0) plus their
  position from a table.
* Bead moves: one loop body over the moves, with per-move constants from
  small tables. The static EV uses `TutSamplerSoA`'s grid, padded to 32
  candidates per cell.
* Rigid-body moves: the proposal, the proposed rows into a scratch buffer,
  then dense EV, static EV, the body's cross-links and the EM term; the
  scratch rows are copied into the state if the move is accepted.
* The random numbers are still generated by XLA before each sweep, with
  the same stream as `TutSamplerSoA`, so the two can be compared move by
  move.

What did not help:

* compiling every move separately with literal constants: an earlier
  version for the bead moves alone (`pallas_sampler.py`, mode `unroll`)
  had not finished compiling after 11 minutes. With a loop over the moves
  it compiled in seconds; the whole-sweep kernel compiles in about 36 s;
* keeping fewer rows in registers, or reading them all from memory: both
  slower, so occupancy is not the limit.

## Files and running

| file | contents |
|---|---|
| `pallas_sweep.py` | `TutSamplerFused`: the whole sweep as one kernel |
| `pallas_sampler.py` | `TutSamplerPallas`: the bead moves as kernels (rigid-body moves in XLA); the base class, with the move tables |
| `run_soa_fused.py` | `run_soa.py` with `TutSamplerFused` (same options and outputs, plus `--no-em-cache`) |
| `test_fused.py` | one sweep against `TutSamplerSoA` from the same state (`--check`), and timings (`--bench`) |
| `breakdown.py` | the time per frame by part |

```bash
python test_fused.py --check --chains 4096
python test_fused.py --bench --chains 4096 16384
mpirun -np 4 --bind-to none python run_soa_fused.py --chains 16384 \
    --frames 20000 --thin 200 --score-every 10 --rescore recorded --out runs/fused
python ../dofs_to_rmf.py runs/fused --best 1000 --procs 8
```

`--bind-to none` keeps Open MPI from pinning each rank, and so its
snapshot thread, to a few cores; the runs above used it.

## Limits

* Written for this tutorial: two rigid bodies, at most 127 dynamic rows
  and 96 EV rows, at most 32 EV rows and 32 other rows per body, 32 grid
  candidates per cell, 64 data Gaussians (50 used), EM covariances that
  rotate with the bodies (`--cov rotate`), dense EV against the bodies (no
  per-body grids). Triton needs power-of-2 shapes, hence the padding.
* The per-frame rescore, records and snapshots stay in XLA.
* NVIDIA GPUs only.
