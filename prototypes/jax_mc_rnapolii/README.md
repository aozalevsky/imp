# Prototype: an exact, batched JAX Monte Carlo sampler for the RNA Pol II tutorial

**Status: a crude prototype for discussion, not proposed for merging.** It
runs the sampling of the IMP tutorial
([salilab/imp_tutorial](https://github.com/salilab/imp_tutorial),
`rnapolii/modeling/modeling.py`) for many independent chains on a GPU. It is
written for this one system; the parts that could be generalized into IMP
are listed under [Toward IMP](#toward-imp).

**"Exact" here means:**
* the same scoring function and the same moves as the C++ run;
* every term agrees with C++ to float64 round-off (see
  [Validation](#exactness-and-validation)).

There are three known differences:
* model Gaussian covariances follow the rigid bodies, while C++ uses a stale
  cache (salilab/imp#1119). This is a deliberate choice, and `--cov frozen`
  reproduces C++;
* the EM restraint uses the pairs within its cutoff, while C++ uses
  history-dependent close-pair lists (salilab/imp#1121);
* sampling runs in float32 by default.

![RNA Pol II tutorial: Monte Carlo throughput on one GPU node](speedup.png)

## What it does

* **Builds the system with the tutorial's own `modeling.py`.**
  * The script is run in-process with `mc1.execute_macro()` removed and
    `--test` passed. That only shortens the tutorial's frame count, which
    is not used here.
  * Restraint parameters are read from the C++ objects. For the EM
    restraint, they are recorded from its constructor arguments, because
    there are no getters (`build_system.py`, `scoring.py`).
* **Runs the same Monte Carlo as PMI:**
  * the `SerialMover` over `dof.get_movers()` (47 `BallMover`s on flexible
    beads, 2 `RigidBodyMover`s on the stalk), one mover per MC step in the
    same order;
  * Metropolis acceptance;
  * PMI's simulated-annealing schedule;
  * 490 MC steps per frame.
* **Many chains at once:** each chain is an independent copy of the
  tutorial's run, and chains are batched with `jax.vmap`. All chains start
  from the tutorial's shuffled model (`build(seed=1)`) and differ in their
  random numbers. `initial_state()` also accepts per-chain starts.
* **Incremental energies.** A step scores only the terms that involve the
  moved particles (`sampler.py`):
  * a bead step: its excluded volume against every other particle, its
    bonds and its cross-links;
  * a rigid-body step: the body's excluded volume, its cross-links and the
    EM restraint.

  A full rescore every frame removes float32 drift.
* **Output:** one RMF file per chain, written by background processes while
  the GPU keeps sampling; per-frame scores and terms; the 10 best-scoring
  models (`run_chains.py`).
* **Several GPUs through `IMP.mpi`**, one rank per GPU (`run_mpi.py`):
  * independent runs;
  * or PMI-style replica exchange, where neighbouring ranks swap chain
    configurations with IMP.mpi's acceptance rule.

## Speed

All speeds are MC steps per second, as IMP counts them (one mover proposal
plus one Metropolis test). One complete tutorial run is 20,000 frames ×
490 = 9.8 million MC steps.

All on one GPU node with four L40s. Software: IMP nightly 20261014, JAX
0.11.2 (CUDA 12). The 356M row is from this commit with nightly 20261019.
All JAX speeds are float32.

| | MC steps/s |
|---|---|
| C++ IMP nightly, one core, as the tutorial runs (`score_moved` off) | 5,157 |
| C++ `imp-fast` nightly, `score_moved=True`, one core (the fastest exact C++) | 21,800 |
| JAX, one CPU core, one chain | 326,000 |
| JAX, one CPU core, 64 chains | 1.0 million |
| JAX, one L40, 32,768 chains, sampler only (`bench_chains.py`) | 356 million |
| JAX, `run_mpi.py` with a rescore every frame, 16,384 chains per GPU: one L40 | 249 million |
| same, four L40s on one node | 1.0 billion |

* **For scale** (an estimate, not measured): a whole 64-core node running
  the fastest C++ would do about 64 × 21,800 ≈ 1.4 million MC steps/s. One
  L40 is about 250 times that.
* **End to end:**
  * 1,024 complete tutorial runs take 3.5 min on one L40, with an RMF frame
    every 200 frames;
  * 16,384 runs take 5.9 min on four L40s, with an RMF frame every 2,000
    frames.

  The C++ tutorial writes RMF, stat files and best-model PDBs every frame.
* **Per chain**, a GPU runs about 40–60k MC steps/s up to ~1,000 chains,
  limited by kernel-launch latency. The throughput comes from running many
  chains at once.

## Exactness and validation

* **Terms** (`validate_terms.py`, float64, at the shuffled start and at
  configurations along a C++ MC trajectory; IMP nightly 20261019):

  | term | max abs difference from C++ |
  |---|---|
  | connectivity | 6e-11 |
  | excluded volume | 5e-13 |
  | EM, with C++'s own close-pair lists | 1e-13 |
  | cross-links | 1.2e-6 (0 for the numpy reference that emulates C++'s `float` arithmetic in `sphere_cap`) |

  With `--cov rotate`, EM is compared with C++ after
  `update_global_covariance()` on the model Gaussians, which is what
  salilab/imp#1118 would do.
* **Sampler** (`check_sampler.py`, nightly 20261019): the incrementally
  tracked energy equals a full recompute to 1e-9 in float64 (energies
  ~3e5).
* **Output** (`check_rmf.py`): RMF frames from a run, loaded into IMP and
  rescored by C++, match the recorded scores once the EM list effect above
  is accounted for:
  * to 2e-2 (2.5e-5 relative) for a float32 GPU run (nightly 20261019);
  * to 3e-4 for a float32 CPU run (2026-10-13).
* **Final results** (`compare_final.py`, nightly 20261019): complete runs
  of the tutorial's protocol (20,000 frames with annealing) from the
  tutorial's shuffled start.
  * 60 C++ runs: the fast build with `score_moved`, which is exact and only
    faster.
  * 1,024 JAX chains, with `--cov frozen` as in C++.

  | | C++ (60 runs) | JAX, EM pairs within cutoff + 2 Å | JAX, EM pairs within cutoff |
  |---|---|---|---|
  | final score | 611.5 ± 2.3 | 611.3 ± 0.5 | 610.1 ± 0.5 |
  | best score per run | 594.5 ± 2.0 | 594.6 ± 0.4 | 593.6 ± 0.5 |
  | EM, second half at kT 1 | 129.6 ± 0.5 | 130.0 ± 0.1 | 128.0 ± 0.1 |
  | total, second half at kT 1 | 615.4 ± 2.1 | 614.8 ± 0.5 | 613.6 ± 0.5 |
  | final Rpb4 / Rpb7 position, distance from the C++ mean | | 1.2 / 1.0 Å | 1.7 / 1.8 Å |

  * **EM pairs within the cutoff + 2 Å** (what a freshly rebuilt C++ list
    holds): all 15 compared quantities agree with C++, with
    Kolmogorov–Smirnov and Mann–Whitney p ≥ 0.23. They are the final and
    best scores, every term, and the final position and orientation of both
    stalk subunits.
  * **EM pairs within the cutoff only** (the default here): EM is 1.6 units
    lower and the total about 1.8 lower, the effect of salilab/imp#1121. The
    stalk placement still agrees (p ≥ 0.1).
* **Distribution** (`ensemble.py`; 2026-10-13, C++ from conda IMP 2.25, an
  earlier version of this code, `--cov frozen`): 128 C++ runs against 256
  JAX chains, from the same shuffled start, 100 frames at kT 1.
  * Over frames 51–100, connectivity, excluded volume, cross-links and the
    total agree within about 1.2 standard errors.
  * EM differs by about one unit, because of the C++ EM lists
    (salilab/imp#1121).
  * At the checkpoints at frames 1–50 the total agrees within 1.5 SE, with
    cross-links 2–3.5 SE high at frames 2–5 only.
  * Per-mover acceptance agrees.
* **Replica exchange** (`run_mpi.py --mode rex`):
  * received configurations rescore to within 2 float32 ulps of what the
    sender sent;
  * neighbour acceptance is 0.20–0.21 on kT 1–2.5.

  It has **not** been compared with C++ PMI replica exchange.

## Files

| file | contents |
|---|---|
| `build_system.py` | builds the tutorial's system in-process, stopping before sampling |
| `scoring.py` | `TutSystem` (static tables read from the C++ objects) and the scoring terms, for numpy or jax.numpy |
| `sampler.py` | `TutSampler`: the batched, incremental MC step |
| `run_chains.py` | N independent tutorial runs on one device, with RMF output |
| `run_mpi.py` | several GPUs, one MPI rank each, through `IMP.mpi` (independent runs or replica exchange) |
| `validate_terms.py` | per-term comparison with C++ in float64 |
| `check_sampler.py` | incremental energy against a full recompute, plus a speed probe |
| `check_rmf.py` | rescores a run's RMF frames with C++ |
| `ensemble.py` | C++ against JAX score distributions |
| `compare_final.py` | final results of complete C++ and JAX runs |
| `bench_chains.py` | throughput against the number of chains |

## Running

You need an IMP nightly (with `IMP.mpi` for `run_mpi.py`), JAX 0.11.2 (with
CUDA for a GPU) and numpy. Set `IMP_TUTORIAL` to a clone of
`salilab/imp_tutorial` (tested at commit 8058229).

```bash
export IMP_TUTORIAL=/path/to/imp_tutorial
JAX_PLATFORMS=cpu python validate_terms.py --configs 4
JAX_PLATFORMS=cpu python check_sampler.py --x64
python bench_chains.py --chains 1024 4096 16384
python run_chains.py --chains 1024 --frames 20000 --thin 200 --writers 8 --out runs/x
python check_rmf.py runs/x
mpirun -np 4 python run_mpi.py --mode independent --chains 4096 --frames 20000 --thin 2000 --out runs/mpi
mpirun -np 4 python run_mpi.py --mode rex --chains 1024 --frames 2000 --out runs/rex
```

## Relation to IMP's JAX path

This sampler does not use IMP's JAX `MonteCarlo` or the restraints'
`_get_jax()`. It reads parameters from the C++ objects and has its own
scoring code.

IMP develop (4c5f9ef0cf) cannot yet run this tutorial through its JAX path.
`isd.GaussianEMRestraint` has no `_get_jax`, and the PMI excluded-volume
restraint (`PairsRestraint` over a `ClosePairContainer` with
`RigidClosePairsFinder`) would also need its rigid-body exclusion in JAX.

## Limitations

* **Specific to this system.**
  * Supported restraints:
    * connectivity: `SphereDistancePairScore(HarmonicUpperBound)` pairs;
    * PMI excluded volume: a soft sphere over a close-pair container with
      rigid-body exclusion, scored densely;
    * single-contribution `CrossLinkMSRestraint`s with fixed σ and ψ;
    * `isd.GaussianEMRestraint`.
  * Supported movers: `BallMover`s on nonrigid members of fixed bodies, and
    `RigidBodyMover`s.
  * Other restraint names, mover types or restraint variants raise
    `NotImplementedError`.
* **EM close-pair lists and covariances:** see the differences at the top.
  `validate_terms.py` checks EM on C++'s own lists.
* **The cross-link forward model** is the one IMP currently computes
  (salilab/imp#1120). If salilab/imp#1123 is merged, `xl_neglogp()` in
  `scoring.py` needs the same change.
* **float32 by default**, on CPU and GPU, with a full rescore every frame.
  float64 works on both (`--dtype float64`).
* **Output is not PMI's:** RMF per chain plus `scores.npy`, not stat files.
  There is no checkpoint/restart.

## Toward IMP

These pieces are generic, not tied to this system:

1. **Many chains per process.** Batch the sampler state over chains. IMP's
   JAX attribute sets (multiple coordinate sets per model, in develop since
   October 2026) are the natural basis.
2. **Incremental (moved-particle) JAX scoring.** Each restraint provides the
   energy change for a move of one particle or rigid body, from per-mover
   index tables built once.
3. **A fused `SerialMover` step without branches.** Each step is specialized
   for its mover (static indices) and uses `where`-based Metropolis, so
   `vmap` over chains does not execute every branch.
4. **Compact, component-first state.** Only the particles the restraints
   use, with static particles shared between chains.
5. **A rank-per-GPU helper for `IMP.mpi`** replica exchange, swapping
   configurations; plus RMF writer processes spawned before `MPI_Init`.
