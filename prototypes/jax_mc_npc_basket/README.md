# NPC_Basket sampled with JAX on GPUs

A research prototype, not part of IMP. It runs the published NPC_Basket
modeling scripts (yeast and mouse) with both replica-exchange stages
sampled by a Monte Carlo sampler written in JAX. Thousands of independent
runs go at once on one GPU, with the same representation, restraints,
movers and step sizes as the published protocol.

Prepared 2026-10-22, updated 2026-10-23 and 2026-10-26 (see "Changes on
2026-10-26" and "Changes on 2026-10-23").
Tested with IMP nightly `20261022.develop.cc48f8e34f`, JAX 0.11.2, Python
3.12, against
[integrativemodeling/NPC_Basket](https://github.com/integrativemodeling/NPC_Basket)
at commit `7511ffd`.

## Changes on 2026-10-26

Tested with IMP nightly `20261026.develop.e3e2a314c8` and JAX 0.11.2 on an
L40 and on CPU (`tests/`).

* Checkpoints and restarts like GROMACS `mdrun -cpi`, for long runs in
  pieces (overnight on a spot GPU, within a cluster time limit): `--cpi`,
  `--cpt`, `--maxh`, a clean stop on SIGTERM / Ctrl-C / a `STOP` file,
  `--frames2 -1` (until stopped), append-only outputs cut back to the
  checkpoint on restart. A continued run is identical, byte for byte, to
  an uninterrupted one. See "Long runs" below.
* Sampled frames go to one compact `traj.dof` per stage instead of one RMF
  file per run; `--to-rmf` writes RMF files from it afterwards (the 100
  best frames, or one run's trajectory), with the same coordinates as RMF
  files written while sampling. `--rmf-live` keeps the old per-run RMF
  output.
* Compiled programs are cached in `--out/jax_cache`, and XLA's kernel
  autotuning is off: runs with the same seed give the same frames.

## Changes on 2026-10-23

These were tested only on an RTX A4000 with IMP nightly
`20261005.develop.21ec66f3d8` (details below); `example_output/` and the
"Tested" section are from 2026-10-22.

* `--one-stage` skips REX1: from the shuffled start, the REX2 call samples
  with all restraints on (cross-links, EM, excluded volume, ...) for
  `--frames2` frames. Output goes to `one_stage/`.
* `--shuffle-each` starts every run from its own shuffle, as in the
  published runs: the script's `shuffle_configuration()` call is replayed
  with a seed per run (about 0.4 s each). The replay runs after the
  symmetry constraints exist, so the shuffle's collision check sees the
  symmetry copies in place; the script's own shuffle ran before them. The
  script's `optimize_flexible_beads(50)` (50 sweeps of 1 A bead moves,
  23 s per run in C++) is not replayed; the sampler makes as many bead
  moves in about 2.5 frames, with the script's step sizes and among the
  rigid-body moves.
* The RMF files also hold the restraints PMI registers for its RMF files:
  cross-links with their psi and sigma priors, Z-axial, membrane and
  excluded volume, with the values IMP.rmf stores at every saved frame,
  as in PMI's own RMF files. The psi and sigma values are stored too.
  `--no-rmf-restraints` leaves them out (faster output).
* A file `STOP` in `--out`, made after the run started, ends the last stage
  early and still closes the RMF files. An RMF file is only complete once
  closed, so killing the process loses the stage being written.
* Cross-link score in float32: for very long links `1 - t` rounds to 1
  (t < 3e-8, as right after a shuffle) and the score became infinite,
  while C++ (double) stays finite; up to t ~ 1e-6 it lost precision (0.18
  at t = 1e-7). `npc_sampler2.py` now computes `1 - prod(1 - t)` as
  `-expm1(sum(log1p(-t)))`.
* New: `best_models.py` collects the best saved frame of each run into one
  RMF file, copying the stored values unchanged (positions, psi and sigma,
  restraint scores; excluded-volume pair features are left out, since they
  differ between runs). `plot_scores.py` plots score against frame.

Tests (A4000, nightly 20261005):

* 16 runs with `--one-stage --shuffle-each`: run 0 rescored by C++ every
  10 frames; all 40 checks agree with the full JAX evaluation to 4e-7
  relative (0.35 at a score of 2.5 million, 0.001 at 3,500).
* 8 fresh shuffles: the cross-link term agrees with C++ to the printed
  precision (0.1); before the fix, 3 of the 8 were infinite.
* 100 shuffle replays: none failed.
* `best_models.py`: every stored value in the merged file equals its
  source frame.

## Speed

Yeast basket, REX2 stage, MC steps per second (one PMI frame = 20 x 354
movers = 7,080 MC steps). Each speedup is against C++ IMP nightly on one
core of the same machine.

| | MC steps/s | speedup |
|---|---|---|
| C++ IMP nightly, 1 core (workstation) | 874 | 1x |
| best exact C++ (IMP 2.25 + a bayesianem patch), 1 core | (2.25: 608 -> 1,056) | 1.7x |
| JAX sampler, 1 CPU core | 11,520 | 13x |
| JAX, RTX A4000, 512 runs | 2.27M | 2,590x |
| JAX, one L40, 2,048 runs | 11.54M | 4,900x* |
| JAX, 4x L40, 2,048 runs each | 46.2M | 19,700x* |

\* Against C++ nightly on the GPU node itself (2,347 MC steps/s).

* The published runs used IMP 2.18: 496 MC steps/s on one core.
* A cloud RTX PRO 6000 Blackwell reached 13.7M (yeast, 2,048 runs) and
  14.3M (mouse, 4,096 runs).
* Mouse (3,420 MC steps per frame) runs at about the same rate.
* These are sampler rates. A full run writing RMF files is about 15%
  slower (A4000, 512 runs: 1.96M yeast, 1.68M mouse), measured before the
  restraints went into the RMF files (`--no-rmf-restraints` gives that
  output again).

For the published REX2 protocol (yeast: 100,000 frames, 7.1e8 MC steps
per run), C++ IMP 2.18 needs 16.7 days per run on one core. One L40 does
2,048 such runs in about 1.5 days at the sampler rate (RMF output adds
~15%); an A4000 does 512 in about 2 days.

## Results match C++

* **Term by term.** Every restraint agrees with C++ through IMP's own JAX
  glue: connectivity, distances, Z-axial, membrane, cross-links and priors
  to 2e-6 relative, excluded volume to 6e-7. GaussianEMRestraint agrees
  exactly once C++ is given current covariances and a fresh pair list
  (see "Known issues").
* **Exact in float64.** After every frame the sampler's state is copied
  into the IMP Model and rescored by C++. Total and per-term differences
  are 0.0 at all printed digits for both systems.
* **float32 (the GPU default).** Per term within 0.02 (pairs) and 4e-4
  (excluded volume, EM) on scores of ~1e5. Acceptance uses each move's
  energy change; the running total is resynced from scratch every 10
  frames.
* **Same sampling as C++.** Yeast REX2 start, T = 1, 10 frames: 64 C++
  runs against 128 GPU runs of `npc_sampler2.py`, with the EM covariances
  handled as in C++ (`--cov-mode frozen`):
  * score distributions per frame: Mann-Whitney p 0.34, 0.048, 0.27,
    0.39, 0.98, 0.66, 1.00, 0.82, 0.64, 0.30 (one of ten below 0.05,
    about what chance gives);
  * mean score per run over frames 5-10: C++ 101,101 +- 2,931, JAX
    101,125 +- 2,691 (p = 0.68);
  * acceptance (C++ / JAX): beads 0.125 / 0.126, rigid bodies 0.059 /
    0.058, super rigid bodies 0.022 / 0.023, nuisances 0.642 / 0.646.

  The earlier sampler (`npc_sampler.py`) passed the same test (frame 10:
  C++ 85,592 +- 3,151, JAX 85,413 +- 3,388, p = 0.86).

## Setup

You need IMP nightly (it has `IMP.jax`, `IMP.bayesianem`, `IMP.npc` and
PMI) and JAX 0.11.2. On a GPU machine, JAX needs the CUDA wheels:
`pip install "jax[cuda12]==0.11.2"`, or `cuda13` with an NVIDIA driver
>= 580. Also install matplotlib and scipy.

```bash
bash install.sh                      # clones NPC_Basket @ 7511ffd, adds bench/
cd NPC_Basket/bench/jax
```

`setup_ubuntu.sh` does everything on a fresh Ubuntu GPU machine (sudo):
the IMP nightly .deb, a venv with JAX, then `install.sh`.

On GPUs the matrix products run in full float32, not TF32 (JAX's default
on recent NVIDIA cards). Here that costs 0.5% and halves the difference to
C++. Set `JAX_DEFAULT_MATMUL_PRECISION=default` to get TF32 back.

### Tested

From a fresh clone with `install.sh`, IMP nightly
`20261022.develop.cc48f8e34f`, JAX 0.11.2, one L40
(`example_output/` holds the logs of the first run):

* `run_npc_jax.py yeast --chains 256 --frames1 5 --frames2 30 --thin 10
  --check-every 5`: 4.4 min in total. Run 0 rescored by C++ agrees with
  a full JAX evaluation to 0.13 at REX2 scores of ~89,000. The running
  total that the sampler reports drifts more early in REX2, when scores
  are 1e6-1e7; it is only for reporting, and is resynced every 10 frames.
* `bench_chains.py yeast --sampler 2 --frames1 100 --chains 2048
  --frames 5`: 11.5M MC steps/s.

## Running

```bash
# CPU checks against C++ (minutes)
JAX_PLATFORMS=cpu python validate_terms.py yeast --configs 3
JAX_PLATFORMS=cpu python run_sampler.py yeast --sampler 2 --frames 4 --check --x64

# a short GPU run: 256 runs, REX1 10 frames, REX2 50 frames
JAX_PLATFORMS=cuda python run_npc_jax.py yeast --chains 256 \
    --frames1 10 --frames2 50 --thin 10 --check-every 10 --out runs/y256

# one stage: all restraints from a separate shuffle per run, 200 frames
JAX_PLATFORMS=cuda python run_npc_jax.py yeast --chains 256 --one-stage \
    --shuffle-each --frames2 200 --thin 10 --out runs/y256_one

# the 100 best saved frames in one RMF file; score plot
python run_npc_jax.py --out runs/y256_one --to-rmf one_stage --best 100
python plot_scores.py runs/y256_one --stage one_stage --out scores.png

# speed only
JAX_PLATFORMS=cuda python bench_chains.py yeast --sampler 2 --frames1 100 \
    --chains 512,2048 --frames 5
```

### Long runs: checkpoints and restarts

Runs stop and continue like GROMACS `mdrun -cpi`: run the same command
again and again (overnight on a spot GPU, within a cluster time limit),
check convergence in between, and continue as long as needed. Nothing is
allocated for the total length.

```bash
# sample until stopped; stop after 11.9 h; checkpoint every 15 min
JAX_PLATFORMS=cuda python run_npc_jax.py yeast --chains 2048 --one-stage \
    --shuffle-each --frames2 -1 --maxh 12 --cpi --out runs/week
# next night: the same command continues where it stopped
# check convergence in between (reads the growing trace.jsonl)
python plot_scores.py runs/week --stage one_stage --out scores.png
# RMF files from the trajectory: the 100 best saved frames, one run's frames
# (the run's settings are read from its checkpoint)
python run_npc_jax.py --out runs/week --to-rmf one_stage --best 100
python run_npc_jax.py --out runs/week --to-rmf one_stage --ladder 17 --frames 0::10
```

* `--cpi`: continue from `--out/state.cpt` if it exists; otherwise start
  a new run. Without `--cpi`, a directory that has a checkpoint is refused.
* `--cpt M`: a checkpoint every M minutes (default 15), written by the
  output thread while the GPU keeps sampling. `state.cpt` is replaced
  atomically and the previous one kept as `state_prev.cpt`, so a crash
  while writing loses nothing.
* The run stops cleanly (within two frames, then a checkpoint) on SIGTERM
  (what a spot instance or a batch system sends before the end), SIGINT
  (Ctrl-C), SIGUSR1/2, a file `STOP` in `--out` made after the start, or at
  99% of `--maxh` hours. A second Ctrl-C stops at once; so does `kill -9` or
  a power cut, which lose only the frames since the last checkpoint.
* `--frames2 -1`: no frame limit. Frame counts are totals for the stage, so
  continuing with a larger `--frames2` extends a finished run.
* Output files are append-only. On restart each one is checked against the
  checkpoint (MD5 of its last MiB) and cut back to the checkpointed frame,
  so frames written after the last checkpoint are written again.
* A continued run gives the same frames, byte for byte, as one
  uninterrupted run, when continued on the same GPU model with the same
  IMP, JAX and code; otherwise it warns, and the continuation is still
  correct sampling. Tested on CPU (8 runs, ladders of 2; SIGTERM, `kill -9`,
  continuing across REX1/REX2, extending a finished run) and on an L40 (256
  runs, ladders of 2, 300 frames; SIGTERM at frame 91, `kill -9` at 200):
  trace, checks, `traj.dof` and every array of the final state identical.
  XLA's kernel autotuning is off (`--xla_gpu_autotune_level=0`, unless
  `XLA_FLAGS` sets a level): with it on, two separately compiled processes
  can pick different kernels for the rotations' small matrix products and
  round differently. Off, separate runs with the same seed are identical
  too; on the L40 it made no speed difference (10.5M against 10.3M MC
  steps/s at 2,048 runs). The settings that define the run (system, chains,
  `--rex`, seed, sampler, `--one-stage`, `--shuffle-each`, `--cov-mode`,
  `--resync-every`) must not change; the run refuses if they do.
* Sampled frames go to `traj.dof` (degrees of freedom of every `--thin`th
  frame of each run: body positions and rotations, bead coordinates,
  sampled nuisances, and the score), not to per-chain RMF files. RMF files
  are made afterwards with `--to-rmf`, with the restraints evaluated by C++
  as during sampling. `--rmf-live` still writes per-chain RMF files while
  sampling (a new set per continued run, `chain_<i>.partNNNN.rmf3`; an RMF
  file is only readable once closed, so a killed run loses its set).
* Reading `traj.dof` from Python: `npc_ckpt.read_dof(path)` returns the
  header and a memory-mapped record array (`frame`, `chain`, `score`,
  `quat`, `xyz`, `nuis`).
* Compiled programs are kept in `--out/jax_cache` (or
  `$JAX_COMPILATION_CACHE_DIR`; `--jax-cache none` turns it off): a
  continued run skips compiling and runs exactly the programs its first
  session compiled.
* Disk and time, yeast, 2,048 runs on one L40 (1.37 s per frame with
  output): `traj.dof` takes 4.8 kB per run and saved frame, so 2.6 GB per
  hour at `--thin 10` and 0.26 GB per hour at `--thin 100`; `trace.jsonl`
  takes 44 kB per frame (115 MB per hour); `state.cpt` is 342 MB and takes
  about 1 s to write, in the output thread. With a checkpoint every 30 s the
  run kept its speed (10.6M MC steps/s, against 10.1M with none).

`run_npc_jax.py SYSTEM` (`yeast` or `mouse`) runs the modeling script
unchanged and replaces each `IMP.pmi.macros.ReplicaExchange` with the JAX
sampler:

* `--chains N` independent runs from the script's starting configuration,
  each with its own random stream. After REX1 every run keeps its own
  configuration for REX2.
* `--shuffle-each`: every run starts from its own shuffle (the script's
  shuffle replayed with a seed per run) instead of the script's one.
* `--one-stage`: no REX1; the REX2 call samples all restraints for
  `--frames2` frames, straight from the start.
* `--rex K` groups the runs into N/K replica-exchange ladders of K
  temperatures (geometric between the macro's minimum and maximum), with
  neighbour swaps after every frame. The default `--rex 1` gives
  independent runs at T = 1.
* `--frames1`, `--frames2`: frames per stage (published: REX2 100,000 for
  yeast, 300,000 for mouse).
* `--cov-mode rotate` (default) rotates the EM model covariances with
  their rigid bodies; `frozen` keeps them fixed, as C++ does (see "Known
  issues").
* `--seed`: for several GPUs, run one process per GPU
  (`CUDA_VISIBLE_DEVICES=i`) with different seeds and `--out`.
* `--cpi`, `--cpt`, `--maxh`, `--to-rmf`, `--rmf-live`: see "Long runs"
  above.

Output in `--out`, per stage (`rex1/`, `rex2/`, or `one_stage/`):

* `traj.dof` and `layout.npz`: every `--thin`th frame of each run (for
  ladders, the lowest-temperature replica of ladder i); `--to-rmf` turns
  them into RMF files with the restraints PMI puts in RMF files and their
  scores (`--rmf-live`: `chain_<i>.rmf3` written while sampling);
* `trace.jsonl`: one line per frame with every run's score;
* `checks.jsonl`: run 0 rescored by C++ every `--check-every` frames;
* `state.cpt`, `state_prev.cpt`: checkpoints (all runs' sampler states,
  about 150 kB per run for yeast);
* `summary.jsonl` and `run.log` for the whole run (one entry per continued
  run);
* `imp_output/`: what the modeling script itself writes before sampling.

## How it works

* `imp_jax_npc.py` adds JAX versions of the restraints that IMP's JAX path
  lacks: `bayesianem.GaussianEMRestraint` (all model x data Gaussian pairs,
  with the cutoff as a mask, plus the slope term),
  `npc.ZAxialPositionRestraint`, `npc.MembraneSurfaceLocationRestraint`,
  and PMI's excluded volume with its same-rigid-body exclusion.
* `npc_sampler2.py` is the sampler. Each move scores only the energy
  change of the restraints that touch the moved particles, from tables
  built once per system: beads, rigid bodies, super rigid bodies
  (`pmi.TransformMover`), nuisances. Many runs are vectorized on one
  device.
* `npc_setup.py` runs the modeling script up to the sampling stage.
* `bench/patched/` holds two small changes to the modeling scripts
  (`*.patch`): matplotlib's colormap API (`cm.get_cmap` was removed in
  matplotlib 3.9) and an optional excluded-volume slack, off by default.
  `install.sh` applies them to copies of the clone's own scripts
  (`Yeast_src/`, `Mouse_src/`), so the package carries no copy of them.
* `bench/gmm_cache_*` are the density GMMs the scripts fit at start-up,
  precomputed so that every start is the same and fast.

## Differences from the published runs

* By default all runs start from one configuration, the script's own
  start-up shuffle; the published runs shuffled each run separately
  (`--shuffle-each` does that, after the symmetry constraints are set up).
* `optimize_flexible_beads()` runs once, in the script; it is not repeated
  for `--shuffle-each` starts.
* `--one-stage` has no REX1: EM and excluded volume are on from the start.
* PMI stat files and mmCIF are not written: scores are in `trace.jsonl`,
  structures in the RMF files.
* EM covariances rotate with the rigid bodies by default, which C++ does
  not do (first issue below). `--cov-mode frozen` reproduces C++.
* The EM pairs are all pairs within the cutoff at each step, so the score
  depends only on the coordinates (second issue below).

## Known issues found along the way

These affect C++ IMP, so also the published sampling; all are open
upstream:

* EM restraints use stale covariances for Gaussians in rigid bodies:
  [salilab/imp#1119](https://github.com/salilab/imp/issues/1119), fix
  proposed in [#1118](https://github.com/salilab/imp/pull/1118). In
  NPC_Basket all 122 (yeast) and 150 (mouse) rigid-member model Gaussians
  keep the orientation they had when REX2 started. The score effect in
  short runs was small (0.05-0.3); whether it changed the published models
  is untested. `stale_cov_test.py` shows it on one Gaussian.
* The GaussianEMRestraint score depends on the close-pair list's history
  (up to ~3 score units here):
  [salilab/imp#1121](https://github.com/salilab/imp/issues/1121), fixes in
  [#1124](https://github.com/salilab/imp/pull/1124) and
  [salilab/bayesianem#4](https://github.com/salilab/bayesianem/pull/4).
* `ClosePairContainer` incremental updates can miss overlapping pairs
  (excluded volume):
  [salilab/imp#1122](https://github.com/salilab/imp/issues/1122), fix in
  [#1125](https://github.com/salilab/imp/pull/1125).

## Files (`bench/jax/`)

| file | |
|---|---|
| `run_npc_jax.py` | the full protocol on GPU or CPU (REX1 + REX2 or one stage, runs or ladders, RMF output, C++ checks) |
| `best_models.py` | for `--rmf-live` runs: the best frame of each run in one RMF file (with the restraints); `--csv` lists every saved score |
| `npc_ckpt.py` | checkpoints (`state.cpt`), append-only outputs cut back on restart, the `traj.dof` format, stop requests (signals, STOP file, `--maxh`) |
| `plot_scores.py` | score against frame over all runs (yeast: with the published models' mean) |
| `npc_sampler2.py` | the sampler (default); `npc_sampler.py` is an earlier, slower version (`--sampler 1`) |
| `imp_jax_npc.py` | JAX versions of the restraints, `install()` attaches them to IMP |
| `npc_setup.py` | builds a system in-process and stops before sampling |
| `validate_terms.py` | term-by-term check against C++ |
| `run_sampler.py` | one run on the CPU, rescored by C++ after every frame (`--x64` for float64) |
| `bench_chains.py` | speed against the number of runs |
| `ensemble.py`, `analyze_ensemble.py` | the C++ vs JAX sampling comparison above |
| `stale_cov_test.py` | the stale-covariance issue on one Gaussian |
| `imp_jax_fused.py`, `imp_jax_patches.py` | helpers from earlier IMP-JAX work |

Top level: `install.sh`, `setup_ubuntu.sh`, `example_output/` (the tested
run's log, summary and C++ checks).
