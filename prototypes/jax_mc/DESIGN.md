# jax_mc: a core plus recipes for fast, exact JAX Monte Carlo samplers

Status, 2026-11-03: steps 1 and 2 are done (`system`, `terms`, `engine`,
`check.terms`, `check.decisions`, `check.drift`, `bench`; see
[Progress](#progress)). The templates are not written yet.

## Why

The Pol II tutorial sampler is fast: 1.16 billion MC steps/s on one L40 with
agy_imp's plain-JAX version, and 4.6 billion on four. But it got there by
specializing everything to one system:
- 2 rigid bodies and 47 beads;
- a fixed mover order;
- tuning thresholds for one GPU model;
- caches that are only valid for one covariance mode.

The other systems got their own samplers, each written from scratch:
- NPC basket: `npc_sampler2.py`;
- mature NPC: `pm_sampler2.py`;
- mzIdentML: IMP's JAX MC with our fused patches.

A test on 2026-11-03 showed that the Pol II techniques don't simply drop
into these samplers. None of them has the copies those techniques remove, and
a chain-last vmap gave no gain on the NPC basket. Each sampler is built
differently: the code is duplicated, and so are the tests and the
"what's exact" arguments.

We want both:
- **a solid core** that runs any supported PMI system exactly, readably and
  reasonably fast, with tests that make changes safe;
- **recipes and templates:** documented, opt-in specializations with stated
  preconditions, measured gains and an exactness test, to be adapted or
  rewritten case by case for maximum speed.

**Non-goals:**
- a general replacement for IMP's own JAX path (`IMP.core.MonteCarlo` on
  JAX);
- supporting every IMP restraint;
- auto-tuning.

**Anti-goals** (a failure even if it works):
- a core that is as hard to read as `sampler_soa.py`;
- recipes that silently assume something about the system;
- system-specific thresholds or indices inside the core.

## What exists today

| system | sampler | design | one L40 | specific to the system |
|---|---|---|---|---|
| Pol II tutorial | `jax_mc_rnapolii/sampler_soa.py` + agy's 5f1778d | incremental; chain-last struct of arrays; unrolled static mover order; static-EV grid; blocked beads; EM cache | 1.16 bn (agy, rescore every 20th frame); 665 M (public, rescore every frame) | 2 bodies, beads only on static bodies, thresholds tuned for the L40, EM cache valid only for `rotate` |
| NPC basket (yeast) | `jax_mc_npc_basket/.../npc_sampler2.py` | incremental; generic over PMI restraint types; dynamic mover index with per-kind padded tables and `lax.switch`; chain-first vmap | 11.5 M | little: generic inventory of PMI restraints and movers |
| mature NPC | `pm_sampler2.py` (local) | every step moves all 112 bodies; full energy over neighbour lists (inner and outer skin) | 1.11-1.22 M | the membrane, overall-position and bbox terms; collective moves |
| mzIdentML tutorial | IMP's JAX MC + `imp_jax_fused` patches | full rescore every step; masked selects | 67-70 M | none (IMP restraint functions) |

All speeds are MC steps/s.

**Where the time goes,** measured on one L40 on 2026-11-03
(`imp_jax/fig1_agy/NOTES.md`):
- NPC basket: EV 30%, EM 32%, cheap terms 23%, overhead 22%.
- mature NPC: neighbour-list rebuilds 28%; the rest is the full energy.
- mzIdentML: ~12 kernels per step, limited by the work of the full rescore.

## The shape

```
            IMP model + restraints + movers (PMI)
                          |
                   jax_mc.system        extract flat tables (numpy, float64)
                          |
       jax_mc.terms    jax_mc.engine    reference incremental sampler
       (full + delta)  (chain-last, any mover order, no tuning)
                          |
         jax_mc.run       drivers: frames, REX ladders, rescore policy,
                          output (DOF snapshots -> RMF), multi-GPU
         jax_mc.check     exactness harness: terms vs C++, decisions vs
                          the engine, drift, ensembles vs C++
  ---------------------------------------------------------------------
  recipes/   one page + reference code per technique: preconditions
             (asserted), gain, test
  templates/ worked samplers that compose the core with recipes:
             few_bodies_beads (Pol II), many_movers (NPC basket,
             mzIdentML), collective (mature NPC)
```

A system-specific sampler is a template copied and adapted. It keeps using
`system`, `terms`, `run` and `check`. It replaces, or specializes, the
engine's step.

## The core

### `jax_mc.system`: what to simulate

This is the host-side extraction from an IMP model into a `System` of flat
float64 numpy tables:
- particle rows;
- rigid bodies, with local coordinates and the initial frame;
- nuisances, with bounds;
- movers in SerialMover order: kind, rows, parameters;
- one entry per scoring term: type, parameters, item lists.

Rules:
- **A registry of extractors keyed by IMP class.** Anything unsupported
  raises `Unsupported` with the class name. There are no name-based guesses.
- **Merge the three extractors we already have:**
  - `TutSystem` (scoring.py);
  - the inventory part of `NPCSampler2.__init__` (it already covers
    harmonic-upper-bound pairs, z-axial, membrane, cross-links, EV, EM,
    nuisances, ball, rigid-body and translation movers);
  - `PMSystem` (pm_jax.py: overall position, membrane depth/binding, bbox).
- **Static analysis lives here, not in samplers.** Examples: which particles
  never move (the static set), which terms couple which movers, which EM
  Gaussians belong to which body. Recipes read these facts. They never
  rederive them.

### `jax_mc.terms`: how to score

Each scoring term is one small class:

```python
class Term:
    rows_read, nuis_read                  # what it reads (engine skips
                                          # terms a mover can't change)
    def full(self, st): ...               # (C,) energy per chain
    def cache(self, st): ...              # per-chain partial sums, or None
    def move_tables(self, s, moves): ...  # host tables, one row per mover
    def delta(self, tab, st, cache, mv): ...      # (dE (C,), new)
    def delta_nuis(self, st, cache, nuis): ...    # the same, nuisance move
```

`new` is `None` (cache unchanged), a new cache (the engine keeps it where
the move is accepted), or `commit(acc)`, which updates a large cache in
place by masked scatter. The defaults for `delta` and `delta_nuis` rescore
the term with the move applied. That is exact and slow, and only allowed
for terms without a cache; the IMP JAX fallback uses it.

They are written once for the chain-last layout, readable, and tested in
float64 against the C++ restraint they replace. Every mover is rigid on
its rows (one bead, one body, or a TransformMover's group), so pair terms
only score pairs with exactly one end moved. The exception is EM with
frozen covariances, where a rotation changes the overlap of two Gaussians
on the same body. Terms made of one particle per item share
`RowTerm`.

The first set covers the four systems:
- connectivity;
- excluded volume;
- cross-links (CrossLinkMS);
- Gaussian EM;
- z-axial;
- membrane;
- overall position;
- bbox;
- nuisance priors.

### `jax_mc.engine`: the reference sampler

- **State layout:** chain-last, written that way natively. `X (3, rows,
  C)`, `quat (4, bodies, C)`, `nuis (N, C)`, the energy per term `E (T, C)`,
  the term caches, `kT (C,)` (so REX ladders need nothing special),
  per-chain keys, the sweep count and accepted moves per mover. Beads'
  degrees of freedom are their rows of `X`. A body's translation is its
  center row (the first member, at local 0), so nothing is stored twice.
  - A `vmap` with `in_axes=-1` is not enough: XLA compiles the same loop as
    with chains first (measured on the NPC basket).
  - Chain-last is part of what made the Pol II struct-of-arrays sampler twice
    as fast on GPU (with the static-EV grid and no array copies).
- **Sweep:** a `lax.scan` over the movers in SerialMover order. Movers are
  grouped by kind (`ball`, `rb`, `transform`, `normal`) and size: the
  number of rows, padded to a power of 2. Each group has its tables, one
  row per mover, and its branch of a `lax.switch`. A branch:
  1. proposes;
  2. asks the terms its group can change for `delta`;
  3. applies Metropolis;
  4. writes the moved rows back with a masked select (`X.at[:,
     rows].set(where(acc, new, old))`), and updates `E` and the caches.

  This works for any number of movers, and compile time grows with the
  number of groups, not movers.
- **`Engine(s, incremental=False)`** is the same sampler with every term
  rescored from scratch after each proposal. It shares the proposals and
  draws, and is the reference for `check.decisions`.
- **Static particles** are read from one copy shared by all chains in the
  EV and isd EM deltas. Pol II: 369 of 453 EV particles are static.
- **`CollectiveEngine`** is the reference when every mover moves at every
  MC step: IMP's MonteCarlo with several movers, as in the mature NPC. All
  movers propose, every term is rescored, and one Metropolis decision
  keeps or rejects the whole step. `engine.make(s)` picks the right
  engine.
- **Not in the engine:**
  - no thresholds tuned to a GPU or a system;
  - no unrolling;
  - no caches that hold only under conditions the engine doesn't check.

**Random numbers (a contract for every sampler built on the core).**
- **The draws,** for chain c, sweep s, in the run's dtype:
  - `chain_key[c] = fold_in(key(seed), c)`, with c the chain's index in
    the whole run (over all processes);
  - `k = fold_in(chain_key[c], s)`;
  - `U = uniform(k, (n_movers, 3))`;
  - `Z = normal(fold_in(k, 1), (n_movers, 6))`.
- **Mover i uses its row.**
  - Ball, or translation: `Z[i, 0:3] / |Z[i, 0:3]| * step *
    cbrt(U[i, 0])`, in global coordinates. Balls on a static body's local
    coordinates (Pol II) use it too: the distribution is the same.
  - Rotation: axis `Z[i, 3:6]` normalized, angle `max_rot * (2 U[i, 1] -
    1)`, then `q' = normalize(q_axis_angle * q)`.
  - TransformMover: about IMP's center (beads, plus body centers weighted
    by their number of members).
  - Nuisance: `clip(x + sigma * Z[i, 0], lower, upper)`.
  - Metropolis: accept if `U[i, 2] < exp(-dE / kT)`.
  - Collective step s (every mover moves): the draws of sweep s, mover i
    proposes from row i, and the decision is `U[0, 2] < exp(-dE / kT)`.

  This is the shape agy_imp proposed (forum #imp t154 p1233). It is what
  its sampler already uses, so no `erf_inv` or Box-Muller on uniforms.
- **Slicing is allowed** if the draws stay bitwise equal: generate the full
  shape, then slice.
- **Per-chain keys** give each chain the same numbers whatever C is.
- **Per-sweep `fold_in`** gives random access, so blocked and unrolled
  samplers can consume the same numbers.

Every template must use the same draws as the engine. That is what makes
"same decisions as the engine" testable.

### `jax_mc.run`: drivers

`python -m jax_mc.run <system> --out DIR ...` runs any sampler that follows
the protocol (`--sampler module:factory`; default: the reference engine).
It takes its ideas from `run_soa.py`, `run_mpi.py` and the NPC basket's
checkpointing:
- **Frames** of N sweeps, in blocks of frames inside one jitted call. The
  next block is dispatched before the previous block's records are
  written.
- **Temperatures:**
  - fixed (`--kT`);
  - PMI's simulated annealing (`--anneal tmin,tmax,nmin,nmax`);
  - replica exchange (`--ladder`, or `--ladder` alone for the protocol's
    ladder). The chains form C / len(ladder) independent ladders of
    consecutive chains. Neighbouring rungs try to swap after each frame,
    even and odd pairs on alternate frames.
- **Rescore policy** (R10): every K-th frame the energies are rescored from
  scratch (`resync`), and the largest |tracked - rescored| is recorded.
- **Outputs are append-only,** so runs can be open-ended:
  - `records.f64`: frame, drift, and every chain's energy and kT;
  - `dofs.f32`: every `--thin` frames, every chain's degrees of freedom;
  - read them with `run.records(out)` and `run.snapshots(out)`.
- **Checkpoints,** GROMACS-style:
  - `--cpt` minutes, written by a thread from a device copy;
  - `--cpi` continues, cutting the outputs back to the checkpoint;
  - `--maxh`, a STOP file, SIGINT or SIGTERM stop the run cleanly.
  - Every program goes into a persistent compile cache, so a restarted
    session runs the same executables.
  - The Python hash seed is pinned to 0, because PMI builds systems in set
    order.
- **RMF afterwards:** `python -m jax_mc.rmf OUT --chains 0,1 --rung R
  --best N` rebuilds the system and writes `OUT/rmf/chain<c>.rmf3`,
  `rung<r>.rmf3` (following one temperature through a ladder) or
  `best.rmf3`. Frames are named with frame, chain and energy.
- **One process per GPU** (`--ranks N --rank r`, or under mpirun or
  srun): each runs `--chains` chains into `OUT/rank<r>/`. Together they are
  exactly one run of N × `--chains` chains, because the keys are per chain
  and the exchange draws per ladder of the whole run. The processes don't
  communicate.

### `jax_mc.check`: the exactness harness

Exactness has four levels. Every sampler states which ones it meets.

| level | meaning | how it's checked |
|---|---|---|
| T (terms) | each term equals C++ to float64 round-off at random states | `check.terms(system)`: IMP `evaluate` against `Term.full`, all terms, all four systems |
| D (decisions) | same accept/reject decisions as the engine for N sweeps, same draws, float64 | `check.decisions(a, b, sweeps, chains, kTs)`. Covers more than one chain-count regime (≤ 2,048 and > 4,096), kT 1 and kT where acceptance is about 50%. **Every recipe must pass it against the engine.** |
| F (float32) | in float32, the fraction of flipped decisions and the tracked-vs-rescore drift are reported and bounded | `check.drift`, `check.decisions(..., dtype=float32)` |
| E (ensembles) | complete runs agree with C++ in final scores and structure statistics | `check.ensembles` (from `compare_final.py`) |

**Note.** "Bit-for-bit identical" is only claimed when the operation order is
the same. Recipes that reorder sums, like blocking, cached parts or
`d*(s/n)`, claim level D, not bitwise equality.

## Recipes

One page each in `recipes/`, in a fixed format:
- what it does;
- **preconditions, each enforced by an assert** in the reference code;
- the measured gain, on which system and GPU;
- the exactness level it keeps, and the `check` call that proves it;
- the reference code, a small function, not a sampler.

| # | recipe | preconditions | measured |
|---|---|---|---|
| R1 | static/movable split; static EV as a packed (x, y, z, r) grid with a compact cell list | a static set exists (from `system`); cell indices fit the index type (assert) | Pol II: 369 to ≤21 candidates per query, part of the 2× on one L40; NPC basket: 351 of 774 EV particles static (gain estimated ≤10%) |
| R2 | unrolled sweep with static row offsets; in-place `dynamic_update_slice` behind `optimization_barrier`; flat tuple carry | few movers (compile time; 49 fine, 354 untested); contiguous rows per mover | agy: ~100-280 copy kernels per sweep removed (Pol II) |
| R3 | blocked bead steps (score B consecutive beads in one reduction, correct within the block) | consecutive bead movers with consecutive rows; every coupling term has a within-block correction; no EM on beads | agy: ~340 to ~28 kernel launches per sweep; GPU only, 19% slower on one CPU core |
| R4 | per-body EM cache (model-data sums; intra-body overlap computed once per call) | covariances rotate with the body (`rotate`), **not** `frozen`; any number of bodies; EM Gaussians only on rigid bodies | Pol II, one CPU core, float64: R1 and R4-R7 together +23% |
| R5 | closed-form 6-component covariances; elementwise 3×3 products | none | removes TF32 error on GPU (max 2.1 kT to 2.6e-4 kT); fewer intermediates |
| R6 | sweep-wide precompute of bead proposals' static terms | one move per bead per sweep, in bead order (assert); the static set is truly static | Pol II, one CPU core: part of that +23% |
| R7 | sliced RNG | follows the random-number contract | Pol II: 48% fewer RNG writes |
| R8 | neighbour lists with inner and outer skins; rebuild of a single chain inside the step | collective moves; bounded displacement per step | mature NPC: rebuilds are 28% of the step, about 0.29 ms each |
| R9 | masked select instead of scatter for the write-back | small arrays (rows × chains); large arrays prefer scatter | mzIdentML uses it; the NPC basket chose scatters on purpose |
| R10 | full rescore only on recorded frames | the tracked energy's drift is checked on the recorded frames | Pol II: 1.06-1.16 bn instead of 0.93-1.0 bn on one L40 |
| R11 | device-side frame blocks, pipelined host output, DOF snapshots with RMF afterwards | none | Pol II: 1,024 runs in 3.1 min, against 3.5 min for the prototype |
| R12 | hand-written fused kernel (Pallas/Triton) | last resort; not maintained | Pol II: 4.19 bn on 4 L40s, now matched by plain JAX (R1-R7) |
| R13 | precision: float32 with drift checks; not bf16, not TF32 | none | bf16: ~6 kT error, 6% flipped decisions, no speedup |

The review of agy's code on 2026-11-03 turned into preconditions above:
- R4 is `rotate` only, and doesn't assume 2 bodies;
- R1 needs the index range check;
- R3 and R6 need the bead-order assert.

## Templates

A template is a complete sampler that a person or an agent copies and
adapts. Each one has:
- a `README.md` with the system shape it fits and the recipes it uses;
- a benchmark script;
- a `check` script that runs levels T and D (plus F for float32).

| template | system shape | starts from | recipes | target on one L40 |
|---|---|---|---|---|
| `few_bodies_beads` | few rigid bodies, many beads, fixed mover order (Pol II) | agy's `sampler_soa.py`, moved onto `system`, `terms` and the random-number contract, with the review fixes | R1-R7, R10, R11 | agy's 1.16 bn, now with D against the engine |
| `many_movers` | hundreds of movers, generic PMI terms (NPC basket, mzIdentML) | the engine plus R1, R5 and R9 | R1, R5, R9-R11 | NPC basket > 11.5 M; mzIdentML > 67 M (incremental instead of full rescore) |
| `collective` | every step moves every body (mature NPC) | `pm_sampler2.py`, moved onto `system`, `terms` and `run`, with its own step | R8, R10, R11 | ≥ 1.2 M; then batch or fuse the rebuilds. Done: `templates/collective/`, 1.06 M, the same as `pm_sampler2` measured the same way |

## From the engine to a specialized sampler

The engine is generic and slower on purpose. The way to full speed is a
series of small steps away from it, each checked against it:
1. make one change;
2. run `check decisions` against the engine, in float64, at ≤ 2,048 and
   > 4,096 chains;
3. run the benchmark;
4. drop the change if it doesn't make things faster.

**The sampler protocol** is all `check` needs, so a specialized sampler can
keep the engine's state or have its own:

```python
class MySampler:
    s                                         # the jax_mc System
    def init(self, C, seed=0, kT=None, dofs=None, chain0=0): ...
        # state: any pytree (a dict with 'kT' (C,) for jax_mc.run);
        # chains chain0 .. chain0 + C - 1 of the run
    def run(self, st, n_sweeps, record=False): ...
        # record=True: (state, decisions (n_sweeps, n_movers, C) bool)
    def dofs(self, st): ...         # the System's degrees of freedom
                                    # (device arrays, jittable)
    def total(self, st): ...        # (C,) tracked total energy
    def resync(self, st): ...       # optional: energies from scratch
```

It must use the random-number contract and the engine's proposals.
`Engine.draws` gives the numbers.

```
python -m jax_mc.check decisions polii --sampler my.module:factory \
    --chains 8192 --ref-chains 512
```

`factory(system)` returns the sampler. It runs every chain, and the
engine, as the reference, runs the first `--ref-chains` of them.

**Three levels of specialization, smallest first:**
1. **A faster term.** Subclass a term and override `cache`, `move_tables`
   and `delta`, then build the system with
   `extract(source, replace={SoftSphereEV: StaticGridEV})`. The engine and
   everything else stay as they are. R1's static EV grid fits here.
2. **A faster branch.** Subclass `Engine` and override `_branch(G)` for
   some groups, keeping the state layout and the draws. For example, R3's
   blocked bead steps replace the `ball` group's branch.
3. **Your own sweep.** Override `Engine.sweep` (R2's unrolled mover order),
   or write a sampler from scratch with its own state layout, keeping only
   the protocol and the draws. `few_bodies_beads` is this level.

Each recipe says which level it works at. A template is a set of steps
that have already been taken and checked.

## Acceptance

- **Level T** passes on all four systems from the IMP model, with no
  system-specific code outside `system` extractors and `terms`.
- **The engine runs Pol II, the NPC basket and mzIdentML.** On one L40:
  - NPC basket: at least `npc_sampler2`'s speed (met: see Progress);
  - mzIdentML: at least 67 M (met);
  - Pol II: within 2× of `few_bodies_beads`. Dropped: the engine is
    4.4× slower than agy's 1.16 bn, and R1-R3 remove what limits it (see
    Progress). Arthur, 2026-11-03: a slower generic engine is fine as long
    as the path to a specialized sampler is clean (the section above).
- **`few_bodies_beads`** reaches agy's speed and passes level D against the
  engine in float64, in both chain-count regimes.
- **Every recipe** has its `check` call in one test script, which runs on a
  GPU node in under 30 minutes.
- **Documentation:** a "make a fast sampler for your system" page that walks
  through choosing a template, then adding recipes one at a time, checking D
  after each.

## Order of work

1. **`system` + `terms` + `check.terms`** for the four systems, reusing the
   three extractors and the existing term code. This is useful even if the
   rest changes. Done 2026-11-03.
2. **`engine` + `check.decisions` + the random-number contract.** Benchmark
   against today's samplers on one L40. Done 2026-11-03.
3. **`few_bodies_beads`:** port agy's sampler onto the core and fix the review
   items. Write recipes R1-R7 from it.
4. **`many_movers` and `collective`;** recipes R8-R11. Done 2026-11-03:
   `collective`, `jax_mc.run` (R10, R11) with RMF output and one process
   per GPU. `many_movers` isn't separate code for now. The engine already
   beats its targets (NPC basket 14.6-20.5 M against 11.5 M; mzIdentML
   1.03 bn against 67 M), and its recipes are R1-R7 (agy_imp's) plus R10
   and R11, which the driver does.
5. **The guide;** clean up the old prototype directories, which keep working
   until the templates replace them.

## Decisions taken without asking (change any)

- **Location:** `prototypes/jax_mc/` on a local branch, `jax_mc_core`, of
  the fork. Nothing is pushed. The old `jax_mc_rnapolii` and
  `jax_mc_npc_basket` stay as they are until the templates match them.
- **Plain JAX only in the core.** Pallas stays a recipe.
- **The engine's sweep is dynamic** (scan + switch), not unrolled.
  Unrolling is recipe R2.
- **The random-number contract** (`fold_in` per sweep, 3 uniforms and 6
  normals per mover). The existing Pol II and NPC samplers draw differently, so their
  old trajectories won't be reproduced. Level D is checked between samplers
  on the core.
- **mzIdentML moves from IMP's full-rescore path to the incremental engine.**
  Its Fig 1 number stays as the "IMP JAX path" reference.

## Open questions for Ben

- Long term, should the core stay a prototype directory or become part of
  IMP? For example, `IMP.jax` gaining an incremental sampler that reads
  per-restraint `delta` functions.
- Should IMP restraints expose a JAX `delta` next to their JAX score
  function, so that `system` and `terms` don't need their own copies of the
  math?

## Progress

**Step 1 (2026-11-03): `system`, `terms`, `check.terms`.** Run with
`python -m jax_mc.check terms polii|npc|pm|mzid`. Each run uses 6 random
states near the start (bodies, beads and nuisances moved by up to 3 mover
steps) and float64 on the CPU. Both checks below pass on all four systems:
- positions from the degrees of freedom equal IMP's exactly;
- every term matches the C++ restraints it replaces to round-off.

| system | rows | bodies / beads / nuisances | terms | worst relative error |
|---|---|---|---|---|
| mzIdentML | 38 | 3 / 11 / 2 | cross-links, priors | 4e-16 |
| Pol II | 688 | 2 / 47 / 4 | connectivity, EV, cross-links ×2, priors, isd EM | 2e-16 (EM 6e-3 with cutoffs) |
| NPC basket | 1,644 | 39 / 309 / 5 | pairs, z-axial, membrane surface, cross-links ×2, priors, bayesianem EM, EV | 2e-16 (EM 6.5e-4 with cutoffs) |
| mature NPC | 32,854 | 112 / 0 / 0 | bbox, membrane binding and depth, overall position ×32,256, isd EM (swapped), EV | 3e-14 (EM 8e-4 with cutoffs) |

**EM is checked with the pair lists the C++ restraint used** at each state
(`cpp_masks`). With plain cutoffs it differs by the slack effect
(salilab/imp#1121).

**Things learned:**
- **C++ doesn't rotate covariances.** IMP still scores Gaussians with the
  global covariance from setup (`--cov frozen`); `rotate` is 1.7e-2 off on
  Pol II.
- **`sphere_cap` is single precision in C++.** Its arguments are floats.
  `CrossLinks.CPP_FLOAT` emulates that for checks, but only eagerly: under
  `jit`, XLA CPU's fused float32 arithmetic differs by about 1e-9. So
  `check.terms` evaluates eagerly. Production code stays plain float64 or
  float32.
- **Host constants must be numpy.** Constants computed with `jnp` in
  `finalize()` were float32 whenever x64 wasn't on yet.
- **Some parameters need recording at construction.** EM cutoffs and particle
  lists, the bounding-box function and the membrane-score parameters have no
  Python getters. `imp_hooks.install()` records their constructor arguments,
  and snapshots the EM restraints' Gaussians, whose setup-time terms C++
  caches. The mature NPC protocol builds its EM restraint as `(data, model)`.
- **Two PMI ball-mover styles.** Pol II's ball movers act on the local
  coordinates of non-rigid members of static bodies (a ball in the parent
  frame). The NPC basket's act on x, y, z.

**Review by agy_imp (t154 p1233), applied:**
- `X (3, rows, C)` stays the layout.
- A body's own particle is now its first member at local (0, 0, 0), so
  `positions()` is a plain concatenation, with no gather.
- EV accumulates in a `fori_loop`, and keeps its rows in static/body/bead
  order (`n_static`) for the delta step and the static-grid recipe.
- bayesianem EM computes its slope distances separately, so XLA can fuse the
  main reduction on GPU.
- Kept apart on purpose, two quaternion-to-matrix forms:
  - positions use IMP's `Rotation3D` form (matches IMP exactly);
  - covariances use Eigen's form (as `core::Gaussian` does).
- For step 2: check on GPU that the scan/switch loop has no full copies of
  `X`, with every branch threading `X` through.
- Split of work: agy_imp takes `few_bodies_beads` and recipes R1-R7 once the
  engine exists. It has fixed the six review items in its sampler
  (antigravity repo ff5cf6a); they were verified on GPU up to 65,536 chains.

**Step 2 (2026-11-03): `engine`, `check.decisions`, `check.drift`,
`bench`.**

Level D: `python -m jax_mc.check decisions <system> [--chains N
--ref-chains R --kT 1,10]` runs the incremental engine against the
full-rescore engine with the same draws, in float64. A chain's draws don't
depend on the number of chains, so the costly reference only runs the first
R chains. No decision differs anywhere:

| system | CPU, 8 chains, kT 1 and 0.3 | L40, 1,024 chains (256 checked), kT 1 and 10 | L40, 8,192 chains (128 checked) |
|---|---|---|---|
| mzIdentML | 0 of 640 each | 0 of 20,480 each | 0 of 6,144 |
| Pol II | 0 of 1,960 each | 0 of 62,720 each | 0 of 18,816 |
| NPC basket | 0 of 14,160 each | 0 of 453,120 each | 0 of 135,936 |

The tracked energy matches a rescore to round-off: 1e-13 to 1e-11, and
3e-7 on the NPC basket, whose energy at the REX2 start is 1.3e8. Level T
was rerun after the changes and still holds on all four systems.

Speed on one L40, float32 (`python -m jax_mc.bench`), in MC steps/s:

| system | movers (groups) | engine | today's sampler |
|---|---|---|---|
| mzIdentML | 16 (4) | 259 M at 16,384 chains; 772 M at 65,536; 1.03 bn at 262,144 | IMP JAX path, 67-70 M |
| NPC basket | 354 (9) | 14.6 M at 4,096; 19.5 M at 16,384; 20.5 M at 32,768 | `npc_sampler2`, 11.5 M |
| Pol II | 49 (2) | 40 M at 4,096; 227 M at 32,768; 264 M at 65,536 | public SoA 665 M; agy 1.16 bn |

**Things learned:**
- **bayesianem EM can't keep running sums in float32.** Densities with no
  model nearby have sums near 0, next to single overlaps of order 10. The
  energy depends on log(sum + 1e-10), so the residue of adding and later
  subtracting an overlap moved it by kT. The term now keeps per-block sums
  (16 Gaussians a block) and recomputes the blocks a move touches. NPC
  basket, 50 sweeps: drift 3.1 down to 7e-4 in float32, and 6e-6 down to
  1e-12 in float64.
- **Large totals drift in float32,** for example the NPC basket's pairs
  term at 1.2e8 (about 200 after 50 sweeps). Decisions use per-move
  differences, so they don't see it. Drivers resync at each frame.
- **Static particles read once** (EV, isd density): Pol II +25% at 16,384
  chains.
- **What limits the engine:**
  - With few chains, it's kernel launches. A step is 19-57 kernels
    (Pol II: 26 for a bead, 51 for a body, before two small cuts), about
    100 µs whatever the chain count.
  - With many chains, it's memory traffic: a step reads every moving EV
    particle of every chain.
  - R1 (static grid), R2 (unrolling) and R3 (blocked beads) remove exactly
    these, which is why the Pol II gap is the template's job.
- **XLA autotuning is off** (`jax_mc.xla_defaults()`). On GPU it took most
  of the compile time and grew with the number of chains (NPC basket,
  4,096 chains: 56 s against 8.5 s), with no effect on speed.
- **Command buffers for the loop and its conditionals:** +4% on Pol II.
  Not used.
- **agy_imp's check passes:** no copy of `X` in the compiled sweep loop.
  The only copy is at entry, when the input isn't donated.

**Step 4, part 1 (2026-11-03): `CollectiveEngine` and
`templates/collective/`.**

- **Collective engine:** the reference for systems whose movers all move
  at every step.
  - Its energies agree with C++ to within the known EM pair-list slack.
    The term check gives 0.26 without C++'s pair lists, and the same 0.22
    already shows at the initial state.
- **Collective template:** `pm_sampler2`, ported onto the System and the
  terms. It reuses the reference's proposals and the membrane and
  bounding-box terms' own energy functions. Its fast parts (neighbour
  lists, closed-form overall position, EM pair tables, membrane
  candidates) each assert their precondition.
- **Level D, float64, against the collective engine:**

  | where | chains (reference) | steps | kT | decisions | rebuilds per chain |
  |---|---|---|---|---|---|
  | CPU | 8 (4) | 10 | 30, 300 | 0 of 40 differ, each | |
  | L40 | 256 (16) | 200 | 300 | 0 of 3,200 differ | 48 inner, 4 outer |
  | L40 | 256 (16) | 200 | 3000 | 0 of 3,200 differ | 155 inner, 15 outer |

  - Energies agree to 1e-11 (1e-10 at kT 3000), and positions exactly.
  - At kT 3000, 4 of 256 chains outgrew their lists, which are sized from
    the initial state. Such chains get a NaN energy by design, so it shows.
    That temperature takes the assembly apart; the protocol's ladder ends
    far lower.
- **Speed,** one L40, float32, 512 chains on the protocol's ladder, timed
  after 20,000 burn-in steps:

  | sampler | MC steps/s | notes |
  |---|---|---|
  | template | 1.06 M | |
  | `pm_sampler2` (`pm_exp.py`) | 1.04 M | same node, same protocol |
  | template, timed from the start | 0.06 M | 6% acceptance instead of 0.1%, so lists are rebuilt all the time |

  `bench` gained `--burn`, `--ladder` and `--kT a,b,...` for this.

**Step 4, part 2 (2026-11-03): `jax_mc.run`.**

- **Replica exchange:** each ladder stays a permutation of its rungs at
  every frame (mzIdentML, 32 chains, 4 rungs, 444 swaps).
- **Snapshots:** energies recomputed from saved snapshots match C++ at the
  stored degrees of freedom. The relative difference is 1e-8 on
  mzIdentML. On Pol II it is 2e-3, the known EM pair-list slack: about
  2 kT, as in the term check.
- **Restarts:**
  - mzIdentML, CPU, float64: a run stopped at frame 20 and continued with
    `--cpi` gives byte-identical records and snapshots to a straight
    40-frame run.
  - Pol II, CPU and GPU: snapshots are byte-identical between separate
    runs and across a restart (GPU: 4,096 chains, 24 frames). The recorded
    energies can differ in the last bit: one float32 unit in the last
    place, in 224 of 98,304 values on the GPU.
  - Two identical processes differ like this too, even with the same
    compiled executables. `--xla_gpu_deterministic_ops` doesn't change it.
    Rescores also refresh the EM caches, so a later decision could in
    principle flip after one. That is at the 1e-7 level.
- **Mature NPC through the driver** (collective template, 512 chains on
  the protocol's ladder, 30 frames of 1,000 steps): 5,084 swaps, and the
  energy at the lowest rung went from 18,474 to 6,181.

**Things learned:**
- **PMI builds depend on Python's hash seed.** The order of cross-link
  restraints and contributions, the orientation of pairs, and the last
  bits of some setup values (mature NPC EM covariances) all change from
  one process to the next.
  - `CrossLinks` now orders its restraints and contributions canonically.
  - `jax_mc.run` pins `PYTHONHASHSEED=0`.
- **JAX only caches compiles slower than 1 s by default.** The driver
  caches every program.

**RMF output (2026-11-03).** Files written from snapshots were read back
with `IMP.rmf`. Their coordinates match the snapshot positions to RMF's
float32 precision:

| system | max difference |
|---|---|
| Pol II | 9e-6 Å |
| mzIdentML | 1e-6 Å |
| mature NPC | 6e-5 Å |

**One process per GPU (2026-11-03).**
- Two processes of 16 chains give exactly the records (energies,
  temperatures) and snapshots of one 32-chain run: mzIdentML, CPU,
  float64, replica exchange.
- Two concurrent processes with Open MPI's rank variables each took their
  own GPU and chain offset: Pol II, 2 × 4,096 chains.
