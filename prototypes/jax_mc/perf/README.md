# Speed: C++ IMP against jax_mc

`speed.png` and `speed.csv` compare, for each system jax_mc runs, first
what one CPU core does (C++, the JAX core engine, the JAX template), then
what one GPU does (the core engine with one chain and at its best chain
count, the template at its best chain count):
- **C++ release** and **C++ fast**: IMP's own `MonteCarlo` on the system's
  restraints and movers, one chain on one core, with the IMP nightly's
  release build (usage checks on) and fast build (checks compiled out). The
  plot uses the faster of `score_moved` off and on whose energy still
  matches a full rescore (on, for every system but the mature NPC, which is
  timed with it off: every step there moves all 112 bodies, so on has
  nothing to skip; setup, one warm-up step and one timed step did not
  finish within 15 minutes).
- **JAX core, 1 CPU core** and **JAX core, 1 GPU, 1 chain**: the core
  engine alone (`jax_mc.engine.make`: `Engine`, or `CollectiveEngine` for
  the mature NPC) with one chain, float32; on the CPU single-threaded XLA.
- **JAX core, 1 GPU, best chain count**: the core engine at the chain
  count that gave the most MC steps/s.
- **JAX template, 1 CPU core** and **JAX template, 1 GPU, best chain
  count**: the system's template, built on the core with its recipes:
  `templates/many_movers` (mzIdentML and the NPC basket: the engine with
  its sweep unrolled, recipe R2), `templates/collective` (mature NPC, its
  own neighbour lists, closed-form overall position, EM pair tables and
  membrane candidates) and `templates/few_bodies_beads` (Pol II, recipes
  R1-R7).

The chain counts tried are in `run.sh`. The mature NPC's MC step moves all
112 bodies, in C++ and in JAX. Its C++ and template runs are timed after
20,000 steps of burn-in: the template's neighbour lists rebuild constantly
from the starting structure. Its core engine scores every pair of its 32,256
beads at every step, so its speed doesn't depend on the state, and it runs
without burn-in. That is also why it is slower than C++ on one CPU core
(0.41 against 99 MC steps/s): C++ scores only close pairs, and so does the
template, which does 1,612 on one CPU core.

For mzIdentML, the NPC basket and Pol II, the core engine is slower with
one chain on the GPU than on one CPU core: a lone chain leaves the GPU
waiting on kernel launches. The GPU pays off with many chains, up to the
count where the arrays a step reads no longer stay in the GPU's cache
(8,192 chains for the NPC basket); all counts are in `results.jsonl`.
Unrolling (`many_movers`) is a GPU technique: on one CPU core the
mzIdentML template is slower than the core engine, and the NPC basket's
has no bar (its 354 unrolled movers didn't finish compiling on one core
within 39 minutes).

The C++, mature NPC and Pol II template records are from 2026-11-03; the
core-engine records of mzIdentML, the NPC basket and Pol II and the
`many_movers` records are from 2026-11-09, after the changes described in
DESIGN.md ("Speed pass over the core").

Reproduce (from `prototypes/jax_mc`, on a GPU node):

```
ENV_RELEASE='...' ENV_FAST='...' perf/run.sh OUT
python -m perf.one_vs_many plot OUT/results.jsonl OUT
```

`run.sh` says what the two environments need. `results.jsonl` here holds
the raw measurements, with versions and devices.

## One node against one node

`speed.png` compares one CPU core with one GPU. `node_results.jsonl`
compares whole nodes, for the NPC basket and the mature NPC: C++ IMP with
one process on every hardware thread against the templates on every GPU.
MC steps/s on one node with an AMD EPYC 7543P (32 cores, 64 threads) and
four NVIDIA L40s, IMP nightly 20261110, 2026-11-10:

| system | C++, 1 thread | C++, 32 processes | C++, 64 processes | JAX, 1 GPU | JAX, 4 GPUs | 4 GPUs / 64 threads |
|---|---|---|---|---|---|---|
| NPC basket | 2,673 | 85,088 | 108,582 | 75.0 M | 299.8 M | 2,761 |
| mature NPC | 181 | 3,720 | 4,323 | 1.355 M | 5.428 M | 1,256 |

- **One common window.** `perf/node.py` starts N processes, lets each set
  up and burn in, then starts them together and counts the steps each
  finishes in 60 s (90 s for the mature NPC's 32 and 64 C++ processes).
  The node's speed is the sum.
- **C++:** the fast build, one single-threaded `MonteCarlo` process pinned
  to each core (32) or each hardware thread (64), as `one_vs_many.py cpp`
  builds it: `score_moved` on for the NPC basket, off for the mature NPC,
  which also burns in for 20,000 steps.
- **JAX:** `templates/many_movers` with 8,192 chains per GPU (NPC basket)
  and `templates/collective` with 1,024 (mature NPC, after 20,000 steps of
  burn-in), float32, one process per GPU.
- **The factor is throughput.** The GPU node runs 32,768 (4,096) chains at
  once and the CPU node 64; the steps are summed over them. Per chain the
  GPU does 9,150 (1,325) MC steps/s, a C++ process 1,697 (67.5) when all
  64 run and 2,673 (181) alone.
- **Four GPUs do 4.0 times one.** The CPU node does not scale that way: 64
  processes do 41 (24) times one thread. The mature NPC is bound by memory
  there; already 32 processes do only 21 times one.
- **The mature NPC template is faster here than in `speed.png`** (1.36 M
  against 1.03 M per L40). `one_vs_many.py` times the first 2,500 steps
  after burn-in; over 60 s fewer moves are accepted, so fewer neighbour
  lists are rebuilt.
- One-thread C++ speeds varied by about 6% between runs that day (2,673 to
  2,850 for the NPC basket).

`node_results.jsonl` has one record per run. Its first one-thread C++
records (`"cpus": "24"` and `"28"`, fast and release build) ran while the
GPU processes were compiling; the table uses the later ones on the idle
node (`"cpus": "8"`). The records were taken with `perf/node.py` before it
moved into this package (the same code, started by path instead of
`-m perf.node`).

Reproduce (from `prototypes/jax_mc`, on an idle GPU node):

```
ENV_RELEASE='...' ENV_FAST='...' perf/node.sh OUT
```

`perf/slide/` builds a slide from these records.
