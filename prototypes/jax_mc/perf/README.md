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
