# Speed: C++ IMP against jax_mc

`speed.png` and `speed.csv` compare, for each system jax_mc runs:
- **C++ release** and **C++ fast**: IMP's own `MonteCarlo` on the system's
  restraints and movers, one chain on one core, with the IMP nightly's
  release build (usage checks on) and fast build (checks compiled out). The
  plot uses the faster of `score_moved` off and on whose energy still
  matches a full rescore (on, for every system but the mature NPC, which is
  timed with it off: every step there moves all 112 bodies, so on has
  nothing to skip; setup, one warm-up step and one timed step did not
  finish within 15 minutes).
- **JAX, 1 CPU core**: jax_mc's fastest sampler for the system, one chain,
  float32, single-threaded XLA.
- **JAX, 1 GPU, 1 chain** and **best chain count**: the same sampler on one
  GPU, with one chain and with the chain count that gave the most MC
  steps/s (the counts tried are in `run.sh`).

The samplers: `templates/few_bodies_beads` (Pol II), `templates/collective`
(mature NPC), and the generic engine (NPC basket, mzIdentML). The mature
NPC's MC step moves all 112 bodies, in C++ and in JAX; it is timed after
20,000 steps of burn-in in both.

One GPU chain is slower than one CPU core for the generic engine: a lone
chain leaves the GPU waiting on kernel launches. The GPU pays off with
many chains. Pol II and the mature NPC lose speed past their best chain
count (all counts are in `results.jsonl`).

Reproduce (from `prototypes/jax_mc`, on a GPU node):

```
ENV_RELEASE='...' ENV_FAST='...' perf/run.sh OUT
python -m perf.one_vs_many plot OUT/results.jsonl OUT
```

`run.sh` says what the two environments need. `results.jsonl` here holds
the raw measurements, with versions and devices.
