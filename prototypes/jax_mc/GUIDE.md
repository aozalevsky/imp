# Make a fast sampler for your system

This page is for a person or an agent with a PMI/IMP system who wants it
sampled fast on a GPU, with the same decisions as IMP. It walks through:
1. getting the system into jax_mc and checking its energies against C++;
2. running it on the generic engine, which is often fast enough;
3. if it isn't, specializing it one checked step at a time.

DESIGN.md explains why things are the way they are. This page only says what
to do. `perf/speed.png` shows what the four systems so far gain over C++ IMP.

## 0. Setup

Everything runs from this directory (`prototypes/jax_mc/`), with IMP and
JAX in the environment. Work on a GPU node, not a workstation.

Every tool below takes a system name from `jax_mc/sources.py` (`polii`,
`npc`, `pm`, `mzid`). The commands are written for `mysys`, your system's
name once you've added it (step 1).

## 1. Get the system in

**Add a loader** to `jax_mc/sources.py`. It calls your existing PMI script's
builder and returns a `Source`:
- the IMP model;
- the top-level restraints (restraint sets are fine);
- the movers, in the order IMP's SerialMover applies them;
- `mode`: `'serial'` (one mover per MC step, PMI's default) or `'all'`
  (IMP's MonteCarlo applying every mover at every step);
- the kT, the replica-exchange ladder if the protocol has one, and the root
  hierarchy (for RMF output).

Register it in `LOADERS`. Build the system exactly as the C++ protocol
does: the loader should call the protocol's code, not copy it.

**Check the energies against C++:**

```
python -m jax_mc.check terms mysys
```

It compares every term with IMP's `evaluate` at random states, in float64.
Each term should agree to round-off (relative 1e-10 or better).

**If extraction raises `Unsupported`:**
- **A restraint:** either add a term class (`jax_mc/terms/`; copy the
  closest existing one and see the interface at the top of
  `terms/__init__.py`), or rely on the IMP-JAX fallback. The fallback
  covers any restraint that has an IMP JAX implementation, but it rescores
  the whole restraint at every move, so it's slow.
- **A mover:** the engine supports BallMover on one particle,
  RigidBodyMover, TransformMover on a mix of beads and bodies, and
  NormalMover on one Scale. Anything else needs code in
  `system._movers` and in the engine; ask before adding it.

## 2. Run the generic engine

**Check it** against the full-rescore reference. It should report 0
decisions differ:

```
python -m jax_mc.check decisions mysys --chains 1024 --ref-chains 64 \
    --sweeps 20 --kT 1,10
```

For a `mode='all'` system the reference is `CollectiveEngine`, which
rescores everything at every step. It's the specialized samplers' reference,
not something to run in production. Go to step 3.

**Measure it:**

```
python -m jax_mc.bench mysys --chains 1024,4096,16384 --kT 1
```

It prints MC steps/s and the acceptance rate. Report speeds in MC steps/s
only. Use `--ladder` to bench at the protocol's temperatures, and `--burn N`
when the start isn't representative (a collective sampler starts with many
rejected steps and rebuilds).

**Run it:**

```
python -m jax_mc.run mysys --out OUT --chains 4096 --frames 1000 \
    --sweeps 10 --ladder --thin 10 --cpt 15
python -m jax_mc.rmf OUT --best 20
```

That's replica exchange at the protocol's temperatures, with checkpoints
every 15 minutes and the 20 best snapshots written as RMF. Restart with
`--cpi`. To use several GPUs, start one process per GPU: `--ranks N --rank r`,
or under mpirun or srun. Together they are exactly one run.

**If the engine is fast enough, stop here.** It beat the hand-written
samplers on the NPC basket and on mzIdentML.

## 3. Specialize, one checked step at a time

### Find where the time goes

Before writing anything, profile one sweep:
- `jax.profiler.trace` around a few `run` calls, then look at the largest
  kernels;
- `python -m jax_mc.bench mysys --hlo` counts the array copies in the
  compiled loop. There should be none.

What was found on the systems so far:

| system | where the time went |
|---|---|
| NPC basket | EV 30%, isd EM 32%, cheap terms 23%, overhead 22% |
| mature NPC | neighbour-list rebuilds 28%; the rest is the full energy |
| Pol II | kernel launches per sweep (a few cheap movers on many chains) |

### Pick a starting point

| your system | start from | level |
|---|---|---|
| one term dominates (EV, EM) | a faster subclass of that term | 1 |
| one kind of move dominates (many beads) | an `Engine` subclass for that group's branch | 2 |
| few movers, cheap terms, launch-bound | `templates/few_bodies_beads` (Pol II: 1.2 bn on one L40, 4.5× the engine) | 3 |
| every mover moves at every step | `templates/collective` (mature NPC) | 3 |

Each template's README says what system shape it fits, what it asserts,
and which numbers are tuned to a GPU. `recipes/` has one page per technique
(R1-R7 so far), with its level, preconditions and measured gain; most come
from `few_bodies_beads`.

**Level 1: a faster term.** Subclass the term, override `cache`,
`move_tables` and `delta`, and build the system with it:

```python
s = system.extract(src, replace={SoftSphereEV: MyEV})
```

The engine, the checks and the driver don't change. A term's `delta` only
needs to be right for the moves it's called with. The engine tells it which
rows move, and that they move rigidly.

**Level 2: a faster branch.** Subclass `Engine` and override
`_branch(G)` for some groups (`G.kind`, `G.movers`, `G.tab`). Keep the
state layout and the draws.

**Level 3: your own sampler.** Copy a template and follow the protocol
below. The state layout is yours.

### The protocol

`check`, `bench` and `run` need only this:

```python
class MySampler:
    s                                   # the jax_mc System
    def init(self, C, seed=0, kT=None, dofs=None, chain0=0): ...
    def run(self, st, n_sweeps, record=False): ...
    def dofs(self, st): ...             # the System's degrees of freedom
    def total(self, st): ...            # (C,) tracked total energy
    def resync(self, st): ...           # optional: energies from scratch
    def stats(self, st): ...            # optional: dict of numbers to print

def factory(s):
    return MySampler(s)
```

- `init` returns any pytree. `jax_mc.run` needs a `'kT'` entry of shape
  (C,), and chains `chain0` to `chain0 + C - 1` of the whole run.
- `run(..., record=True)` also returns the decisions: (n_sweeps, n_movers,
  C), or (n_sweeps, 1, C) for collective steps.
- Pass it to any tool with `--sampler mymodule:factory`.

### The loop

For every change:
1. Make one change.
2. Check it against the engine in float64, at a small and a large chain
   count, at a low kT and at one where about half the moves are accepted:

   ```
   python -m jax_mc.check decisions mysys --sampler mymodule:factory \
       --chains 1024 --ref-chains 64 --sweeps 20 --kT 1,10
   python -m jax_mc.check decisions mysys --sampler mymodule:factory \
       --chains 8192 --ref-chains 64 --sweeps 20 --kT 1,10
   ```

   0 decisions should differ. In float64 a differing decision is almost
   always a bug, not round-off: look at the first chain and sweep where it
   differs.
3. Check float32: `check decisions ... --float32` reports flipped decisions
   and the drift of tracked against rescored energies. It should be close to
   the engine's.
4. Benchmark. If it isn't faster, drop the change.

### Rules that keep the check meaningful

- **Same random numbers.** Use `Engine.draws(st)` or `engine.draws`, and
  the proposal formulas in DESIGN.md ("Random numbers"). Slicing the draws
  is fine. Generating fewer is not, since the numbers would change.
- **Same proposals.** Reuse the engine's proposal code where possible
  (`CollectiveEngine.propose`, the branch formulas). A different but
  equivalent formula can flip a decision when an energy difference lands
  exactly at the edge.
- **Assert every precondition.** A shortcut that holds only for some
  systems (covariances that rotate, static particles, bead order, bounded
  displacements) asserts its condition in `__init__`. If the condition can
  break at run time, as with neighbour lists, flag it in `stats` (count
  overflowed chains) and make the energy NaN rather than wrong.
- **No system-specific constants.** Derive thresholds and indices from the
  System, or take them as arguments with the values you measured.

## 4. Pitfalls already met

- **Float32 sums that cancel.** isd EM's log of a sum drifted by 3 kT with
  running sums. Recomputing blocks of 16 from positions fixed it.
  Large energies (1e8 on the NPC basket) drift by about 200 in float32.
  Decisions use differences, so that's harmless, but rescore every frame
  (`--rescore-every 1`).
- **Compile time growing with chains:** XLA's GPU autotuning.
  `jax_mc.xla_defaults()` turns it off. Call it before importing JAX in any
  new script.
- **PMI builds depend on Python's hash seed** (set order). The tools pin
  `PYTHONHASHSEED=0`. Any new script that builds a system should call
  `jax_mc.pin_hash_seed()` first.
- **Cold starts.** Collective samplers rebuild their lists constantly from
  the starting structure, so benchmark after `--burn`.
- **Reproducibility.** Positions are bitwise reproducible between runs and
  restarts. Rescored energies can differ by one unit in the last place
  between runs, even with the same compiled programs.
- **bf16 and TF32** don't pay: about 6 kT error and 6% flipped decisions,
  with no speedup. Stay in float32.

## 5. Give it back

When a change is worth reusing on other systems, write it up as a recipe
in `recipes/`:
- what it does and at which level;
- its preconditions, each asserted in the code;
- the measured gain (system, GPU, chains, MC steps/s);
- the `check` command that proves it.

Add that command to `tests/check_all.sh`.
