# Many-movers template

**Fits:** `SerialMover` systems with generic PMI terms and no structure to
exploit beyond what the core already does: many movers of many shapes
(rigid bodies, beads, a `TransformMover`, nuisances). Examples: the
mzIdentML tutorial (16 movers) and the NPC basket (354 movers).

**What it is:** the core engine (its terms, mover groups, proposals and
draws) with recipe R2's unrolled sweep and no limit on the number of
movers. Nothing in it is specific to a system or tuned to a GPU.

**Why it is faster.** The engine picks each mover's step with a
`lax.switch` inside a `lax.scan`. On a GPU that costs about 25 µs per MC
step (the branch index is read back from the device before the step's
kernels can be queued), and every table lookup by mover index is a kernel
of its own. Unrolled, each mover is a step whose tables are constants: no
switch, no lookup kernels, and the positions are updated in place.

**What it costs:** compile time, which grows with the number of movers
instead of the number of mover groups. The run driver keeps compiled
programs in its cache.

| system | movers | engine | this template |
|---|---|---|---|
| mzIdentML | 16 | 4 s | 6 s |
| NPC basket | 354 | 1.5 min | 5.5 min |

**Recipes** (preconditions asserted in `UnrolledEngine.__init__`):

| recipe | what it does | precondition |
|---|---|---|
| R2 | unrolled sweep, in-place position updates behind `optimization_barrier` | `s.mode == 'serial'`; a mover whose rows aren't one slice (the NPC basket's `TransformMover`s) takes the engine's own step, with constant tables |

**Check** against the reference engine, in float64:

```bash
python -m jax_mc.check decisions mzid \
    --sampler templates.many_movers.sampler:factory \
    --chains 1024 --ref-chains 256 --kT 1,10
python -m jax_mc.check decisions npc \
    --sampler templates.many_movers.sampler:factory \
    --chains 1024 --ref-chains 256 --kT 1,10
```

Energies and positions come out equal to the engine's, not only the
decisions: the arithmetic is the same.

**Benchmark** (one NVIDIA L40, float32):

```bash
python -m jax_mc.bench mzid \
    --sampler templates.many_movers.sampler:factory \
    --chains 32768,65536,131072,262144
```

| system | chains | engine (MC steps/s) | this template (MC steps/s) |
|---|---:|---:|---:|
| mzIdentML | 65,536 | 964 M | 2.35 bn |
| mzIdentML | 131,072 | | **2.64 bn** |
| mzIdentML | 262,144 | **1.19 bn** | 1.95 bn |
| NPC basket | 4,096 | | 57.6 M |
| NPC basket | 8,192 | **54.3 M** | **75.1 M** |
| NPC basket | 12,288 | 53.5 M | 54.6 M |

On one CPU core unrolling doesn't pay: mzIdentML does 764 k MC steps/s
with this template and 942 k with the engine, and the NPC basket's 354
unrolled movers didn't finish compiling on one core within 39 minutes.
On the GPU, the NPC basket at 16,384 chains didn't finish its timing
within 20 minutes either (not looked into; 12,288 is already past the
best count).

**Where to go from here.** A step is still about 15 to 30 kernels, one
after the other, and at the best chain counts the GPU waits for them to be
launched. Scoring several beads in one kernel (recipe R3, as in
`few_bodies_beads`) is the next step for the NPC basket; its beads carry
EM Gaussians and ambiguous cross-links, which R3 doesn't cover yet.
