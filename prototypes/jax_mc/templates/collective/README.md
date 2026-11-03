# Collective template

**Fits:** systems where IMP's MonteCarlo proposes all its movers at once,
with one rigid-body mover per body. Example: the mature NPC refinement of
PM_NPC_Assembly (112 bodies, 32,256 EV beads).

**Starts from:** `pm_sampler2.py` (PM_NPC_Assembly bench), moved onto
jax_mc's `System` and terms. It reuses the terms' own energy functions for
the membrane and bounding-box terms, and the reference's proposals
(`CollectiveEngine.propose`).

**Recipes** (preconditions asserted in `CollectiveNL.__init__`):

| recipe | what it does | precondition |
|---|---|---|
| R8 | EV over an outer (30 Å) and an inner (8 Å) neighbour list, rebuilt per chain inside the step | half the inner skin exceeds the largest one-step displacement of any bead |
| OP | overall position in closed form per body | tolerance 0; reference points rigid within each body |
| EM | isd EM pair tables, candidates within cutoff + skin/2 | static model side, frozen covariances, no slope |
| MEM | membrane depth over beads near the membrane | (rebuilt with the inner list) |

**Check** against the reference, in float64:

```
python -m jax_mc.check decisions pm \
    --sampler templates.collective.sampler:factory \
    --chains 64 --ref-chains 4 --kT 30,300
```

**Benchmark:** `python -m jax_mc.bench pm --sampler
templates.collective.sampler:factory --chains 512`.
