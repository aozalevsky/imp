"""Many-movers template: the engine's sweep, unrolled.

For SerialMover systems with generic PMI terms and no structure to exploit
beyond that (mzIdentML, the NPC basket): the core engine's terms, groups
and proposals, with recipe R2's unrolled sweep and no limit on the number
of movers.

The engine picks each mover's step with a `lax.switch` inside a `lax.scan`.
On a GPU that switch costs about 25 us per MC step (the branch index is
read back from the device), and every table lookup by mover is a kernel of
its own. Unrolled, each mover is a step with constant tables: no switch,
no lookups, and positions updated in place. Compile time then grows with
the number of movers instead of the number of groups (NPC basket, 354
movers: about 5 minutes on one L40, against 1.5 for the engine). The run
driver keeps compiled programs in its cache.

  python -m jax_mc.check decisions npc \\
      --sampler templates.many_movers.sampler:factory \\
      --chains 1024 --ref-chains 256 --kT 1,10
"""
from recipes.r2_unrolled_sweep import UnrolledEngine


class ManyMovers(UnrolledEngine):
    MAX_MOVERS = 100000         # compile time is the limit, not correctness


def factory(s):
    return ManyMovers(s)
