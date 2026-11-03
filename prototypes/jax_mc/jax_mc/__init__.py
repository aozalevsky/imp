"""jax_mc: a core for fast, exact JAX Monte Carlo samplers of IMP systems.
See ../DESIGN.md."""


def xla_defaults():
    """XLA flags for jax_mc programs; call before JAX starts its backend.
       Autotuning is off: on GPU it took most of the compile time and grew
       with the number of chains (NPC basket, 4,096 chains: 56 s against
       8.5 s), with no effect on speed."""
    import os
    f = os.environ.get('XLA_FLAGS', '')
    if 'xla_gpu_autotune_level' not in f:
        os.environ['XLA_FLAGS'] = (f + ' --xla_gpu_autotune_level=0').strip()
