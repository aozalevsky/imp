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


def pin_hash_seed():
    """Re-run this process with PYTHONHASHSEED=0 unless it is set. PMI
       builds systems by iterating over sets (cross-link contributions,
       connectivity pairs, rigid-body setup), so with Python's default
       random hash seed two builds differ in order and in the last bits of
       some setup values, and a restarted run would not continue the
       checkpointed one exactly."""
    import os
    import sys
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '0'
        spec = getattr(sys.modules.get('__main__'), '__spec__', None)
        args = ['-m', spec.name] + sys.argv[1:] if spec else sys.argv
        os.execv(sys.executable, [sys.executable] + args)
