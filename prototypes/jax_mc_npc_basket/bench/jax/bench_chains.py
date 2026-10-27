"""Throughput of the fused sampler with many chains in one process
(jax.vmap over chains; e.g. replicas of a REX ladder or independent runs).

  JAX_PLATFORMS=cuda python bench_chains.py yeast --chains 1,8,32,128,512
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('system', nargs='?', default='yeast')
ap.add_argument('--frames1', type=int, default=2)
ap.add_argument('--chains', default='1,8,32,128')
ap.add_argument('--frames', type=int, default=3)
ap.add_argument('--mc-steps', type=int, default=20)
ap.add_argument('--tag', default='')
ap.add_argument('--mode', default='shared', choices=['shared', 'select'],
                help='shared: one mover switch for all chains (default); '
                     'select: vmap of the single-chain run')
ap.add_argument('--json', default=None)
ap.add_argument('--sampler', type=int, default=1, choices=[1, 2])
ap.add_argument('--ablate', default='',
                help='timing only: comma list of ev,em,cheap,xl,leaves,hub')
a = ap.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import IMP  # noqa: E402

import imp_jax_npc  # noqa: E402
from npc_setup import build  # noqa: E402
from npc_sampler import NPCSampler  # noqa: E402
from npc_sampler2 import NPCSampler2  # noqa: E402

imp_jax_npc.install()
S = build(a.system, frames1=a.frames1)
smp = (NPCSampler if a.sampler == 1 else NPCSampler2)(S.model, S.rs, S.movers,
                 ablate=[x for x in a.ablate.split(',') if x])
nsteps = a.mc_steps * smp.nmov
s0 = smp.initial_state(seed=0)
ap_mode = a.mode
if ap_mode == 'select':
    # naive: vmap the single-chain run; the mover switch becomes a select
    run_b = jax.jit(jax.vmap(smp._run_py, in_axes=(0, None)),
                    static_argnums=1)
else:
    run_b = smp.run_batched
print(f'{jax.default_backend()} {jax.devices()[0]}; {nsteps} steps/frame')

for n in [int(x) for x in a.chains.split(',')]:
    keys = jax.random.split(jax.random.key(1), n)
    temps = jnp.geomspace(1.0, 4.0, n).astype(s0.temperature.dtype) \
        if n > 1 else jnp.ones(1, s0.temperature.dtype)
    sb = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s0)
    sb.rkey = keys
    sb.temperature = temps
    t0 = time.time()
    sb = run_b(sb, nsteps)
    sb.score.block_until_ready()
    t_first = time.time() - t0
    t0 = time.perf_counter()
    for f in range(a.frames):
        sb = run_b(sb, nsteps)
    sb.score.block_until_ready()
    dt = time.perf_counter() - t0
    tot = n * a.frames * nsteps
    row = dict(tag=a.tag, sampler=a.sampler, mode=a.mode, ablate=a.ablate,
               backend=jax.default_backend(),
               chains=n,
               t_first=t_first, us_per_step_per_chain=dt / tot * 1e6,
               steps_per_s=tot / dt, ms_per_frame=dt / a.frames * 1e3,
               score_min=float(sb.score.min()),
               score_t1=float(sb.score[0]),
               nan=int(jnp.isnan(sb.score).sum()))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in row.items()}), flush=True)
    if a.json:
        with open(a.json, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
