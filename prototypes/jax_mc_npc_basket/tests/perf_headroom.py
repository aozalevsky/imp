"""How far the basket JAX sampler is from the GPU's limits.

1. Time per MC step against the number of chains (one step of all chains
   is one iteration of the sampler's loop): flat at small counts means
   per-step overhead (kernel launches, loop control), linear at large
   counts means the GPU's arithmetic or memory is the limit.
2. Kernels per MC step: GPU kernels (fusions, custom calls, copies) in the
   loop body of the compiled frame, including the mover switch's branches.
3. Work per MC step from XLA's cost analysis, per mover (one compiled
   step with the mover fixed): floating-point operations, transcendental
   functions (exp, log, ...) and bytes read and written, summed over a
   frame (monte_carlo_steps x every mover) and divided by the measured
   frame time.

Run from NPC_Basket/bench/jax:
  JAX_PLATFORMS=cuda python perf_headroom.py --chains 256,512,1024,2048,4096,8192,16384
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('system', nargs='?', default='yeast')
ap.add_argument('--frames1', type=int, default=20)
ap.add_argument('--chains', default='256,512,1024,2048,4096,8192,16384')
ap.add_argument('--hlo-chains', type=int, default=2048)
ap.add_argument('--frames', type=int, default=2)
ap.add_argument('--out', default='perf_headroom.json')
a = ap.parse_args()

sys.path.insert(0, os.getcwd())          # the sampler modules (bench/jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import IMP  # noqa: E402
import IMP.core  # noqa: E402
import IMP.isd  # noqa: E402
import IMP.pmi  # noqa: E402

import imp_jax_npc  # noqa: E402
from npc_setup import build  # noqa: E402
from npc_sampler2 import NPCSampler2  # noqa: E402

imp_jax_npc.install()
S = build(a.system, frames1=a.frames1)
smp = NPCSampler2(S.model, S.rs, S.movers)
mc_steps = S.rex2.vars['monte_carlo_steps']
nsteps = mc_steps * smp.nmov
s0 = smp.initial_state(seed=0)
dev = jax.devices()[0]
res = dict(device=str(dev.device_kind), jax=jax.__version__, nmov=smp.nmov,
           mc_steps=mc_steps, nsteps_per_frame=nsteps, K=smp.K)
print(json.dumps(res), flush=True)


def batch(n):
    sb = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s0)
    sb.rkey = jax.random.split(jax.random.key(1), n)
    sb.temperature = jnp.ones(n, s0.temperature.dtype)
    return sb


# ---- 1. time per step against chains --------------------------------
sweep = []
for n in [int(x) for x in a.chains.split(',')]:
    sb = batch(n)
    sb = smp.run_batched(sb, nsteps)
    sb.score.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(a.frames):
        sb = smp.run_batched(sb, nsteps)
    sb.score.block_until_ready()
    dt = (time.perf_counter() - t0) / a.frames
    row = dict(chains=n, s_per_frame=dt, us_per_step=dt / nsteps * 1e6,
               mc_steps_per_s=n * nsteps / dt)
    sweep.append(row)
    print(json.dumps(row), flush=True)
    del sb
res['sweep'] = sweep

# ---- 2. kernels per MC step ------------------------------------------
sb = batch(a.hlo_chains)
hlo = smp.run_batched.lower(sb, nsteps).compile().as_text()
comps, cur = {}, None
for line in hlo.splitlines():
    m = re.match(r'^(?:ENTRY )?%?([\w.\-]+) .*\{\s*$', line)
    if m and not line.startswith(' '):
        cur = m.group(1)
        comps[cur] = []
    elif cur is not None and line.startswith('  '):
        comps[cur].append(line)
KERNEL = re.compile(r'= [^=]*? (fusion|custom-call|copy|copy-start|'
                    r'dynamic-update-slice|scatter|gather|reduce|sort|'
                    r'convolution|dot|all-reduce)\(')


def kernels(name, seen=()):
    """kernels launched by one run of computation name (fusion bodies
       count as one kernel; while/conditional bodies are expanded)"""
    c = collections.Counter()
    for line in comps.get(name, []):
        m = KERNEL.search(line)
        if m:
            c[m.group(1)] += 1
        if ' while(' in line:
            body = re.search(r'body=%?([\w.\-]+)', line).group(1)
            c['while'] += 1
            c.update({'body:' + k: v for k, v in kernels(body).items()})
        if ' conditional(' in line:
            br = re.search(r'branch_computations=\{([^}]*)\}', line)
            names = [x.strip().lstrip('%') for x in br.group(1).split(',')]
            ks = [sum(kernels(b).values()) for b in names]
            c['conditional'] += 1
            c['branch_kernels_min'] += min(ks)
            c['branch_kernels_max'] += max(ks)
    return c


entry = [k for k in comps if k.startswith('main') or 'ENTRY' in k]
whiles = [(line, n) for n, ls in comps.items() for line in ls
          if ' while(' in line]
loop = []
for line, owner in whiles:
    body = re.search(r'body=%?([\w.\-]+)', line).group(1)
    loop.append(dict(owner=owner, body=body,
                     kernels=dict(kernels(body)),
                     instructions=len(comps.get(body, []))))
res['loops'] = loop
for x in loop:
    print('loop', json.dumps(x), flush=True)
with open(a.out.replace('.json', '_hlo.txt'), 'w') as fh:
    fh.write(hlo)

# ---- 3. work per MC step, per mover -----------------------------------
vstep = jax.vmap(smp._step, in_axes=(0, None, 0, 0))
n = a.hlo_chains
U = jnp.zeros((n, 10), s0.xyz.dtype) + 0.5
Z = jnp.zeros((n,), s0.xyz.dtype)
kind = collections.Counter()
tot = collections.Counter()
per_kind = collections.defaultdict(collections.Counter)
for i, d in enumerate(smp.movers):
    k = type(d).__name__
    kind[k] += 1
    ca = jax.jit(lambda sb, u, z, i=i: vstep(sb, i, u, z)).lower(
        sb, U, Z).compile().cost_analysis()
    if isinstance(ca, list):
        ca = ca[0]
    for key in ('flops', 'transcendentals', 'bytes accessed'):
        tot[key] += ca.get(key, 0.) / n
        per_kind[k][key] += ca.get(key, 0.) / n
res['movers'] = dict(kind)
res['per_chain_per_frame'] = {k: v * mc_steps for k, v in tot.items()}
res['per_chain_per_frame_by_mover'] = {
    k: {kk: vv * mc_steps for kk, vv in v.items()} for k, v in per_kind.items()}
row = [r for r in sweep if r['chains'] == n] or sweep[-1:]
fr = row[0]
res['achieved_at'] = dict(
    chains=fr['chains'],
    flop_per_s=res['per_chain_per_frame']['flops'] * fr['chains']
    / fr['s_per_frame'],
    transcendentals_per_s=res['per_chain_per_frame']['transcendentals']
    * fr['chains'] / fr['s_per_frame'],
    bytes_per_s=res['per_chain_per_frame']['bytes accessed'] * fr['chains']
    / fr['s_per_frame'])
print(json.dumps({k: res[k] for k in ('movers', 'per_chain_per_frame',
                                      'achieved_at')}), flush=True)
with open(a.out, 'w') as fh:
    json.dump(res, fh, indent=1)
print('PERF_DONE', flush=True)
