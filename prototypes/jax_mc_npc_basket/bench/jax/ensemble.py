"""Sampling equivalence: independent C++ and JAX chains from one identical
starting configuration (the REX2 start after --frames1 C++ REX1 frames,
seed 1), at T=1, recording the score after every PMI frame.

  python ensemble.py cpp --system yeast --chains 0-3 --frames 10 \\
      --out ens.jsonl
  JAX_PLATFORMS=cuda python ensemble.py jax --system yeast --nchains 64 ...

JAX runs with cov_mode='frozen' by default, i.e. EM model covariances fixed
at their cached values like C++ (see npc_sampler.py), so both sample the
same target (up to the EM close-pair slack band).
"""
import argparse
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['cpp', 'jax'])
ap.add_argument('--system', default='yeast')
ap.add_argument('--frames1', type=int, default=20)
ap.add_argument('--frames', type=int, default=10)
ap.add_argument('--mc-steps', type=int, default=20)
ap.add_argument('--chains', default='0-3', help='C++: chain ids a-b')
ap.add_argument('--nchains', type=int, default=64, help='JAX: chains')
ap.add_argument('--cov-mode', default='frozen')
ap.add_argument('--out', required=True)
ap.add_argument('--key', type=int, default=7)
ap.add_argument('--sampler', type=int, default=1, choices=[1, 2])
a = ap.parse_args()

import numpy as np  # noqa: E402
import IMP  # noqa: E402
import IMP.pmi.samplers  # noqa: E402

import imp_jax_npc  # noqa: E402
from npc_setup import build  # noqa: E402

imp_jax_npc.install()
S = build(a.system, frames1=a.frames1, seed=1)
m, rs = S.model, S.rs
nmov = None
start = m.get_spheres_numpy()[0].copy()


def emit(row):
    with open(a.out, 'a') as fh:
        fh.write(json.dumps(row) + '\n')


if a.mode == 'cpp':
    import IMP.core
    from IMP.core import _jax_rigid
    lo, hi = (int(x) for x in a.chains.split('-'))
    # save the full start state (coordinates, rigid-body frames, nuisances)
    xyz0 = m.get_spheres_numpy()[0].copy()
    q0 = m.get_numpy(_jax_rigid._RB_QUAT_KEY).copy()
    nk = IMP.isd.Scale.get_scale_key()
    n0 = m.get_numpy(nk).copy()
    for c in range(lo, hi + 1):
        m.get_spheres_numpy()[0][:] = xyz0
        m.get_numpy(_jax_rigid._RB_QUAT_KEY)[:] = q0
        m.get_numpy(nk)[:] = n0
        m.update()
        IMP.random_number_generator.seed(1000 + c)
        for mv in S.movers:
            mv.reset_statistics()
        mc = IMP.pmi.samplers.MonteCarlo(m, S.movers, 1.0, score_moved=True)
        mc.mc.set_scoring_function([rs])
        scores = [rs.evaluate(False)]
        t0 = time.time()
        for f in range(a.frames):
            mc.optimize(a.mc_steps)
            scores.append(rs.evaluate(False))
        acc = {}
        for mv in mc.smv.get_movers():
            d = mv.get_derived_object()
            k = type(d).__name__
            p, r = d.get_number_of_proposed(), d.get_number_of_accepted()
            acc.setdefault(k, [0, 0])
            acc[k][0] += r
            acc[k][1] += p
        emit(dict(mode='cpp', system=a.system, chain=c, scores=scores,
                  acc={k: v[0] / max(1, v[1]) for k, v in acc.items()},
                  t=time.time() - t0))
        print('chain', c, scores[-1], flush=True)
else:
    import jax
    import jax.numpy as jnp
    from npc_sampler import NPCSampler
    from npc_sampler2 import NPCSampler2
    smp = (NPCSampler if a.sampler == 1 else NPCSampler2)(
        m, rs, S.movers, cov_mode=a.cov_mode)
    nsteps = a.mc_steps * smp.nmov
    s0 = smp.initial_state(seed=0)
    n = a.nchains
    sb = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s0)
    sb.rkey = jax.random.split(jax.random.key(a.key), n)
    scores = [np.asarray(sb.score)]
    t0 = time.time()
    for f in range(a.frames):
        sb = smp.run_batched(sb, nsteps)
        if (f + 1) % 10 == 0:
            sb = smp.resync_batched(sb)
        scores.append(np.asarray(sb.score))
    scores = np.array(scores)               # (frames+1, n)
    prop = np.asarray(sb.proposed_mover)
    accm = np.asarray(sb.accepted_mover)
    names = [type(d).__name__ for d in smp.movers]
    for c in range(n):
        acc = {}
        for j, k in enumerate(names):
            acc.setdefault(k, [0, 0])
            acc[k][0] += int(accm[c, j])
            acc[k][1] += int(prop[c, j])
        emit(dict(mode='jax', system=a.system, chain=c,
                  scores=[float(x) for x in scores[:, c]],
                  acc={k: v[0] / max(1, v[1]) for k, v in acc.items()},
                  t=time.time() - t0, cov_mode=a.cov_mode,
                  sampler=a.sampler))
    print('done', scores[-1].mean(), flush=True)
