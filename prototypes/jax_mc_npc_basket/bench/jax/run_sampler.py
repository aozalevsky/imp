"""Run the fused JAX sampler on NPC_Basket REX2 and check it against C++.

Builds the system (REX1 in C++ for --frames1 frames), creates NPCSampler
from the REX2 scoring function and PMI's movers, then runs --frames PMI
frames (one frame = --mc-steps x n_movers MC steps, as PMI does). After
each frame (with --check) the state is copied into the IMP Model and
compared with C++ (rs.evaluate) and with a from-scratch JAX evaluation.

  JAX_PLATFORMS=cpu python run_sampler.py yeast --frames 10 --check
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('system', nargs='?', default='yeast')
ap.add_argument('--frames1', type=int, default=2)
ap.add_argument('--frames', type=int, default=5)
ap.add_argument('--mc-steps', type=int, default=20,
                help='PMI monte_carlo_steps (x n_movers per frame)')
ap.add_argument('--check', action='store_true')
ap.add_argument('--x64', action='store_true')
ap.add_argument('--cov-mode', default='rotate', choices=['rotate', 'frozen'])
ap.add_argument('--em-extra', type=float, default=2.0)
ap.add_argument('--no-switch', action='store_true')
ap.add_argument('--resync-every', type=int, default=10)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--tag', default='')
ap.add_argument('--sampler', type=int, default=1, choices=[1, 2],
                help='1: npc_sampler (full candidate state), 2: npc_sampler2'
                     ' (delta-based)')
ap.add_argument('--ablate', default='',
                help='timing only: comma list of ev,em,cheap to skip')
ap.add_argument('--cpp-frames', type=int, default=0,
                help='also time C++ MC (PMI sampler, score_moved=True) '
                     'for this many frames from the same start')
ap.add_argument('--json', default=None)
a = ap.parse_args()

import jax  # noqa: E402
if a.x64:
    jax.config.update('jax_enable_x64', True)
import IMP  # noqa: E402

import imp_jax_npc  # noqa: E402
from npc_setup import build  # noqa: E402
from npc_sampler import NPCSampler  # noqa: E402
from npc_sampler2 import NPCSampler2  # noqa: E402

imp_jax_npc.install()
t0 = time.time()
S = build(a.system, frames1=a.frames1)
t_build = time.time() - t0
m, rs = S.model, S.rs
t0 = time.time()
smp = (NPCSampler if a.sampler == 1 else NPCSampler2)(
    m, rs, S.movers, temperature=a.temperature,
    dtype='float64' if a.x64 else 'float32',
    cov_mode=a.cov_mode, em_extra=a.em_extra,
    use_switch=not a.no_switch,
    ablate=[x for x in a.ablate.split(',') if x])
t_setup = time.time() - t0
nsteps = a.mc_steps * smp.nmov
print(f'IMP {IMP.get_module_version()} jax {jax.__version__} '
      f'{jax.default_backend()} {"f64" if a.x64 else "f32"}; build '
      f'{t_build:.1f}s, sampler setup {t_setup:.1f}s')
print(f'  compact rows {smp.K}, movable bodies {smp.nb}, movers {smp.nmov} '
      f'(ball/rb/transform/normal {list(smp.n_kind)}), EV particles '
      f'{smp.nev}, EM model x data {smp.M} x {smp.D}, connectivity pairs '
      f'{smp.n_hub}, other restraints {smp.n_leaves}')

# C++ reference for the starting configuration (fresh covariances)
for p in smp.em_model_pis:
    IMP.core.Gaussian(m, IMP.ParticleIndex(int(p))).update_global_covariance()
e_cpp0 = rs.evaluate(False)
t0 = time.time()
s = smp.initial_state(seed=a.seed)
s.score.block_until_ready()
s0 = s
t_init = time.time() - t0
print(f'start: C++ {e_cpp0:.4f}  JAX {float(s.score):.4f}  (init+resync '
      f'compile {t_init:.1f}s)')

t0 = time.time()
s = smp.run(s, nsteps)
s.score.block_until_ready()
t_first = time.time() - t0
print(f'first frame incl. compile: {t_first:.1f}s')

res = dict(tag=a.tag, system=a.system, backend=jax.default_backend(),
           x64=a.x64, cov_mode=a.cov_mode, nsteps_per_frame=nsteps,
           K=smp.K, nev=smp.nev, M=smp.M, D=smp.D, nmov=smp.nmov,
           t_build=t_build, t_setup=t_setup, t_first=t_first, frames=[])
t_run = 0.0
prev_acc = int(s.accepted)
for f in range(a.frames):
    t0 = time.perf_counter()
    s = smp.run(s, nsteps)
    s.score.block_until_ready()
    dt = time.perf_counter() - t0
    t_run += dt
    row = dict(frame=f, t=dt, score=float(s.score),
               acc=(int(s.accepted) - prev_acc) / nsteps)
    prev_acc = int(s.accepted)
    if a.check:
        tm = jax.device_get(smp.terms(s))
        smp.write_model(s)
        e_cpp = rs.evaluate(False)
        by = {}
        for i in range(rs.get_number_of_restraints()):
            r = rs.get_restraint(i)
            by[r.get_name()] = by.get(r.get_name(), 0.) + r.evaluate(False)
        cheap_cpp = sum(v for k, v in by.items()
                        if k not in ('GaussianEMRestraint',
                                     'ExcludedVolumeSphere'))
        row.update(cpp=e_cpp, d_total=float(s.score) - e_cpp,
                   d_cheap=float(tm['cheap']) - cheap_cpp,
                   d_ev=float(tm['ev']) - by.get('ExcludedVolumeSphere', 0),
                   d_em=float(tm['em']) - by.get('GaussianEMRestraint', 0),
                   em_drift=float(tm['em_running']) - float(tm['em']))
        if 'ev_running' in tm:
            row['ev_drift'] = float(tm['ev_running']) - float(tm['ev'])
        else:
            row['score_drift'] = (float(tm['total_running'])
                                  - float(tm['total']))
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in row.items()}))
    res['frames'].append(row)
    if a.resync_every and (f + 1) % a.resync_every == 0:
        s = smp.resync(s)

n = a.frames * nsteps
res['us_per_step'] = t_run / n * 1e6
res['steps_per_s'] = n / t_run
if a.cpp_frames:
    import IMP.pmi.samplers
    smp.write_model(s0, refresh_covariances=False)
    mc = IMP.pmi.samplers.MonteCarlo(m, S.movers, a.temperature,
                                     score_moved=True)
    mc.mc.set_scoring_function([rs])
    mc.optimize(1)
    t0 = time.perf_counter()
    for f in range(a.cpp_frames):
        mc.optimize(a.mc_steps)
    t_cpp = time.perf_counter() - t0
    res['cpp_us_per_step'] = t_cpp / (a.cpp_frames * nsteps) * 1e6
    res['cpp_steps_per_s'] = a.cpp_frames * nsteps / t_cpp
    res['cpp_score'] = rs.evaluate(False)
    print(f'C++ (this IMP, score_moved): {res["cpp_steps_per_s"]:.0f} '
          f'MC steps/s over {a.cpp_frames} frames; final score '
          f'{res["cpp_score"]:.2f}')
res['accept'] = float(np.mean([r['acc'] for r in res['frames']]))
print(f'\n{a.frames} frames x {nsteps} steps: {t_run:.2f}s -> '
      f'{res["steps_per_s"]:.0f} MC steps/s ({res["us_per_step"]:.2f} us/MC '
      f'step), acceptance {res["accept"]:.3f}')
if a.json:
    with open(a.json, 'a') as fh:
        fh.write(json.dumps(res) + '\n')
