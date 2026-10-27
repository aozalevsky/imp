"""Check the JAX implementations against C++ IMP, term by term, through
IMP's own JAX glue (Restraint._get_jax / RestraintSet._get_jax), on
several configurations: after REX1 and after successive C++ MC frames of
the REX2 scoring function.

  python validate_terms.py yeast --configs 5 --steps 200
"""
import argparse
import collections
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument('system', nargs='?', default='yeast')
ap.add_argument('--frames1', type=int, default=2)
ap.add_argument('--configs', type=int, default=4)
ap.add_argument('--steps', type=int, default=500,
                help='C++ MC steps between configurations')
ap.add_argument('--x64', action='store_true')
ap.add_argument('--full', action='store_true',
                help='also evaluate the whole scoring function via IMP '
                     'JAX (slow to compile: thousands of PairRestraints)')
ap.add_argument('--json', default=None)
ap.add_argument('--only', default=None,
                help='comma-separated restraint-set names to check')
a = ap.parse_args()

import jax  # noqa: E402
if a.x64:
    jax.config.update('jax_enable_x64', True)
import IMP  # noqa: E402
import IMP.core  # noqa: E402
import IMP.pmi.samplers  # noqa: E402

import imp_jax_npc  # noqa: E402
from npc_setup import build  # noqa: E402

imp_jax_npc.install()
S = build(a.system, frames1=a.frames1)
m = S.model
rs = S.rs
print('IMP', IMP.get_module_version(), 'jax', jax.__version__,
      jax.default_backend(), 'x64' if a.x64 else 'f32')

# group top-level restraint sets by label
tops = [rs.get_restraint(i) for i in range(rs.get_number_of_restraints())]
names = collections.Counter(r.get_name() for r in tops)


def jax_eval_fn(r):
    ji = r.get_derived_object()._get_jax(None) if not isinstance(
        r, IMP.RestraintSet) else IMP.RestraintSet.get_from(r)._get_jax(None)
    f = jax.jit(ji.score_func)
    return ji, f


import IMP.jax  # noqa: E402
space = IMP.jax.FreeSpace
fns = []
t0 = time.time()
only = set(a.only.split(',')) if a.only else None
for r in tops:
    if only and r.get_name() not in only:
        continue
    rset = IMP.RestraintSet.get_from(r)
    ji = rset._get_jax(space)
    fns.append((r, ji, jax.jit(ji.score_func)))
print(f'built {len(fns)} JAX restraint-set functions in '
      f'{time.time() - t0:.1f}s')

full = None
if a.full:
    sf = IMP.core.RestraintsScoringFunction([rs])
    t0 = time.time()
    ji_full = sf._get_jax(space)
    f_full = jax.jit(ji_full.score_func)
    full = (ji_full, f_full)

mc = IMP.pmi.samplers.MonteCarlo(m, S.movers, 1.0)
mc.mc.set_scoring_function([rs])
res = dict(system=a.system, x64=a.x64, configs=[])
worst = collections.defaultdict(float)
for c in range(a.configs):
    if c:
        mc.optimize(a.steps)
    cpp_total = rs.evaluate(False)
    by = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    t_cpp = t_jax = 0.0
    for r, ji, f in fns:
        t = time.perf_counter()
        e_cpp = r.evaluate(False)
        t_cpp += time.perf_counter() - t
        jm = ji.get_jax_model()
        t = time.perf_counter()
        e_jax = float(f(jm))
        t_jax += time.perf_counter() - t
        b = by[r.get_name()]
        b[0] += e_cpp
        b[1] += e_jax
        b[2] = max(b[2], abs(e_cpp - e_jax))
    row = dict(config=c, cpp_total=cpp_total,
               jax_total=sum(v[1] for v in by.values()), terms={})
    print(f'\n-- config {c}: C++ total {cpp_total:.6f}, JAX sum of sets '
          f'{row["jax_total"]:.6f}')
    for nm, (ec, ej, md) in by.items():
        rel = abs(ec - ej) / max(1.0, abs(ec))
        worst[nm] = max(worst[nm], rel)
        row['terms'][nm] = dict(cpp=ec, jax=ej, absdiff=abs(ec - ej),
                                rel=rel)
        print(f'   {nm:45s} C++ {ec:16.6f}  JAX {ej:16.6f}  '
              f'|d| {abs(ec - ej):.3g}  rel {rel:.2g}')
    if full:
        ji_full, f_full = full
        jm = ji_full.get_jax_model()
        t = time.time()
        e = float(f_full(jm))
        print(f'   whole scoring function via IMP JAX: {e:.6f} '
              f'(first call incl. compile {time.time() - t:.1f}s)')
        row['jax_full'] = e
    res['configs'].append(row)

print('\nworst relative difference per term:')
for nm, v in worst.items():
    print(f'   {nm:45s} {v:.3g}')
res['worst_rel'] = dict(worst)
if a.json:
    with open(a.json, 'a') as fh:
        fh.write(json.dumps(res) + '\n')
