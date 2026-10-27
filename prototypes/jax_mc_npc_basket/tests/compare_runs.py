"""Compare two run_npc_jax.py output directories stage by stage: trace,
checks and traj.dof byte for byte, and the sampler state in state.cpt.

    python compare_runs.py REF TEST STAGE [STAGE ...]
"""
import hashlib
import os
import sys

import numpy as np


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


ref, test, stages = sys.argv[1], sys.argv[2], sys.argv[3:]
bad = 0
for st in stages:
    for f in ('trace.jsonl', 'checks.jsonl', 'traj.dof'):
        a, b = os.path.join(ref, st, f), os.path.join(test, st, f)
        ok = md5(a) == md5(b)
        bad += not ok
        print(f'{st}/{f}: {"identical" if ok else "DIFFERENT"} '
              f'({os.path.getsize(a)} / {os.path.getsize(b)} bytes, '
              f'{sum(1 for _ in open(a, "rb"))} / {sum(1 for _ in open(b, "rb"))} lines)')
za, zb = np.load(os.path.join(ref, 'state.cpt')), np.load(os.path.join(test, 'state.cpt'))
for k in sorted(za.files):
    if k == '__meta__':
        continue
    ok = za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k], equal_nan=True)
    bad += not ok
    print(f'state.cpt {k}: {"identical" if ok else "DIFFERENT"} {za[k].dtype} {za[k].shape}')
print('ALL IDENTICAL' if not bad else f'{bad} DIFFERENCES')
