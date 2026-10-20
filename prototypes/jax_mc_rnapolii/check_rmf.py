"""Rescore frames of a run_chains.py output with C++ IMP: load each RMF
frame into the tutorial model, refresh the model Gaussians' covariances
(--cov rotate runs), evaluate the restraint set and compare with the score
the JAX run recorded. C++ EM uses its own close-pair lists, so the EM term
is also recomputed with scoring.py on those lists to separate that effect.

    python check_rmf.py runs/x [--chains 0 1 2] [--frames 4]
"""
import argparse
import json
import os

import numpy as np
import IMP
import IMP.core
import IMP.rmf
import RMF

import scoring as T
from build_system import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--chains', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--frames', type=int, default=4)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.run, 'summary.json')))
    scores = np.load(os.path.join(a.run, 'scores.npy'))
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode=summ['cov'])
    worst = 0.
    for c in a.chains:
        fh = RMF.open_rmf_file_read_only(os.path.join(a.run, 'rmfs',
                                                      f'{c}.rmf3'))
        IMP.rmf.link_hierarchies(fh, [S.root_hier])
        nf = fh.get_number_of_frames()
        for i in np.linspace(0, nf - 1, min(a.frames, nf)).astype(int):
            IMP.rmf.load_frame(fh, RMF.FrameID(int(i)))
            f = int(fh.get_name(RMF.FrameID(int(i))))
            S.m.update()
            if summ['cov'] == 'rotate':
                for p in s.em_model_pi:
                    IMP.core.Gaussian(S.m, IMP.ParticleIndex(int(p))) \
                        .update_global_covariance()
            cpp = T.cpp_terms(S)
            # the same configuration through scoring.py, with C++'s EM lists
            q, t, b = T.get_imp_dofs(s)
            s.em_masks_from_cpp()
            ref = T.energy_terms(s, q, t, b, np)
            s.em_md_mask = s.em_mm_mask = None
            K = summ.get('score_every', 1)   # scores of frames K-1, 2K-1, ...
            jax_e = float(scores[(f + 1) // K - 1, c, 0])
            d = cpp['total'] - ref['total']
            worst = max(worst, abs(jax_e - (cpp['total'] - (
                ref['em'] - T.energy_terms(s, q, t, b, np)['em']))))
            print(f'chain {c} frame {f}: JAX run {jax_e:.4f}, C++ '
                  f'{cpp["total"]:.4f} (EM list effect '
                  f'{ref["em"] - T.energy_terms(s, q, t, b, np)["em"]:+.4f}'
                  f'; C++ vs scoring.py on C++ lists {d:+.1e})')
    print(f'worst |JAX run - C++ (EM on cutoff pairs)|: {worst:.2e}')


if __name__ == '__main__':
    main()
