"""Statistical check of the JAX sampler against C++ IMP: both start from the
tutorial's shuffled configuration (build(seed=1)) and run frames of 490 MC
steps at kT 1 (no annealing); the per-frame score distributions must agree.

    python ensemble.py cpp --run 0 --frames 100      # one C++ trajectory
    python ensemble.py jax --chains 256 --frames 100 [--cov frozen]
    python ensemble.py compare

C++ runs use the restraint set as the tutorial does (stale covariances, so
compare with --cov frozen); JAX scores are full recomputes after each frame.
"""
import argparse
import glob
import os
import time

import numpy as np

OUT = os.path.abspath('ens')          # results go to ./ens


def run_cpp(a):
    import IMP
    import IMP.pmi.samplers
    import scoring as T
    from build_system import build
    S = build(seed=1)
    IMP.random_number_generator.seed(1000 + a.run)
    smp = IMP.pmi.samplers.MonteCarlo(S.m, S.movers, 1.0)
    mc = smp.get_mc()
    terms = []
    t = time.perf_counter()
    for f in range(a.frames):
        mc.optimize(10 * len(S.movers))
        terms.append([v for v in T.cpp_terms(S).values()])
    np.savez(os.path.join(OUT, f'cpp_{a.run}.npz'), terms=np.array(terms),
             keys=list(T.cpp_terms(S).keys()),
             acc=mc.get_number_of_accepted_steps()
             / mc.get_number_of_proposed_steps(),
             seconds=time.perf_counter() - t)


def run_jax(a):
    import jax
    if a.dtype == 'float64':
        jax.config.update('jax_enable_x64', True)
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode=a.cov, em_slack=a.em_slack)
    smp = TutSampler(s, dtype=a.dtype)
    st = smp.initial_state(a.chains, kT=1.0, seed=a.seed)
    run = smp.make_run(10)
    terms = []
    for f in range(a.frames):
        st = smp.resync(run(st))
        q, tr, b = smp.dofs(st)
        terms.append([[v for v in T.energy_terms(
            s, q[c], tr[c], b[c], np).values()] for c in range(a.chains)])
    acc = np.asarray(st.accepted).sum() / (a.chains * a.frames * 490)
    tag = f'_slack{a.em_slack:g}' if a.em_slack else ''
    np.savez(os.path.join(OUT, f'jax_{a.cov}_{a.dtype}{tag}.npz'),
             terms=np.array(terms).transpose(1, 0, 2),
             keys=list(T.energy_terms(s, q[0], tr[0], b[0], np).keys()),
             acc=acc, device=str(jax.devices()[0]))


def compare(a):
    cpp = [np.load(f) for f in sorted(glob.glob(os.path.join(OUT,
                                                             'cpp_*.npz')))]
    C = np.array([c['terms'] for c in cpp])          # (runs, frames, terms)
    keys = list(cpp[0]['keys'])
    print(f'C++: {len(cpp)} runs')
    for f in sorted(glob.glob(os.path.join(OUT, 'jax_*.npz'))):
        J = np.load(f)
        X = J['terms']                               # (chains, frames, terms)
        print(f'\n{os.path.basename(f)}: {X.shape[0]} chains')
        nf = min(X.shape[1], C.shape[1])
        for fr in sorted({0, 1, 4, 9, 19, 49, nf - 1}):
            if fr >= nf:
                continue
            line = [f'frame {fr + 1:3d}']
            for k, name in enumerate(keys):
                c, x = C[:, fr, k], X[:, fr, k]
                se = np.sqrt(c.var(ddof=1) / len(c) + x.var(ddof=1) / len(x))
                line.append(f'{name} C++ {c.mean():9.2f} JAX {x.mean():9.2f}'
                            f' ({(x.mean() - c.mean()) / se:+.1f} SE)')
            print('  ' + '; '.join(line))
        # last half of the frames pooled
        h = nf // 2
        print('  frames %d-%d pooled:' % (h + 1, nf))
        for k, name in enumerate(keys):
            c = C[:, h:nf, k].mean(1)
            x = X[:, h:nf, k].mean(1)
            se = np.sqrt(c.var(ddof=1) / len(c) + x.var(ddof=1) / len(x))
            print(f'    {name:13s} C++ {c.mean():9.3f} +- '
                  f'{c.std() / np.sqrt(len(c)):.3f}'
                  f'   JAX {x.mean():9.3f} +- {x.std() / np.sqrt(len(x)):.3f}'
                  f'   ({(x.mean() - c.mean()) / se:+.1f} SE)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cpp', 'jax', 'compare'])
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--chains', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--cov', default='frozen')
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--em-slack', type=float, default=0.)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {'cpp': run_cpp, 'jax': run_jax, 'compare': compare}[a.mode](a)


if __name__ == '__main__':
    main()
