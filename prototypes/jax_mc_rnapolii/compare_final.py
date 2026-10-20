"""Final results of complete tutorial runs, C++ IMP vs the JAX sampler.

Both run the tutorial's protocol (20,000 frames of 490 MC steps with PMI's
annealing schedule) from the tutorial's shuffled start (build(seed=1)), with
different random numbers per run. Recorded: the frame-end score of every
frame, and every 100 frames the terms and the stalk (Rpb4, Rpb7) rigid-body
frames.

    python compare_final.py cpp --run 0 [--frames 20000] [--score-moved]
    python compare_final.py jax --chains 1024 [--cov frozen] [--em-slack 0]
    python compare_final.py compare

C++ uses the restraints as the tutorial does (stale covariances), so compare
with --cov frozen. score_moved only speeds C++ up (it is exact in current
IMP). Results go to ./final.
"""
import argparse
import glob
import os
import time

import numpy as np

OUT = os.path.abspath('final')
EVERY = 100


def sa_temperature(nframe, tmin=1.0, tmax=2.5, nmin=200, nmax=20):
    return tmin if nframe % (nmin + nmax) < nmin else tmax


def body_frames(s):
    q, t = [], []
    for b in s.bodies:
        tr = b.get_reference_frame().get_transformation_to()
        q.append(tr.get_rotation().get_quaternion())
        t.append(tr.get_translation())
    return np.array(q), np.array(t)


def run_cpp(a):
    import IMP
    import IMP.pmi.samplers
    import scoring as T
    from build_system import build
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode='frozen')
    IMP.random_number_generator.seed(1000 + a.run)
    smp = IMP.pmi.samplers.MonteCarlo(S.m, S.movers, 1.0,
                                      score_moved=a.score_moved)
    smp.set_simulated_annealing(1.0, 2.5, 200, 20)
    score = np.zeros(a.frames)
    kT = np.zeros(a.frames)
    nrec = a.frames // EVERY
    terms = np.zeros((nrec, 5))
    quat = np.zeros((nrec, s.nb, 4))
    trans = np.zeros((nrec, s.nb, 3))
    t0 = time.perf_counter()
    for f in range(a.frames):
        kT[f] = smp.get_mc().get_kt()
        score[f] = smp.optimize(10)
        if (f + 1) % EVERY == 0:
            i = (f + 1) // EVERY - 1
            terms[i] = list(T.cpp_terms(S).values())
            quat[i], trans[i] = body_frames(s)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'cpp_{a.run}.npz'), score=score, kT=kT,
             terms=terms, quat=quat, trans=trans, imp=IMP.get_module_version(),
             seconds=time.perf_counter() - t0)


def run_jax(a):
    import jax
    import jax.numpy as jnp
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    S = build(seed=1)
    s = T.TutSystem(S, cov_mode=a.cov, em_slack=a.em_slack)
    smp = TutSampler(s)
    N = a.chains
    st = smp.initial_state(N, kT=1.0, seed=a.seed)
    run = smp.make_run(10)
    score = np.zeros((a.frames, N), np.float32)
    kT = np.zeros(a.frames)
    nrec = a.frames // EVERY
    terms = np.zeros((nrec, N, 5), np.float32)
    quat = np.zeros((nrec, N, s.nb, 4), np.float32)
    trans = np.zeros((nrec, N, s.nb, 3), np.float32)
    t0 = time.perf_counter()
    kt = 1.0
    for f in range(a.frames):
        kT[f] = kt
        st = run(st)
        st, tr = smp.resync(st, terms=True)
        e, tr = jax.device_get((st.energy, tr))
        score[f] = e
        if (f + 1) % EVERY == 0:
            i = (f + 1) // EVERY - 1
            terms[i, :, :4] = tr
            terms[i, :, 4] = e
            quat[i] = np.asarray(st.quat)
            trans[i] = np.asarray(st.trans)
        kt = sa_temperature(f)
        st.kT = jnp.full(N, kt, smp.dt)
    os.makedirs(OUT, exist_ok=True)
    tag = f'jax_{a.cov}_slack{a.em_slack:g}'
    np.savez(os.path.join(OUT, tag + '.npz'), score=score, kT=kT,
             terms=terms, quat=quat, trans=trans,
             device=str(jax.devices()[0]),
             seconds=time.perf_counter() - t0)


def rotation_angle(q, qref):
    """Angle (degrees) between unit quaternions q (..., 4) and qref (4,)"""
    d = np.abs(np.sum(q * qref, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


def compare(a):
    from scipy.stats import ks_2samp, mannwhitneyu
    cpp = [np.load(f) for f in sorted(glob.glob(os.path.join(OUT,
                                                             'cpp_*.npz')))]
    C = dict(score=np.array([c['score'] for c in cpp]),          # runs, F
             terms=np.array([c['terms'] for c in cpp]),          # runs, R, 5
             quat=np.array([c['quat'] for c in cpp]),
             trans=np.array([c['trans'] for c in cpp]))
    kT = cpp[0]['kT']
    print(f'C++: {len(cpp)} runs ({cpp[0]["imp"]}), '
          f'{np.median([c["seconds"] for c in cpp]) / 60:.1f} min per run')
    names = ['connectivity', 'ev', 'xl', 'em', 'total']
    rec_frames = np.arange(EVERY - 1, C['score'].shape[1], EVERY)
    # recorded frames of the second half that ran at kT 1
    late = [i for i, f in enumerate(rec_frames)
            if f >= C['score'].shape[1] // 2 and kT[f] == 1.0]
    # a fixed reference orientation per body (sign-free angles below)
    qref = C['quat'][0, -1]

    def summary(D, runs_axis):
        out = {}
        sc = D['score']
        if runs_axis == 1:          # JAX arrays are (F, N, ...)
            sc = sc.T
            terms = D['terms'].transpose(1, 0, 2)
            quat = D['quat'].transpose(1, 0, 2, 3)
            trans = D['trans'].transpose(1, 0, 2, 3)
        else:
            terms, quat, trans = D['terms'], D['quat'], D['trans']
        out['final score'] = sc[:, -1]
        out['best score'] = sc.min(1)
        for k, n in enumerate(names):
            out[f'late {n} (run mean)'] = terms[:, late, k].mean(1)
        for b, bn in enumerate(['Rpb4', 'Rpb7']):
            for x, xn in enumerate('xyz'):
                out[f'final {bn} {xn}'] = trans[:, -1, b, x]
            out[f'final {bn} rotation (deg)'] = rotation_angle(
                quat[:, -1, b], qref[b])
        return out

    ref = summary(C, 0)
    for f in sorted(glob.glob(os.path.join(OUT, 'jax_*.npz'))):
        d = np.load(f)
        J = summary(dict(score=d['score'], terms=d['terms'], quat=d['quat'],
                         trans=d['trans']), 1)
        print(f'\n== {os.path.basename(f)}: {d["score"].shape[1]} chains '
              f'({d["device"]}, {float(d["seconds"]) / 60:.1f} min)')
        print(f'  {"quantity":28s} {"C++ mean +- SE":>22s} '
              f'{"JAX mean +- SE":>22s}  {"KS p":>6s} {"MW p":>6s}')
        for k in ref:
            c, j = ref[k], J[k]
            ks = ks_2samp(c, j).pvalue
            mw = mannwhitneyu(c, j).pvalue
            se_c = c.std(ddof=1) / np.sqrt(len(c))
            se_j = j.std(ddof=1) / np.sqrt(len(j))
            print(f'  {k:28s} {c.mean():12.3f} +- {se_c:7.3f} '
                  f'{j.mean():12.3f} +- {se_j:7.3f}  {ks:6.3f} {mw:6.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cpp', 'jax', 'compare'])
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--frames', type=int, default=20000)
    ap.add_argument('--score-moved', action='store_true')
    ap.add_argument('--chains', type=int, default=1024)
    ap.add_argument('--cov', default='frozen', choices=['rotate', 'frozen'])
    ap.add_argument('--em-slack', type=float, default=0.)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    {'cpp': run_cpp, 'jax': run_jax, 'compare': compare}[a.mode](a)


if __name__ == '__main__':
    main()
