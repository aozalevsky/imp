"""RMF files from the DOF snapshots of a run_soa.py run, for the frames you
need, in parallel.

  python dofs_to_rmf.py runs/x --best 1000 [--procs 8]
      the N best-scoring snapshot configurations over all ranks and chains,
      split over PROCS files runs/x/best_00.rmf3, ... (frame name
      'r<rank>_c<chain>_f<frame>', best first), and runs/x/best.csv with
      their scores;
  python dofs_to_rmf.py runs/x --per-chain [--ranks 0] [--chains 0 5 9]
      one RMF per chain, as PMI and run_chains.py write them:
      runs/x/rank<r>/rmfs/<chain>.rmf3, one frame per snapshot, named by
      frame number (check_rmf.py reads these).

Configurations are set with fast_dofs.FastDofs (identical to
scoring.set_imp_dofs), one tutorial model per process.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np


def load_rank(run, r, mmap=True):
    d = os.path.join(run, f'rank{r}')
    if not os.path.exists(os.path.join(d, 'dofs_quat.npy')):
        raise SystemExit(f'{d} has no DOF snapshots (run run_soa.py with '
                         '--thin)')
    mode = 'r' if mmap else None
    return dict(
        summary=json.load(open(os.path.join(d, 'summary.json'))),
        quat=np.load(os.path.join(d, 'dofs_quat.npy'), mmap_mode=mode),
        trans=np.load(os.path.join(d, 'dofs_trans.npy'), mmap_mode=mode),
        bead=np.load(os.path.join(d, 'dofs_bead.npy'), mmap_mode=mode),
        frames=np.load(os.path.join(d, 'dofs_frames.npy')),
        scores=np.load(os.path.join(d, 'scores.npy'), mmap_mode=mode))


def worker(wid, run, cov, ev_res, jobs, q_done):
    """jobs: list of (path, [(frame name, rank, snapshot, chain), ...])"""
    import io
    import contextlib
    import tempfile
    import IMP.rmf
    import RMF
    import scoring as T
    from build_system import build
    from fast_dofs import FastDofs
    with contextlib.redirect_stdout(io.StringIO()):
        S = build(seed=1, workdir=tempfile.mkdtemp(prefix=f'd2r{wid}_'),
                  ev_resolution=ev_res)
    s = T.TutSystem(S, cov_mode=cov)
    fd = FastDofs(s)
    ranks = {}
    n, t0 = 0, time.perf_counter()
    for path, frames in jobs:
        fh = RMF.create_rmf_file(path)
        IMP.rmf.add_hierarchy(fh, S.root_hier)
        for name, r, k, c in frames:
            if r not in ranks:
                ranks[r] = load_rank(run, r)
            R = ranks[r]
            fd.set(R['quat'][k, c], R['trans'][k, c], R['bead'][k, c])
            IMP.rmf.save_frame(fh, name)
            n += 1
        del fh
    q_done.put((wid, n, time.perf_counter() - t0))


def run_workers(run, cov, ev_res, jobs, procs):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    parts = [jobs[i::procs] for i in range(procs)]
    ps = [ctx.Process(target=worker, args=(i, run, cov, ev_res, p, q))
          for i, p in enumerate(parts) if p]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
        if p.exitcode:
            raise RuntimeError('a worker failed')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--best', type=int, default=0)
    ap.add_argument('--per-chain', action='store_true')
    ap.add_argument('--ranks', type=int, nargs='+', default=None)
    ap.add_argument('--chains', type=int, nargs='+', default=None)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    top = json.load(open(os.path.join(run, 'summary.json')))
    ranks = a.ranks if a.ranks is not None else list(range(top['ranks']))
    R = {r: load_rank(run, r) for r in ranks}
    cov = R[ranks[0]]['summary']['cov']
    ev_res = R[ranks[0]]['summary'].get('ev_res')
    t0 = time.time()
    if a.best:
        cand = []                  # (E, rank, snapshot, chain, score row)
        for r, d in R.items():
            K = d['summary']['score_every']
            for k, f in enumerate(d['frames']):
                i = (f + 1) // K - 1
                e = np.asarray(d['scores'][i, :, 0])
                c = np.argsort(e)[:a.best]
                cand += [(float(e[j]), r, k, int(j), i) for j in c]
        cand.sort()
        cand = cand[:a.best]
        P = min(a.procs, len(cand))
        jobs, rows = [], []
        for p in range(P):
            sel = cand[p::P]
            path = os.path.join(run, f'best_{p:02d}.rmf3')
            frames = []
            for e, r, k, c, i in sel:
                f = int(R[r]['frames'][k])
                name = f'r{r}_c{c}_f{f}'
                frames.append((name, r, k, c))
                sc = np.asarray(R[r]['scores'][i, c])
                rows.append((name, path, len(frames) - 1, r, c, f, *sc))
            jobs.append((path, frames))
        res = run_workers(run, cov, ev_res, jobs, P)
        rows.sort(key=lambda x: x[6])
        with open(os.path.join(run, 'best.csv'), 'w') as fh:
            fh.write('name,file,frame_index,rank,chain,frame,E,kT,conn,EV,'
                     'XL,EM\n')
            for x in rows:
                fh.write(','.join(str(v) for v in x) + '\n')
        what = f'{len(cand)} best configurations into {P} files'
    elif a.per_chain:
        jobs = []
        for r, d in R.items():
            os.makedirs(os.path.join(run, f'rank{r}', 'rmfs'), exist_ok=True)
            chains = a.chains if a.chains is not None else \
                range(d['quat'].shape[1])
            for c in chains:
                jobs.append((os.path.join(run, f'rank{r}', 'rmfs',
                                          f'{c}.rmf3'),
                             [(str(int(f)), r, k, c)
                              for k, f in enumerate(d['frames'])]))
        res = run_workers(run, cov, ev_res, jobs, min(a.procs, len(jobs)))
        what = f'{len(jobs)} per-chain RMF files'
    else:
        raise SystemExit('give --best N or --per-chain')
    n = sum(x[1] for x in res)
    busy = max(x[2] for x in res)
    print(f'wrote {what}: {n} frames in {time.time() - t0:.1f} s wall '
          f'({len(res)} processes; slowest busy {busy:.1f} s, '
          f'{1e3 * sum(x[2] for x in res) / max(n, 1):.2f} ms per frame '
          f'per process)')


if __name__ == '__main__':
    main()
