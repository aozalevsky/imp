"""The imp_tutorial sampling protocol on JAX: N independent copies of the
tutorial's run (rnapolii/modeling/modeling.py without MPI, i.e. one replica
with simulated annealing), as N chains of the fused sampler.

Per frame and chain: 490 MC steps (10 sweeps over the 49 movers), then the
temperature for the next frame from PMI's annealing schedule (200 frames at
kT 1.0, 20 at 2.5), a full rescore (removes float32 drift), and the score
and its terms are recorded. Every --thin frames each chain's configuration
goes to its own RMF file, written by --writers background processes (each
with its own copy of the IMP model) while the device keeps sampling. At the
end the 10 best-scoring configurations over all chains go to best.rmf3.

  JAX_PLATFORMS=cuda python run_chains.py --chains 1024 --frames 20000 \\
      --thin 100 --writers 8 --out runs/x
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np


def sa_temperature(nframe, tmin=1.0, tmax=2.5, nmin=200, nmax=20):
    """IMP.pmi.samplers._SamplerBase.temp_simulated_annealing"""
    return tmin if nframe % (nmin + nmax) < nmin else tmax


def writer_main(wid, chains, out, cov, q_in, q_done):
    """Background RMF writer for a subset of chains"""
    import io
    import contextlib
    import IMP
    import IMP.rmf
    import RMF
    import scoring as T
    from build_system import build
    with contextlib.redirect_stdout(io.StringIO()):
        S = build(seed=1, workdir=os.path.join(out, f'.writer{wid}'))
    s = T.TutSystem(S, cov_mode=cov)
    fhs = {}
    for c in chains:
        fh = RMF.create_rmf_file(os.path.join(out, 'rmfs', f'{c}.rmf3'))
        IMP.rmf.add_hierarchy(fh, S.root_hier)
        fhs[c] = fh
    q_done.put(('ready', wid))
    busy = 0.
    while True:
        item = q_in.get()
        if item is None:
            break
        t = time.perf_counter()
        f, q, tr, b = item
        for k, c in enumerate(chains):
            T.set_imp_dofs(s, q[k], tr[k], b[k])
            IMP.rmf.save_frame(fhs[c], str(f))
        busy += time.perf_counter() - t
    del fhs
    q_done.put(('done', wid, busy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', type=int, default=64)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--thin', type=int, default=10,
                    help='RMF frame every THIN frames (0: no RMF)')
    ap.add_argument('--writers', type=int, default=4)
    ap.add_argument('--cov', default='rotate', choices=['rotate', 'frozen'])
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--score-every', type=int, default=1,
                    help='store the scores of every K-th frame only (long '
                         'runs); the drift record keeps every frame')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.makedirs(os.path.join(a.out, 'rmfs'), exist_ok=True)

    import resource
    s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    log = open(os.path.join(a.out, 'run.log'), 'a')

    def say(*x):
        msg = ' '.join(str(y) for y in x)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    # writers first (spawned before JAX initializes)
    writers, q_ins = [], []
    ctx = mp.get_context('spawn')
    q_done = ctx.Queue()
    N = a.chains
    if a.thin:
        parts = np.array_split(np.arange(N), a.writers)
        for w, part in enumerate(parts):
            qi = ctx.Queue(maxsize=8)
            p = ctx.Process(target=writer_main, args=(
                w, part.tolist(), a.out, a.cov, qi, q_done), daemon=True)
            p.start()
            writers.append((p, part))
            q_ins.append(qi)

    t0 = time.time()
    import io
    import contextlib
    import jax
    import jax.numpy as jnp
    import IMP
    import IMP.rmf
    import RMF
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    with contextlib.redirect_stdout(io.StringIO()):
        S = build(seed=1, workdir=os.path.join(a.out, '.main'))
    s = T.TutSystem(S, cov_mode=a.cov)
    smp = TutSampler(s, dtype=a.dtype)
    st = smp.initial_state(N, kT=1.0, seed=a.seed)
    run = smp.make_run(10)
    t_setup = time.time() - t0
    say(f'IMP {IMP.get_module_version()}, JAX {jax.__version__} on '
        f'{jax.devices()[0]}; {N} chains x {a.frames} frames x '
        f'{10 * smp.n_movers} MC steps; cov {a.cov}; setup {t_setup:.1f} s')
    # compile before timing
    t1 = time.time()
    jax.block_until_ready(run(st).energy)
    smp.resync(st, terms=True)
    t_compile = time.time() - t1
    for _ in writers:
        q_done.get()
    say(f'compile {t_compile:.1f} s; writers ready')

    K = a.score_every
    if a.thin and a.thin % K:
        raise SystemExit('--thin must be a multiple of --score-every')
    scores = np.zeros((a.frames // K, N, 6), np.float32)  # E, kT, 4 terms
    # per frame: max and median over chains of |incrementally tracked
    # energy - full rescore| (the float32 drift removed by the resync)
    drift = np.zeros((a.frames, 2), np.float32)
    top = []                                           # (E, chain, frame)
    best_dofs = {}
    kT = np.ones(N)
    t_start = time.time()
    t_wait = 0.
    for f in range(a.frames):
        st = run(st)
        e_track = st.energy
        st, terms = smp.resync(st, terms=True)
        dd = jnp.abs(e_track - st.energy)
        e, tr, dmax, dmed = jax.device_get((st.energy, terms, jnp.max(dd),
                                            jnp.median(dd)))
        drift[f] = dmax, dmed
        if (f + 1) % K == 0:            # frames K-1, 2K-1, ... (RMF frames
            i = (f + 1) // K - 1        # are THIN-1, ...: THIN multiple of K)
            scores[i, :, 0] = e
            scores[i, :, 1] = kT
            scores[i, :, 2:] = tr
        # best-scoring configurations (PMI keeps the 10 best)
        cand = np.argsort(e)[:10]
        worst = top[-1][0] if len(top) == 10 else np.inf
        if e[cand[0]] < worst:
            q, t_, b = smp.dofs(st)
            for c in cand:
                if e[c] < (top[-1][0] if len(top) == 10 else np.inf):
                    top.append((float(e[c]), int(c), f))
                    best_dofs[(int(c), f)] = (q[c], t_[c], b[c])
                    top.sort()
                    for old in top[10:]:
                        best_dofs.pop(old[1:], None)
                    top = top[:10]
        if a.thin and (f + 1) % a.thin == 0:
            q, t_, b = smp.dofs(st)
            tw = time.time()
            for (p, part), qi in zip(writers, q_ins):
                qi.put((f, q[part], t_[part], b[part]))
            t_wait += time.time() - tw
        kT[:] = sa_temperature(f)
        st.kT = jnp.full(N, kT[0], smp.dt)
        if (f + 1) % max(1, a.frames // 10) == 0:
            say(f'  frame {f + 1}/{a.frames}: median score '
                f'{np.median(e):.1f}, best {e.min():.1f}, kT {kT[0]}; '
                f'wall {time.time() - t_start:.1f} s')
    jax.block_until_ready(st.energy)
    t_sample = time.time() - t_start
    for qi in q_ins:
        qi.put(None)
    busy = []
    for _ in writers:
        msg = q_done.get()
        busy.append(msg[2])
    for p, _ in writers:
        p.join()
    wall = time.time() - t_start
    # best models
    fh = RMF.create_rmf_file(os.path.join(a.out, 'best.rmf3'))
    IMP.rmf.add_hierarchy(fh, S.root_hier)
    for E, c, f in top:
        T.set_imp_dofs(s, *best_dofs[(c, f)])
        IMP.rmf.save_frame(fh, f'chain {c} frame {f} score {E:.3f}')
    del fh
    np.save(os.path.join(a.out, 'scores.npy'), scores)
    np.save(os.path.join(a.out, 'drift.npy'), drift)
    acc = np.asarray(st.accepted).sum(0) / (N * a.frames * 10)
    steps = N * a.frames * 10 * smp.n_movers
    res = dict(chains=N, frames=a.frames, steps=steps, cov=a.cov,
               score_every=K,
               device=str(jax.devices()[0]), setup_s=t_setup,
               compile_s=t_compile, sample_s=t_sample, wall_s=wall,
               mc_steps_per_s_sampling=steps / t_sample,
               mc_steps_per_s_end_to_end=steps / wall,
               thin=a.thin, writers=a.writers, writer_busy_s=busy,
               queue_wait_s=t_wait, best=top[:3],
               drift_max=float(drift[:, 0].max()),
               drift_median_of_medians=float(np.median(drift[:, 1])),
               drift_max_by_tenth=[float(x.max()) for x in
                                   np.array_split(drift[:, 0], 10)],
               acceptance_beads=float(np.delete(
                   acc, [i for i, p in enumerate(smp.steps)
                         if p['kind'] == 'rb']).mean()),
               acceptance_rbs=[float(acc[i]) for i, p in
                               enumerate(smp.steps) if p['kind'] == 'rb'])
    say(json.dumps(res))
    with open(os.path.join(a.out, 'summary.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
