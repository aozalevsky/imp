"""The tutorial's runs on one or several GPUs with the struct-of-arrays
sampler, the device kept busy: frames run in blocks inside one jitted call,
and the output is the sampled degrees of freedom, not RMF.

    mpirun -np R python run_soa.py --chains B --frames F [--block 20] \\
        [--thin T] [--score-every K] [--rescore recorded] --out runs/x
    python dofs_to_rmf.py runs/x ...       # RMF for the frames you need

Same protocol as run_mpi.py --mode independent (R x B independent copies of
the tutorial's run with PMI's annealing schedule; per frame 10 sweeps and a
full rescore; with --rescore recorded only the recorded frames are
rescored), but:

* --block frames run in one device call (lax.scan): the per-frame rescore,
  the drift statistics, the scores and each frame's 10 best configurations
  stay on the GPU until the block ends;
* the next block is dispatched before the previous block's records are
  read, so the host's work overlaps with sampling;
* every --thin frames a snapshot of every chain's degrees of freedom (rigid
  body quaternions and translations, bead positions; the sampler's float32
  state, 620 bytes per chain) goes to dofs_*.npy, written by a background
  thread. Blocks end on snapshot frames. dofs_to_rmf.py turns snapshots
  into RMF files.

How EV is scored against the moving rigid bodies (dense or per-body grids,
both exact) is picked by ev_select.py: --ev auto (default: a heuristic,
else a timed trial of both, remembered in --tune-cache), --ev dense or
--ev body.

Ranks only need their rank and local rank (from the Open MPI environment);
the only exchange between ranks is the EV trial's timings, through files in
OUT/ev_select/.
Outputs in OUT/rank<r>/:
  scores.npy (frames // K, B, 6): E, kT, conn, EV, XL, EM of frames K-1,
      2K-1, ...;
  drift.npy (frames, 2): max and median |tracked - rescored| per frame
      (NaN on frames that are not rescored);
  dofs_quat.npy (S, B, nb, 4), dofs_trans.npy (S, B, nb, 3),
      dofs_bead.npy (S, B, nbead, 3), dofs_frames.npy (S,) frame numbers;
  best.npz: the 10 best-scoring configurations over all frames and chains
      (by tracked energies on frames that are not rescored);
  summary.json; and OUT/summary.json (rank 0) with the totals.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np

from run_chains import sa_temperature
from run_mpi import rank_info


def make_block(smp, sweeps=10, top=10, rescore_every=1):
    """jitted f(state, kTs (F,)) -> state, per-frame records. With
       rescore_every K > 1 only every K-th frame is rescored; the others
       record the sampler's tracked energies, NaN terms and NaN drift."""
    import jax
    import jax.numpy as jnp
    o, nbead = smp.bead_off, smp.s.nbead
    K = rescore_every

    def sample(st, kT):
        st.kT = jnp.full_like(st.kT, kT)
        return jax.lax.fori_loop(0, sweeps, lambda _, x: smp._sweep(x), st)

    def best_of(st, e):
        neg, idx = jax.lax.top_k(-e, top)
        return (idx, -neg, st.quat[:, :, idx], st.trans[:, :, idx],
                st.X[:, o:o + nbead, idx])

    def frame(st, kT):
        st = sample(st, kT)
        e_track = st.energy
        e, em, tr, eb, ebody = smp._full(st.X, st.quat) \
            + smp._static_ev(st.X)
        st.energy, st.em, st.es_bead, st.es_body = e, em, eb, ebody
        dd = jnp.abs(e_track - e)
        return st, (e, tr, jnp.max(dd), jnp.median(dd), best_of(st, e))

    def frame_tracked(st, kT):
        st = sample(st, kT)
        e = st.energy
        nan = jnp.array(jnp.nan, e.dtype)
        tr = jnp.full(e.shape + (4,), jnp.nan, e.dtype)
        return st, (e, tr, nan, nan, best_of(st, e))

    def group(st, kTs):
        st, r1 = jax.lax.scan(frame_tracked, st, kTs[:-1])
        st, r2 = frame(st, kTs[-1])
        return st, jax.tree.map(
            lambda a, b: jnp.concatenate([a, b[None]]), r1, r2)

    def block(st, kTs):
        if K == 1:
            return jax.lax.scan(frame, st, kTs)
        st, rec = jax.lax.scan(group, st, kTs.reshape(-1, K))
        return st, jax.tree.map(
            lambda a: a.reshape((-1,) + a.shape[2:]), rec)
    return jax.jit(block, donate_argnums=0)


class SnapshotWriter(threading.Thread):
    """Copies device snapshots to the host and into preallocated .npy
       files, off the main thread"""

    def __init__(self, out, n_snap, B, nb, nbead):
        super().__init__(daemon=True)
        mm = np.lib.format.open_memmap
        self.quat = mm(os.path.join(out, 'dofs_quat.npy'), 'w+',
                       np.float32, (n_snap, B, nb, 4))
        self.trans = mm(os.path.join(out, 'dofs_trans.npy'), 'w+',
                        np.float32, (n_snap, B, nb, 3))
        self.bead = mm(os.path.join(out, 'dofs_bead.npy'), 'w+',
                       np.float32, (n_snap, B, nbead, 3))
        self.frames = np.full(n_snap, -1, np.int64)
        self.path_frames = os.path.join(out, 'dofs_frames.npy')
        self.q = queue.Queue(maxsize=4)
        self.busy = 0.
        self.error = None

    def run(self):
        import jax
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                k, f, sn = item
                t = time.perf_counter()
                qd, td, bd = jax.device_get(sn)
                self.quat[k] = np.moveaxis(qd, -1, 0)
                self.trans[k] = np.moveaxis(td, -1, 0)
                self.bead[k] = bd.transpose(2, 1, 0)
                self.frames[k] = f
                self.busy += time.perf_counter() - t
            for x in (self.quat, self.trans, self.bead):
                x.flush()
            np.save(self.path_frames, self.frames)
        except Exception as e:          # reported by the main thread
            self.error = e

    def put(self, item):
        if self.error is not None:
            raise RuntimeError(f'snapshot writer failed: {self.error!r}')
        self.q.put(item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', type=int, default=16384,
                    help='chains per rank')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--block', type=int, default=20,
                    help='frames per device call')
    ap.add_argument('--thin', type=int, default=0,
                    help='snapshot of the DOFs every THIN frames (0: none)')
    ap.add_argument('--score-every', type=int, default=1,
                    help='record the scores of every K-th frame')
    ap.add_argument('--gpus', default=None,
                    help='comma-separated GPU per local rank (default: the '
                         'local rank)')
    ap.add_argument('--cov', default='rotate', choices=['rotate', 'frozen'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--ev', default='auto', choices=['auto', 'dense', 'body'],
                    help='EV against the moving rigid bodies: dense, '
                         'per-body grids, or picked by ev_select.py')
    ap.add_argument('--tune-cache', default=os.path.expanduser(
        '~/.cache/jax_mc_rnapolii/ev_select.json'),
        help='where --ev auto remembers its trials ("" for none)')
    ap.add_argument('--rescore', default='every',
                    choices=['every', 'recorded'],
                    help='full rescore of every frame (as run_mpi.py), or '
                         'only of the frames whose scores are recorded '
                         '(every --score-every-th); the others keep the '
                         "sampler's tracked energies")
    ap.add_argument('--ev-res', type=float, default=None,
                    help='resolution of the excluded volume (tutorial: 10)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rank, size, local = rank_info()
    gpu = a.gpus.split(',')[local] if a.gpus else str(local)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    out = os.path.join(os.path.abspath(a.out), f'rank{rank}')
    os.makedirs(out, exist_ok=True)
    B, F, K = a.chains, a.block, a.score_every
    if a.thin:
        if a.thin % F and F % a.thin:
            raise SystemExit('--block and --thin must divide one another')
        F = min(F, a.thin)
        if a.thin % K:
            raise SystemExit('--thin must be a multiple of --score-every')
    if a.frames % F or F % K and K % F:
        raise SystemExit('--frames must be a multiple of the block, and '
                         'the block and --score-every divide one another')
    RS = K if a.rescore == 'recorded' else 1
    if F % RS:
        raise SystemExit('--rescore recorded needs a block that is a '
                         'multiple of --score-every')
    log = open(os.path.join(out, 'run.log'), 'a')

    def say(*x, all_ranks=False):
        msg = ' '.join(str(y) for y in x)
        log.write(msg + '\n')
        log.flush()
        if rank == 0 or all_ranks:
            print(f'[rank {rank}] {msg}', flush=True)

    t0 = time.time()
    import io
    import contextlib
    import jax
    import jax.numpy as jnp
    import IMP
    import scoring as T
    import ev_select
    from sampler_soa import TutSamplerSoA
    from build_system import build
    with contextlib.redirect_stdout(io.StringIO()):
        S = build(seed=1, workdir=os.path.join(out, '.main'),
                  ev_resolution=a.ev_res)
    s = T.TutSystem(S, cov_mode=a.cov)
    smp = TutSamplerSoA(s)
    st = smp.initial_state(B, kT=1.0, seed=a.seed + 7919 * rank)
    o, nbead = smp.bead_off, s.nbead
    t_setup = time.time() - t0
    kT1 = jnp.ones(F, smp.dt)
    smp, block, ev_info = ev_select.select(
        smp, st, kT1,
        lambda v: make_block(v, rescore_every=RS).lower(st, kT1).compile(),
        mode=a.ev, cache=a.tune_cache or None, rank=rank, size=size,
        share=os.path.join(os.path.abspath(a.out), 'ev_select'), say=say)
    snap = jax.jit(lambda st: (st.quat, st.trans, st.X[:, o:o + nbead])) \
        .lower(st).compile()
    t_compile = time.time() - t0 - t_setup
    n_snap = a.frames // a.thin if a.thin else 0
    writer = None
    if n_snap:
        writer = SnapshotWriter(out, n_snap, B, s.nb, nbead)
        writer.start()
    say(f'IMP {IMP.get_module_version()}, JAX {jax.__version__}; {size} '
        f'ranks x {B} chains, blocks of {F} frames; this rank: GPU {gpu} '
        f'({jax.devices()[0]}), local rank {local}; setup {t_setup:.1f} s, '
        f'compile {t_compile:.1f} s', all_ranks=True)

    # kT of frame f: 1 for the first frame, then the schedule of f - 1
    kT_all = np.array([1.0] + [sa_temperature(f) for f in
                               range(a.frames - 1)], np.float32)
    scores = np.lib.format.open_memmap(
        os.path.join(out, 'scores.npy'), 'w+', np.float32,
        (a.frames // K, B, 6))
    drift = np.zeros((a.frames, 2), np.float32)
    top = []                                      # (E, chain, frame, dofs)

    def process(k, rec):
        f0 = k * F
        e, tr, dmax, dmed, best = jax.device_get(rec)
        sel = [j for j in range(F) if (f0 + j + 1) % K == 0]
        for j in sel:
            i = (f0 + j + 1) // K - 1
            scores[i, :, 0] = e[j]
            scores[i, :, 1] = kT_all[f0 + j]
            scores[i, :, 2:] = tr[j]
        drift[f0:f0 + F] = np.stack([dmax, dmed], -1)
        idx, eb, qb, tb, bb = best
        for j in range(F):
            for m in range(idx.shape[1]):
                if len(top) < 10 or eb[j, m] < top[-1][0]:
                    top.append((float(eb[j, m]), int(idx[j, m]), f0 + j,
                                (qb[j, :, :, m], tb[j, :, :, m],
                                 bb[j, :, :, m].T)))
                    top.sort(key=lambda x: x[0])
                    del top[10:]

    nblk = a.frames // F
    t_start = time.time()
    pending = None
    for k in range(nblk):
        st, rec = block(st, jnp.asarray(kT_all[k * F:(k + 1) * F]))
        f_end = (k + 1) * F
        if n_snap and f_end % a.thin == 0:
            writer.put((f_end // a.thin - 1, f_end - 1, snap(st)))
        if pending is not None:
            process(*pending)
        pending = (k, rec)
        if (k + 1) % max(1, nblk // 10) == 0:
            say(f'  block {k + 1}/{nblk} dispatched; wall '
                f'{time.time() - t_start:.1f} s')
    process(*pending)
    jax.block_until_ready(st.energy)
    t_sample = time.time() - t_start
    if writer is not None:
        writer.put(None)
        writer.join()
        if writer.error is not None:
            raise RuntimeError(f'snapshot writer failed: {writer.error!r}')
    scores.flush()
    t_total = time.time() - t_start
    np.save(os.path.join(out, 'drift.npy'), drift)
    np.savez(os.path.join(out, 'best.npz'),
             energy=np.array([x[0] for x in top]),
             chain=np.array([x[1] for x in top]),
             frame=np.array([x[2] for x in top]),
             quat=np.array([x[3][0] for x in top]),
             trans=np.array([x[3][1] for x in top]),
             bead=np.array([x[3][2] for x in top]))
    acc = np.asarray(st.accepted).sum(1) / (B * a.frames * 10)
    steps = B * a.frames * 10 * smp.n_movers
    res = dict(rank=rank, size=size, local_rank=local, gpu=gpu,
               device=str(jax.devices()[0]), chains=B, frames=a.frames,
               block=F, thin=a.thin, cov=a.cov, score_every=K,
               ev_res=S.ev_resolution, ev=ev_info, rescore=a.rescore,
               steps=int(steps), setup_s=t_setup, compile_s=t_compile,
               sample_s=t_sample, total_s=t_total,
               mc_steps_per_s=steps / t_total,
               snapshot_writer_busy_s=writer.busy if writer else 0.,
               drift_max=float(np.nanmax(drift[:, 0])),
               drift_median=float(np.nanmedian(drift[:, 1])),
               acceptance_beads=float(acc[[i for i, p in
                                           enumerate(smp.steps)
                                           if p['kind'] == 'bead']].mean()),
               acceptance_rb=[float(acc[i]) for i, p in enumerate(smp.steps)
                              if p['kind'] == 'rb'],
               best_energy=top[0][0])
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    say(f'done: {res["mc_steps_per_s"] / 1e6:.1f}M MC steps/s on this rank '
        f'including output; sampling {t_sample:.1f} s, total {t_total:.1f} s'
        f' (snapshot writer busy {res["snapshot_writer_busy_s"]:.1f} s)',
        all_ranks=True)
    if rank == 0:
        top_dir = os.path.abspath(a.out)
        paths = [os.path.join(top_dir, f'rank{r}', 'summary.json')
                 for r in range(size)]
        while not all(os.path.exists(p_) for p_ in paths):
            time.sleep(1)
        time.sleep(1)
        R = [json.load(open(p_)) for p_ in paths]
        tot = sum(x['steps'] for x in R)
        wall = max(x['total_s'] for x in R)
        merged = dict(ranks=size, chains_per_rank=B, frames=a.frames,
                      block=F, thin=a.thin, score_every=K, steps=tot,
                      ev_res=S.ev_resolution, rescore=a.rescore,
                      ev_mode=[x['ev']['mode'] for x in R],
                      mc_steps_per_s_aggregate=tot / wall, total_s_max=wall,
                      per_rank=[{k: x[k] for k in (
                          'rank', 'gpu', 'device', 'mc_steps_per_s',
                          'total_s', 'drift_max')} for x in R])
        json.dump(merged, open(os.path.join(top_dir, 'summary.json'), 'w'),
                  indent=1)
        print(f'[rank 0] {size} ranks: {tot / wall / 1e6:.1f}M MC steps/s '
              f'aggregate including output ({tot:.3e} steps in {wall:.1f} s)',
              flush=True)


if __name__ == '__main__':
    main()
