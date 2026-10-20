"""The imp_tutorial on several GPUs, one MPI rank per GPU, through IMP.mpi.

    mpirun -np R python run_mpi.py --mode independent|rex --chains B \\
        --frames F [--thin T --writers W] --out runs/x

Every rank holds B chains of the fused sampler on its own GPU (the rank's
local index on its node, or --gpus). The rank, the number of ranks and the
local rank come from the Open MPI environment first, because the RMF writer
processes have to be spawned before MPI is initialised (Open MPI does not
support fork after MPI_Init). IMP.mpi.ReplicaExchange then initialises MPI
and must agree.

--mode independent
    R x B independent copies of the tutorial's run (no MPI: one replica with
    PMI's annealing schedule), chain seeds differing between ranks.
--mode rex
    PMI's replica exchange (what `mpirun -np R python modeling.py` does,
    with annealing off), with B independent copies per rank:
    - rank r samples its chains at T_r (IMP.mpi create_temperatures(tmin,
      tmax, R), geometric);
    - after every --exchange-every frames each rank pairs with its IMP.mpi
      friend (get_friend_index(step): even/odd neighbours, 0 <-> R-1
      wrap-around);
    - chain slot i of the pair (r, f) is accepted with IMP.mpi's rule, a
      uniform u <= exp((E_r,i - E_f,i)(1/T_r - 1/T_f)). Both ranks draw the
      same u (an RNG seeded by the seed, the step and the lower rank), so
      they reach the same decisions without another message;
    - accepted slots exchange their configurations (all
      configuration-dependent sampler state), sent through IMP.mpi's
      parameters (doubles, which carry the float32 values exactly).
    Temperatures stay with the ranks, so rank 0 always holds T_min. PMI
    swaps temperatures instead, which gives the same distribution. A walker
    id per slot follows the configurations (walkers.npy).

Outputs: OUT/rank<r>/ (scores.npy, drift.npy, rmfs/, summary.json; rex:
walkers.npy, swaps.npy), and OUT/summary.json from rank 0 with the totals.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from run_chains import sa_temperature, writer_main

CONFIG_FIELDS = ('X', 'quat', 'trans', 'energy', 'em', 'es_bead', 'es_body')


def rank_info():
    e = os.environ
    for r, s, l_ in (('OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE',
                      'OMPI_COMM_WORLD_LOCAL_RANK'),
                     ('PMI_RANK', 'PMI_SIZE', 'MPI_LOCALRANKID')):
        if r in e:
            return int(e[r]), int(e[s]), int(e.get(l_, e[r]))
    return 0, 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['independent', 'rex'],
                    default='independent')
    ap.add_argument('--chains', type=int, default=1024,
                    help='chains per rank')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--thin', type=int, default=0,
                    help='RMF frame every THIN frames (0: no RMF)')
    ap.add_argument('--writers', type=int, default=2,
                    help='RMF writer processes per rank')
    ap.add_argument('--score-every', type=int, default=1)
    ap.add_argument('--tmin', type=float, default=1.0)
    ap.add_argument('--tmax', type=float, default=2.5,
                    help='rex: the tutorial\'s replica_exchange_maximum_'
                         'temperature')
    ap.add_argument('--exchange-every', type=int, default=1,
                    help='rex: frames between exchange attempts (0: none, '
                         'each rank samples at its own kT; a reference)')
    ap.add_argument('--check-swaps', action='store_true',
                    help='rex: after every swap, rescore the received '
                         'configurations and compare with the energies the '
                         'friend sent (data-integrity check; slower)')
    ap.add_argument('--gpus', default=None,
                    help='comma-separated GPU per local rank (default: the '
                         'local rank)')
    ap.add_argument('--cov', default='rotate', choices=['rotate', 'frozen'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rank, size, local = rank_info()
    gpu = a.gpus.split(',')[local] if a.gpus else str(local)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    out = os.path.join(os.path.abspath(a.out), f'rank{rank}')
    os.makedirs(os.path.join(out, 'rmfs'), exist_ok=True)
    B, K = a.chains, a.score_every
    if a.thin and a.thin % K:
        raise SystemExit('--thin must be a multiple of --score-every')
    import resource                 # one RMF file per chain in the writers
    s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    log = open(os.path.join(out, 'run.log'), 'a')

    def say(*x, all_ranks=False):
        msg = ' '.join(str(y) for y in x)
        log.write(msg + '\n')
        log.flush()
        if rank == 0 or all_ranks:
            print(f'[rank {rank}] {msg}', flush=True)

    # writers before MPI and JAX initialise
    writers, q_ins = [], []
    ctx = mp.get_context('spawn')
    q_done = ctx.Queue()
    if a.thin:
        for w, part in enumerate(np.array_split(np.arange(B), a.writers)):
            qi = ctx.Queue(maxsize=8)
            p = ctx.Process(target=writer_main, args=(
                w, part.tolist(), out, a.cov, qi, q_done), daemon=True)
            p.start()
            writers.append((p, part))
            q_ins.append(qi)

    if a.mode == 'rex' and a.exchange_every and size % 2:
        # IMP.mpi's friend pattern pairs the replicas only for an even
        # count (with an odd one the 0 <-> R-1 wrap-around collides and the
        # exchange deadlocks); PMI refuses odd counts for the same reason
        raise SystemExit('--mode rex with exchanges needs an even number '
                         'of ranks')
    import IMP.mpi
    rem = IMP.mpi.ReplicaExchange()          # MPI_Init
    assert (rem.get_number_of_replicas(), rem.get_my_index()) == (size, rank)
    t0 = time.time()
    import io
    import contextlib
    import jax
    import jax.numpy as jnp
    import IMP
    import scoring as T
    from sampler import TutSampler
    from build_system import build
    with contextlib.redirect_stdout(io.StringIO()):
        S = build(seed=1, workdir=os.path.join(out, '.main'))
    s = T.TutSystem(S, cov_mode=a.cov)
    smp = TutSampler(s)
    st = smp.initial_state(B, kT=1.0, seed=a.seed + 7919 * rank)
    run = smp.make_run(10)
    temps = np.array(rem.create_temperatures(a.tmin, a.tmax, size)) \
        if size > 1 else np.array([a.tmin])
    kT_rank = float(temps[rank]) if a.mode == 'rex' else 1.0
    st.kT = jnp.full(B, kT_rank, smp.dt)
    t_setup = time.time() - t0
    jax.block_until_ready(run(st).energy)
    smp.resync(st, terms=True)
    t_compile = time.time() - t0 - t_setup
    import queue

    def from_writers():
        while True:          # a writer that died must not hang the rank
            try:
                return q_done.get(timeout=10)
            except queue.Empty:
                if not all(p.is_alive() for p, _ in writers):
                    raise RuntimeError(f'rank {rank}: an RMF writer died')
    for _ in writers:
        from_writers()
    say(f'IMP {IMP.get_module_version()}, JAX {jax.__version__}; {size} '
        f'ranks x {B} chains, mode {a.mode}; this rank: GPU {gpu} '
        f'({jax.devices()[0]}), local rank {local}, kT '
        f'{kT_rank if a.mode == "rex" else "annealing"}; setup '
        f'{t_setup:.1f} s, compile {t_compile:.1f} s', all_ranks=True)

    nrec = a.frames // K
    scores = np.zeros((nrec, B, 6), np.float32)
    drift = np.zeros((a.frames, 2), np.float32)
    walker = rank * B + np.arange(B)            # walker id in each slot
    walkers = np.zeros((nrec, B), np.int32) if a.mode == 'rex' else None
    swaps = []                                   # (step, friend, accepted)
    swap_check = [0., 0, 0.]          # max |rescored - sent|, n, max in ulps
    kT = np.full(B, kT_rank)
    t_ex = 0.
    t_start = time.time()
    for f in range(a.frames):
        st = run(st)
        e_track = st.energy
        st, terms = smp.resync(st, terms=True)
        dd = jnp.abs(e_track - st.energy)
        e, tr, dmax, dmed = jax.device_get((st.energy, terms, jnp.max(dd),
                                            jnp.median(dd)))
        drift[f] = dmax, dmed
        if (f + 1) % K == 0:
            i = (f + 1) // K - 1
            scores[i, :, 0] = e
            scores[i, :, 1] = kT
            scores[i, :, 2:] = tr
            if walkers is not None:
                walkers[i] = walker
        if a.thin and (f + 1) % a.thin == 0:
            q, t_, b = smp.dofs(st)
            for (p, part), qi in zip(writers, q_ins):
                qi.put((f, q[part], t_[part], b[part]))
        if a.mode == 'independent':
            kT[:] = sa_temperature(f)
            st.kT = jnp.full(B, kT[0], smp.dt)
        elif size > 1 and a.exchange_every and \
                (f + 1) % a.exchange_every == 0:
            tx = time.time()
            step = (f + 1) // a.exchange_every - 1
            fr = rem.get_friend_index(step)
            if fr != rank:
                e64 = np.asarray(e, np.float64)
                rem.set_my_parameter('E', e64)       # numpy: no list copy
                fe = np.asarray(rem.get_friend_parameter('E', fr))
                delta = (e64 - fe) * (1. / kT_rank - 1. / temps[fr])
                u = np.random.default_rng(
                    [a.seed, step, min(rank, fr)]).random(B)
                acc = u <= np.exp(np.minimum(delta, 0.))
                idx = np.nonzero(acc)[0]
                if len(idx):
                    # whole fields to the host and back (fixed shapes: an
                    # index of varying length would recompile every step)
                    full = [np.array(x) for x in jax.device_get(
                        [getattr(st, n) for n in CONFIG_FIELDS])]
                    vec = np.concatenate([x[idx].astype(np.float64).ravel()
                                          for x in full]
                                         + [walker[idx].astype(np.float64)])
                    rem.set_my_parameter('C', vec)
                    fv = np.asarray(rem.get_friend_parameter('C', fr))
                    o = 0
                    for n, x in zip(CONFIG_FIELDS, full):
                        k = len(idx) * int(np.prod(x.shape[1:]))
                        x[idx] = fv[o:o + k].reshape(
                            (len(idx),) + x.shape[1:]).astype(x.dtype)
                        setattr(st, n, jnp.asarray(x))
                        o += k
                    walker[idx] = fv[o:o + len(idx)].astype(np.int64)
                    if a.check_swaps:
                        sent = full[CONFIG_FIELDS.index('energy')][idx]
                        st = smp.resync(st)
                        got = np.asarray(jax.device_get(st.energy))[idx]
                        swap_check[0] = max(swap_check[0], float(
                            np.abs(got - sent).max()))
                        swap_check[1] += len(idx)
                        # in float32 ulps of the sent energy
                        swap_check[2] = max(swap_check[2], float(
                            (np.abs(got - sent) / np.spacing(
                                np.abs(sent).astype(np.float32))).max()))
                swaps.append((step, fr, len(idx)))
            t_ex += time.time() - tx
        if (f + 1) % max(1, a.frames // 10) == 0:
            say(f'  frame {f + 1}/{a.frames}: median score {np.median(e):.1f}'
                f', best {e.min():.1f}; wall {time.time() - t_start:.1f} s'
                + (f'; exchange {t_ex:.1f} s' if a.mode == 'rex' else ''))
    jax.block_until_ready(st.energy)
    t_sample = time.time() - t_start
    for qi in q_ins:
        qi.put(None)
    busy = [from_writers()[2] for _ in writers]
    for p, _ in writers:
        p.join()
    np.save(os.path.join(out, 'scores.npy'), scores)
    np.save(os.path.join(out, 'drift.npy'), drift)
    if walkers is not None:
        np.save(os.path.join(out, 'walkers.npy'), walkers)
        np.save(os.path.join(out, 'swaps.npy'), np.array(swaps, np.int64)
                .reshape(-1, 3))
    steps = B * a.frames * 10 * smp.n_movers
    sw = np.array(swaps, np.int64).reshape(-1, 3)
    res = dict(rank=rank, size=size, local_rank=local, gpu=gpu,
               device=str(jax.devices()[0]), mode=a.mode, chains=B,
               frames=a.frames, steps=int(steps), kT=kT_rank,
               setup_s=t_setup, compile_s=t_compile, sample_s=t_sample,
               exchange_s=t_ex, mc_steps_per_s=steps / t_sample,
               writer_busy_s=busy, drift_max=float(drift[:, 0].max()),
               swap_check_max_abs=swap_check[0] if a.check_swaps else None,
               swap_check_n=swap_check[1] if a.check_swaps else None,
               swap_check_max_ulps=swap_check[2] if a.check_swaps else None,
               swap_attempts=int(len(sw) * B),
               swap_accepted=int(sw[:, 2].sum()) if len(sw) else 0,
               swap_by_friend={str(int(fr)): float(
                   sw[sw[:, 1] == fr, 2].sum()
                   / max((sw[:, 1] == fr).sum() * B, 1))
                   for fr in np.unique(sw[:, 1])} if len(sw) else {})
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    say(f'done: {res["mc_steps_per_s"] / 1e6:.1f}M MC steps/s on this rank, '
        f'sampling {t_sample:.1f} s (exchange {t_ex:.1f} s)', all_ranks=True)
    if rank == 0:          # merge (ranks finish at different times)
        top = os.path.abspath(a.out)
        paths = [os.path.join(top, f'rank{r}', 'summary.json')
                 for r in range(size)]
        while not all(os.path.exists(p_) for p_ in paths):
            time.sleep(1)
        time.sleep(1)
        R = [json.load(open(p_)) for p_ in paths]
        tot = sum(x['steps'] for x in R)
        wall = max(x['sample_s'] for x in R)
        merged = dict(mode=a.mode, ranks=size, chains_per_rank=B,
                      frames=a.frames, steps=tot,
                      temperatures=temps.tolist() if a.mode == 'rex'
                      else None,
                      mc_steps_per_s_aggregate=tot / wall,
                      sample_s_max=wall,
                      per_rank=[{k: x[k] for k in (
                          'rank', 'gpu', 'device', 'kT', 'mc_steps_per_s',
                          'sample_s', 'exchange_s', 'swap_accepted',
                          'swap_attempts', 'swap_by_friend', 'drift_max')}
                          for x in R])
        json.dump(merged, open(os.path.join(top, 'summary.json'), 'w'),
                  indent=1)
        print(f'[rank 0] {size} ranks: {tot / wall / 1e6:.1f}M MC steps/s '
              f'aggregate ({tot:.3e} steps in {wall:.1f} s)', flush=True)


if __name__ == '__main__':
    main()
