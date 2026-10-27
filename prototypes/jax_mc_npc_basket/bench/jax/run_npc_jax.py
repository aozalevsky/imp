"""Run an NPC_Basket modeling script with both REX stages sampled by the
fused JAX sampler, for many chains in one process (GPU or CPU).

The modeling script runs unchanged; each IMP.pmi.macros.ReplicaExchange
.execute_macro() is replaced by:
  * N chains (--chains) from the current configuration, each with its own
    random stream. With --rex K the chains form N/K replica-exchange
    ladders of K temperatures (geometric between the macro's minimum and
    maximum temperature; neighbour swaps after every frame, alternating
    even/odd pairs, as IMP.pmi/IMP.mpi do). --rex 1 (default) gives
    independent T=1 chains, which is what the published serial runs were.
  * one PMI frame = monte_carlo_steps x n_movers MC steps, as in PMI.
  * after REX1 every chain keeps its own configuration for REX2.
  * --one-stage skips REX1: the script still sets everything up, and the
    REX2 call samples with all restraints on (crosslinks, EM, excluded
    volume, ...) for --frames2 frames, straight from the shuffled start.
  * --shuffle-each gives every chain its own start: before the first
    sampled stage the script's shuffle_configuration() call is replayed for
    each chain with its own seed (otherwise all chains start from the
    script's one shuffle). The replay runs after the symmetry constraints
    exist: the constraints put the symmetry copies back in place, and the
    shuffle's collision check sees them there. The script's
    optimize_flexible_beads() is not replayed (23 s per chain in C++).

Checkpoints and restarts work as in GROMACS mdrun:
  * --cpi continues from --out/state.cpt if it exists (otherwise the run
    starts fresh), so the same command line runs, stops and continues a
    run any number of times. Continuing gives exactly the frames an
    uninterrupted run would (same GPU type and software versions).
  * --cpt M writes a checkpoint every M minutes (default 15) while the GPU
    keeps sampling; the previous one is kept as state_prev.cpt.
  * the run stops cleanly (finishes the frame, writes a checkpoint) on
    SIGTERM, SIGINT (Ctrl-C), SIGUSR1/2, a file STOP in --out made after
    the start, or after 99% of --maxh hours. A second Ctrl-C stops at once.
  * --frames2 -1 samples until stopped. Frame counts are totals for the
    stage: continuing with a larger --frames2 (or -1) extends a run.
  * output files are append-only; on restart they are cut back to the
    checkpointed frame and continue from there (frames written after the
    last checkpoint, e.g. before a power cut, are written again).
  * compiled programs are cached in --out/jax_cache, so a continued run
    starts sampling without compiling, and XLA's kernel autotuning is off
    (XLA_FLAGS=--xla_gpu_autotune_level=0 unless XLA_FLAGS sets a level), so
    runs with the same seed give the same frames.

Output (per stage, in --out/<stage>/, <stage> = rex1, rex2 or one_stage):
trace.jsonl (one line per frame with every chain's score and temperature),
traj.dof (every --thin'th frame of each ladder's T_min replica: score and
degrees of freedom, a few kB per chain and frame; layout.npz holds the
rest of the model state), and checks.jsonl (chain 0 / ladder 0 rescored by
C++ every --check-every frames).

RMF files are made from traj.dof afterwards, with the restraints PMI
registers for RMF output (crosslinks, Z-axial, membrane, excluded volume)
evaluated by C++ at every frame, as PMI writes them (--no-rmf-restraints
leaves them out); the run's settings are read from its checkpoint:
  --to-rmf STAGE --best N     the N lowest-score saved frames of all ladders
                              (STAGE/best_<N>.rmf3, listed in best_<N>.csv)
  --to-rmf STAGE --ladder I [--frames START:STOP:STEP]
                              one ladder's trajectory (STAGE/ladder_<I>.rmf3)
--rmf-live also writes chain_<i>.rmf3 while sampling, as before (one set
of files per continued run: chain_<i>.partNNNN.rmf3). An RMF file is only
complete once closed, so a killed run loses its live RMF files.

  JAX_PLATFORMS=cuda python run_npc_jax.py yeast --chains 128 \\
      --frames1 50 --frames2 200 --out runs/y128
  JAX_PLATFORMS=cuda python run_npc_jax.py yeast --chains 2048 \\
      --one-stage --shuffle-each --frames2 -1 --maxh 12 --cpi --out runs/w
  python run_npc_jax.py --out runs/w --to-rmf one_stage --best 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('system', nargs='?', default='yeast')
ap.add_argument('--chains', type=int, default=8)
ap.add_argument('--rex', type=int, default=1,
                help='replicas per REX ladder (1 = independent chains)')
ap.add_argument('--frames1', type=int, default=10,
                help='REX1 frames (not used with --one-stage)')
ap.add_argument('--frames2', type=int, default=10,
                help='frames of the last stage; -1: until stopped')
ap.add_argument('--thin', type=int, default=10)
ap.add_argument('--check-every', type=int, default=0)
ap.add_argument('--resync-every', type=int, default=10)
ap.add_argument('--cov-mode', default='rotate')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', required=True)
ap.add_argument('--sampler', type=int, default=2, choices=[1, 2],
                help='2: delta-based npc_sampler2 (default), 1: npc_sampler '
                     '(no checkpoints or traj.dof; RMF output only)')
ap.add_argument('--one-stage', action='store_true',
                help='skip REX1; sample all restraints from the start')
ap.add_argument('--shuffle-each', action='store_true',
                help='a separate shuffle of the start for every chain')
ap.add_argument('--no-rmf-restraints', action='store_true',
                help='RMF files without the restraints (faster output)')
ap.add_argument('--cpi', action='store_true',
                help='continue from --out/state.cpt if it exists')
ap.add_argument('--cpt', type=float, default=15.,
                help='minutes between checkpoints (0: only at the end)')
ap.add_argument('--maxh', type=float, default=-1.,
                help='stop after 99%% of this many hours (with a checkpoint)')
ap.add_argument('--rmf-live', action='store_true',
                help='also write per-chain RMF files while sampling')
ap.add_argument('--to-rmf', default=None, metavar='STAGE',
                help='write RMF files from STAGE/traj.dof, no sampling')
ap.add_argument('--best', type=int, default=0)
ap.add_argument('--ladder', type=int, action='append', default=[])
ap.add_argument('--frames', default='::',
                help='START:STOP:STEP of the saved frames for --ladder')
ap.add_argument('--jax-cache', default='',
                help='persistent compilation cache directory (default: '
                     '$JAX_COMPILATION_CACHE_DIR, else --out/jax_cache; '
                     '"none": off)')
a = ap.parse_args()
t_launch = time.time()
if a.to_rmf:
    # the run's settings come from its checkpoint (as trjconv reads the tpr)
    import npc_ckpt
    _m = npc_ckpt.load_meta(os.path.abspath(a.out))
    if _m is None:
        sys.exit(f'--to-rmf: no checkpoint in {a.out}')
    for _k, _v in _m['fingerprint'].items():
        setattr(a, _k, _v)
assert a.chains % a.rex == 0
a.out = os.path.abspath(a.out)
if a.sampler == 1:
    a.rmf_live, a.cpt = True, -1.
    if a.cpi or a.to_rmf:
        sys.exit('--cpi and --to-rmf need --sampler 2')
if a.frames1 < 0 and not a.one_stage:
    sys.exit('--frames1 must be finite (REX2 follows)')

# No kernel autotuning: XLA would otherwise pick some kernels (the small
# matrix products of the rotations) by timing them, so two compilations of
# the same program can round differently and runs with the same seed
# drift apart. Measured on an L40 (2,048 chains): no speed difference.
if 'xla_gpu_autotune_level' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '')
                               + ' --xla_gpu_autotune_level=0').strip()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
# Persistent compilation cache: a continued run loads the programs its
# first session compiled instead of compiling again (faster restarts, and
# the same kernels: XLA picks some kernels by timing them at compile time,
# so a fresh compile can round differently)
_cache = a.jax_cache or os.environ.get('JAX_COMPILATION_CACHE_DIR') or \
    os.path.join(a.out, 'jax_cache')
if _cache != 'none':
    jax.config.update('jax_compilation_cache_dir', _cache)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
import IMP  # noqa: E402
import IMP.core  # noqa: E402
import IMP.isd  # noqa: E402
import IMP.pmi.macros  # noqa: E402
import IMP.pmi.tools  # noqa: E402
import IMP.rmf  # noqa: E402
import RMF  # noqa: E402
from IMP.core import _jax_rigid  # noqa: E402

import imp_jax_npc  # noqa: E402
import npc_ckpt  # noqa: E402
import npc_setup  # noqa: E402
from npc_sampler import NPCSampler  # noqa: E402
from npc_sampler2 import NPCSampler2, _state_class  # noqa: E402

imp_jax_npc.install()
os.makedirs(a.out, exist_ok=True)
# one RMF file per chain/ladder is open for the whole stage: raise the
# open-file limit (Ubuntu's default soft limit is 1024)
import resource  # noqa: E402
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < _hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
L, K = a.chains // a.rex, a.rex
N = a.chains
stage_no = [0]
saved = {}          # per-chain full-model state between stages
log = open(os.path.join(a.out, 'run.log' if not a.to_rmf else 'to_rmf.log'),
           'a')

# what must not change when a run is continued; and what should not
FINGERPRINT = dict(system=a.system, chains=a.chains, rex=a.rex, seed=a.seed,
                   sampler=a.sampler, one_stage=a.one_stage,
                   shuffle_each=a.shuffle_each, cov_mode=a.cov_mode,
                   resync_every=a.resync_every)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ['npc_sampler2.py', 'imp_jax_npc.py', 'imp_jax_fused.py',
        'imp_jax_patches.py', 'run_npc_jax.py']
VERSIONS = dict(imp=IMP.get_module_version(), jax=jax.__version__,
                device=str(jax.devices()[0].device_kind),
                code_md5=npc_ckpt.file_md5(
                    [os.path.join(HERE, f) for f in CODE]))


def say(*x):
    msg = ' '.join(str(y) for y in x)
    print(msg, flush=True, file=sys.__stdout__)
    log.write(msg + '\n')
    log.flush()


# ---- checkpoint to continue from -------------------------------------
CK = None
if os.path.exists(os.path.join(a.out, npc_ckpt.CPT)) and not (
        a.cpi or a.to_rmf):
    sys.exit(f'{a.out} has a checkpoint: add --cpi to continue that run, '
             'or pick another --out')
if a.cpi or a.to_rmf:
    CK = npc_ckpt.load(a.out)
    if CK is None and a.cpi:
        say(f'--cpi: no checkpoint in {a.out}; starting a new run')
if CK is not None:
    meta = CK[0]
    diff = {k: (meta['fingerprint'].get(k), v) for k, v in FINGERPRINT.items()
            if meta['fingerprint'].get(k) != v}
    if diff:
        sys.exit(f'{a.out}/{meta["_file"]} is a run with other settings '
                 f'(checkpoint, now): {diff}')
    if not a.to_rmf:
        vd = {k: (meta['versions'].get(k), v) for k, v in VERSIONS.items()
              if meta['versions'].get(k) != v}
        if vd:
            say(f'note: {vd} differ from the checkpoint: the continued run '
                'samples correctly but is not bit-identical to an '
                'uninterrupted one')
        say(f'== continuing from {meta["_file"]}: stage {meta["name"]} after '
            f'{meta["frame_next"]} frames (session {meta["session"] + 1})')
SESSION = 0 if CK is None else CK[0]['session'] + 1
stopper = npc_ckpt.StopRequest(a.out, t_launch, a.maxh)


class _Stop(Exception):
    pass


def model_state(m):
    return (m.get_spheres_numpy()[0].copy(),
            m.get_numpy(_jax_rigid._RB_QUAT_KEY).copy(),
            m.get_numpy(IMP.isd.Scale.get_scale_key()).copy())


def set_model_state(m, st):
    xyz, q, n = st
    m.get_spheres_numpy()[0][:len(xyz)] = xyz
    m.get_numpy(_jax_rigid._RB_QUAT_KEY)[:len(q)] = q
    m.get_numpy(IMP.isd.Scale.get_scale_key())[:len(n)] = n
    m.update()


def saved_arrays():
    """the per-chain model states kept for the next stage, stacked"""
    return {f'saved/{j}': np.stack([saved[c][j] for c in range(N)])
            for j in range(3)}


def restore_saved(arrays):
    for c in range(N):
        saved[c] = tuple(arrays[f'saved/{j}'][c] for j in range(3))


# the script's shuffle, replayed for every chain by --shuffle-each
start_calls = []


def _recorded(fn):
    def call(*args, **kw):
        if stage_no[0] == 0:
            start_calls.append((fn, args, kw))
        return fn(*args, **kw)
    return call


IMP.pmi.tools.shuffle_configuration = _recorded(
    IMP.pmi.tools.shuffle_configuration)


def replay_start(m, seed):
    """Redo the script's shuffle with IMP's random numbers seeded by
       seed. Rigid bodies placed by symmetry constraints (none in the yeast
       script, whose copies are not rigid bodies) are left out:
       Model.update() in the shuffle's collision check would put them back
       each time, and it gives up after 100 tries."""
    IMP.random_number_generator.seed(seed)
    copies = [IMP.core.RigidBody(m, pi) for pi in m.get_particle_indexes()
              if IMP.core.RigidBody.get_is_setup(m, pi)
              and IMP.core.Reference.get_is_setup(m, pi)]
    for fn, args, kw in start_calls:
        kw = dict(kw, excluded_rigid_bodies=list(
            kw.get('excluded_rigid_bodies', [])) + copies)
        fn(*args, **kw)
    m.update()


def make_swap(temps):
    """Jitted temperature swaps within each ladder. chain_of_temp (L, K)
       holds, for each temperature slot, the (ladder-local) chain there.
       Pairs (i, i+1) with i % 2 == parity are tried, accepted with
       min(1, exp((b_i - b_i+1) (E_i - E_i+1)))."""
    beta = 1.0 / temps
    i = jnp.arange(K - 1)

    def swap(cot, scores, parity, key):
        E = scores.reshape(L, K)[jnp.arange(L)[:, None], cot]
        d = (beta[i] - beta[i + 1])[None] * (E[:, i] - E[:, i + 1])
        u = jax.random.uniform(key, (L, K - 1))
        acc = (i % 2 == parity)[None] & (jnp.log(u) < d)
        z = jnp.zeros((L, 1), bool)
        up = jnp.concatenate([acc, z], 1)        # slot t <-> t+1
        down = jnp.concatenate([z, acc], 1)      # slot t <-> t-1
        new = jnp.where(up, jnp.roll(cot, -1, 1),
                        jnp.where(down, jnp.roll(cot, 1, 1), cot))
        return new, acc
    return jax.jit(swap)


def rmf_files(macro, m, paths):
    """RMF files with the macro's hierarchy and, as PMI does, the
       restraints registered for RMF output"""
    rrs = IMP.pmi.tools.get_restraint_set(m, rmf=True)
    rmf_rs = ([] if a.no_rmf_restraints else
              [rrs.get_restraint(i)
               for i in range(rrs.get_number_of_restraints())])
    out = []
    for p in paths:
        fh = RMF.create_rmf_file(p)
        IMP.rmf.add_hierarchy(fh, macro.root_hier)
        if rmf_rs:
            IMP.rmf.add_restraints(fh, rmf_rs)
        out.append(fh)
    return out


def load_layout(sdir):
    with np.load(os.path.join(sdir, 'layout.npz')) as z:
        return {k: z[k] for k in z.files}


def to_rmf(macro, m, smp, sdir, name):
    """--to-rmf: RMF files from sdir/traj.dof"""
    lay = load_layout(sdir)
    h, rec = npc_ckpt.read_dof(os.path.join(sdir, 'traj.dof'))
    say(f'== {name}: traj.dof has {len(rec)} saved frames x {h["ladders"]} '
        f'ladders')
    jobs = []                  # (output name, [(record, slot, title)])
    if a.best:
        sc = np.asarray(rec['score'])                  # (frames, L)
        order = np.argsort(sc, axis=None, kind='stable')[:a.best]
        sel = [divmod(int(k), sc.shape[1]) for k in order]
        jobs.append((f'best_{a.best}', [
            (r, i, f'frame {int(rec["frame"][r])} ladder {i} chain '
                   f'{int(rec["chain"][r, i])} score {sc[r, i]:.2f}')
            for r, i in sel]))
        with open(os.path.join(sdir, f'best_{a.best}.csv'), 'w') as fh:
            fh.write('rank,frame,ladder,chain,score\n')
            for k, (r, i) in enumerate(sel):
                fh.write(f'{k},{int(rec["frame"][r])},{i},'
                         f'{int(rec["chain"][r, i])},{sc[r, i]:.6f}\n')
    sl = slice(*[int(x) if x else None for x in a.frames.split(':')])
    for i in a.ladder:
        jobs.append((f'ladder_{i}', [(r, i, str(int(rec['frame'][r])))
                                     for r in range(len(rec))[sl]]))
    if not jobs:
        sys.exit('--to-rmf: give --best N and/or --ladder I')
    import types
    for out_name, frames in jobs:
        fh, = rmf_files(macro, m, [os.path.join(sdir, out_name + '.rmf3')])
        for r, i, title in frames:
            xyz, nuis, quat = npc_ckpt.full_state_arrays(lay, rec[r], i)
            smp.write_model(types.SimpleNamespace(xyz=xyz, nuis=nuis,
                                                  quat=quat))
            IMP.rmf.save_frame(fh, title)
        del fh
        say(f'  wrote {out_name}.rmf3 ({len(frames)} frames)')


def run_stage(macro, g):
    stage_no[0] += 1
    stage = stage_no[0]
    if a.one_stage and stage == 1:
        say('== REX1 skipped (--one-stage)')
        return
    name = 'one_stage' if a.one_stage else f'rex{stage}'
    first = stage == 1 or a.one_stage
    last = stage >= 2
    sdir = os.path.join(a.out, name)
    m = macro.model
    ck_meta, ck_arr = CK if CK is not None else (None, None)
    target = a.frames1 if stage == 1 else a.frames2      # -1: no limit

    # ---- where this stage stands in a continued run -----------------
    if a.to_rmf:
        if a.to_rmf != name:
            return
        resume = False
    elif ck_meta is not None and stage < ck_meta['stage']:
        say(f'== {name}: finished before the checkpoint')
        return
    elif ck_meta is not None and stage == ck_meta['stage']:
        if target >= 0 and ck_meta['frame_next'] >= target:
            say(f'== {name}: {ck_meta["frame_next"]} frames done '
                f'(--frames{2 if last else 1} {target})')
            if not last:
                restore_saved(ck_arr)
                set_model_state(m, saved[0])
                return
            raise _Stop()
        resume = True
    else:
        resume = False
    if resume or a.to_rmf:
        # the model as it was when the stage started (the sampler is
        # built from it)
        lay = load_layout(sdir)
        set_model_state(m, tuple(lay[f'model0/{j}'] for j in range(3)))
    model0 = model_state(m)

    os.makedirs(sdir, exist_ok=True)
    rs = IMP.pmi.tools.get_restraint_set(m)
    t0 = time.time()
    smp = (NPCSampler if a.sampler == 1 else NPCSampler2)(
        m, rs, macro.monte_carlo_sample_objects, cov_mode=a.cov_mode)
    if a.to_rmf:
        to_rmf(macro, m, smp, sdir, name)
        raise _Stop()
    nsteps = macro.vars['monte_carlo_steps'] * smp.nmov
    tmin = macro.vars['replica_exchange_minimum_temperature']
    tmax = macro.vars['replica_exchange_maximum_temperature']
    temps = (jnp.geomspace(tmin, tmax, K) if K > 1
             else jnp.full(1, tmin)).astype(smp.dt)
    swap = make_swap(temps) if K > 1 else None

    def set_temps(sb, cot):
        temp_of_chain = jnp.zeros((L, K), jnp.int32).at[
            jnp.arange(L)[:, None], cot].set(jnp.arange(K)[None])
        sb.temperature = temps[temp_of_chain.reshape(-1)]
        return sb

    if resume:
        sb = npc_ckpt.state_from_arrays(_state_class(), ck_arr)
        chain_of_temp = jnp.asarray(ck_arr['chain_of_temp'])
        key = npc_ckpt.key_from_arrays('swap_key', ck_arr)
        f0 = ck_meta['frame_next']
        swaps_acc = np.array(ck_meta['swaps_acc'])
        swaps_try = np.array(ck_meta['swaps_try'])
    else:
        # initial states: the first stage from the current model (or from
        # each chain's own shuffle), REX2 from each chain's REX1 result
        if first and a.shuffle_each and not start_calls:
            raise RuntimeError('--shuffle-each: the script made no '
                               'shuffle_configuration() call to replay')
        states = []
        for c in range(N):
            if not first and c in saved:
                set_model_state(m, saved[c])
            elif first and a.shuffle_each:
                replay_start(m, a.seed * 100003 + c)
            states.append(smp.initial_state(seed=a.seed * 100003 + c))
        sb = jax.tree.map(lambda *x: jnp.stack(x), *states)
        del states
        chain_of_temp = jnp.tile(jnp.arange(K), (L, 1))  # ladder-local ids
        sb = set_temps(sb, chain_of_temp)
        key = jax.random.key(a.seed + 17 * stage)
        f0 = 0
        swaps_acc = np.zeros(max(K - 1, 1))
        swaps_try = np.zeros(max(K - 1, 1))
    say(f'== {name}: {N} chains ({L} ladders x {K}), '
        f'{"until stopped" if target < 0 else f"{target} frames"} x '
        f'{nsteps} steps, from frame {f0}; K={smp.K} rows, EV {smp.nev}, '
        f'EM {smp.M}x{smp.D}; setup {time.time() - t0:.1f}s; temps '
        f'{np.round(np.asarray(temps), 3)}')

    # ---- outputs (append-only; cut back to the checkpoint) -----------
    offsets = ck_meta['outputs'] if resume else {}

    def out_file(fname, cls=npc_ckpt.TextLog):
        rel = os.path.join(name, fname)
        return cls(os.path.join(a.out, rel), offsets.get(rel)
                   if resume else None)
    trace = out_file('trace.jsonl')
    checks = out_file('checks.jsonl')
    outfiles = [trace, checks]
    traj = None
    if a.sampler == 2:
        if resume:
            lay = load_layout(sdir)
            if npc_ckpt.file_md5([os.path.join(sdir, 'layout.npz')]) != \
                    ck_meta['layout_md5']:
                raise RuntimeError(f'{sdir}/layout.npz differs from the one '
                                   'the checkpoint was written with')
        else:
            lay = npc_ckpt.dof_layout(smp, sb, smp.nuis_dof_rows)
            lay.update({f'model0/{j}': x for j, x in enumerate(model0)})
            np.savez(os.path.join(sdir, 'layout.npz'), **lay)
        layout_md5 = npc_ckpt.file_md5([os.path.join(sdir, 'layout.npz')])
        traj = out_file('traj.dof', npc_ckpt.AppendFile)
        if not resume:
            traj.write(npc_ckpt.dof_header(lay, L, name, dict(
                thin=a.thin, nsteps=nsteps, fingerprint=FINGERPRINT)))
        outfiles.append(traj)
        nb, free = int(lay['nb']), jnp.asarray(lay['free_rows'])
        nrows = jnp.asarray(lay['nuis_rows'])
        rec_dt = npc_ckpt.dof_dtype(L, int(nb), len(lay['free_rows']),
                                    len(lay['nuis_rows']))
        gather = jax.jit(lambda sb, idx: (
            sb.quat[idx, :nb], sb.xyz[idx][:, free], sb.nuis[idx][:, nrows],
            sb.score[idx]))
    else:
        layout_md5 = None
    rmfs = []
    if a.rmf_live:
        part = '' if SESSION == 0 else f'.part{SESSION:04d}'
        rmfs = rmf_files(macro, m, [os.path.join(sdir, f'chain_{i}{part}.rmf3')
                                    for i in range(L)])

    t_out = 0.
    t_start = time.time()
    t_ckpt = [time.time()]
    last_ckpt = [f0 if resume else -1]       # frame of the newest state.cpt
    nframes = [0]                    # frames sampled in this session

    def checkpoint(fdone, sb_c, cot_c, key_c, extra=None):
        """state.cpt after frame fdone - 1 (fdone frames done); called once
           every earlier frame's output is written"""
        t = time.time()
        outputs = {os.path.relpath(fh.path, a.out): fh.mark()
                   for fh in outfiles}
        arrays = npc_ckpt.state_arrays(sb_c)
        arrays['chain_of_temp'] = np.asarray(cot_c)
        arrays.update(npc_ckpt.key_arrays('swap_key', key_c))
        arrays.update(extra or {})
        meta = dict(version=1, stage=stage, name=name, frame_next=fdone,
                    session=SESSION, outputs=outputs, layout_md5=layout_md5,
                    swaps_acc=swaps_acc.tolist(),
                    swaps_try=swaps_try.tolist(), fingerprint=FINGERPRINT,
                    versions=VERSIONS, written=time.strftime('%F %T'))
        npc_ckpt.save(a.out, meta, arrays)
        say(f'  checkpoint after frame {fdone} ({time.time() - t:.1f}s)')

    def process(f, sb_f, cot_f, acc_f, final=False):
        """Host side of frame f (trace, swap stats, traj, RMF, C++ check);
           runs while the device works on frame f+1. final: only the RMF
           frame of the last frame before a stop."""
        nonlocal t_out, swaps_acc, swaps_try
        scores = np.asarray(sb_f.score)
        t2 = time.perf_counter()
        if acc_f is not None:
            swaps_acc += np.asarray(acc_f).sum(0)
            swaps_try += (np.arange(K - 1) % 2 == f % 2) * L
        cot = np.asarray(cot_f)
        tmin_chain = np.arange(L) * K + cot[:, 0]    # global id per ladder
        if not final:
            trace.write(json.dumps(dict(
                frame=f, score=[float(x) for x in scores],
                tmin_chain=[int(x) for x in tmin_chain])) + '\n')
            if traj is not None and (f + 1) % a.thin == 0:
                q, x, n, sc = jax.device_get(
                    gather(sb_f, jnp.asarray(tmin_chain)))
                traj.write(npc_ckpt.dof_record(rec_dt, f, tmin_chain, sc,
                                               q, x, n))
        if rmfs and (final or (f + 1) % a.thin == 0 or f == target - 1):
            host = jax.device_get(sb_f)
            for i in range(L):
                c = int(tmin_chain[i])
                one = jax.tree.map(lambda x: x[c], host)
                smp.write_model(one, refresh_covariances=False)
                IMP.rmf.save_frame(rmfs[i], str(f))
        if a.check_every and not final and (f + 1) % a.check_every == 0:
            one = jax.tree.map(lambda x: x[int(tmin_chain[0])],
                               jax.device_get(sb_f))
            full = float(smp.terms(jax.tree.map(jnp.asarray, one))['total'])
            smp.write_model(one)
            e = rs.evaluate(False)
            checks.write(json.dumps(dict(frame=f, jax_running=float(one.score),
                                         jax_full=full, cpp=e)) + '\n')
        t_out += time.perf_counter() - t2
        every = max(1, target // 10) if target > 0 else 100
        if (f + 1) % every == 0:
            say(f'  frame {f + 1}{"" if target < 0 else f"/{target}"}: '
                f'T_min scores {np.round(scores[tmin_chain][:4], 1)}... best '
                f'{scores.min():.1f}; wall {time.time() - t_start:.1f}s, '
                f'host output {t_out:.1f}s')

    # the device runs ahead; a writer thread does each frame's host side
    # (it is the only thread touching the IMP Model during the loop) and
    # the periodic checkpoints
    import queue
    import threading
    work = queue.Queue(maxsize=2 * a.thin + 2)
    errors = []

    def writer():
        while True:
            item = work.get()
            if item is None:
                return
            try:
                if item[0] == 'ckpt':
                    checkpoint(*item[1:])
                else:
                    process(*item[1:])
            except Exception as e:          # report in the main thread
                errors.append(e)
                return
    th = threading.Thread(target=writer, daemon=True)
    th.start()
    f = f0
    prev = None
    while target < 0 or f < target:
        if errors:
            break
        if stopper.check():
            break
        sb = smp.run_batched(sb, nsteps)
        # at most two frames in flight on the device: it always has the
        # next frame queued, and a stop request waits for two frames at
        # most (the writer queue only holds finished frames)
        if prev is not None:
            prev.block_until_ready()
        prev = sb.score
        if a.resync_every and (f + 1) % a.resync_every == 0:
            sb = smp.resync_batched(sb)
        acc_low = None
        if swap is not None:
            key, k1 = jax.random.split(key)
            chain_of_temp, acc_low = swap(chain_of_temp, sb.score, f % 2, k1)
            sb = set_temps(sb, chain_of_temp)
        work.put(('frame', f, sb, chain_of_temp, acc_low))
        f += 1
        nframes[0] += 1
        if a.cpt > 0 and time.time() - t_ckpt[0] >= 60 * a.cpt:
            work.put(('ckpt', f, sb, chain_of_temp, key))
            t_ckpt[0] = time.time()
            last_ckpt[0] = f
    stopped = stopper.reason
    if stopped and rmfs and f > f0 and f % a.thin:
        work.put(('frame', f - 1, sb, chain_of_temp, None, True))
    work.put(None)
    th.join()
    if errors:
        raise errors[0]
    if stopped:
        say(f'  {stopped}: {name} stops after {f} frames')
    t_dev = time.time() - t_start - t_out
    wall = time.time() - t_start
    done = f
    complete = target >= 0 and done >= target
    extra = None
    if complete and not last:
        # every chain's final configuration, for the next stage
        host = jax.device_get(sb)
        for c in range(N):
            smp.write_model(jax.tree.map(lambda x: x[c], host),
                            refresh_covariances=False)
            saved[c] = model_state(m)
        extra = saved_arrays()
    if a.cpt >= 0 and a.sampler == 2 and (done != last_ckpt[0] or extra):
        checkpoint(done, sb, chain_of_temp, key, extra)
    for fh in outfiles:
        fh.close()
    rmfs.clear()                      # closes the RMF files
    res = dict(stage=stage, name=name, session=SESSION, chains=N, ladders=L,
               replicas=K, frames=done, frames_this_run=nframes[0],
               nsteps=nsteps, wall=wall, t_host_output=t_out,
               t_device=t_dev,
               steps_per_s_wall=nframes[0] * nsteps * N / wall,
               steps_per_s_per_chain_wall=nframes[0] * nsteps / wall,
               swap_acc=list(swaps_acc / np.maximum(swaps_try, 1)),
               best=float(np.min(np.asarray(sb.score))),
               stopped=stopped)
    say('  ' + json.dumps(res))
    with open(os.path.join(a.out, 'summary.jsonl'), 'a') as fh:
        fh.write(json.dumps(dict(res, system=a.system,
                                 backend=jax.default_backend())) + '\n')
    if stopped or last:
        raise _Stop()
    # leave the Model at chain 0 so the script can continue
    set_model_state(m, saved[0])


def execute(self):
    return run_stage(self, sys._getframe(1).f_globals)


IMP.pmi.macros.ReplicaExchange.execute_macro = execute
cfg = npc_setup.SYSTEMS[a.system]
out = os.path.abspath(os.path.join(a.out, 'imp_output'))
os.makedirs(out, exist_ok=True)
import shutil  # noqa: E402
for fn in os.listdir(cfg['gmm']):
    shutil.copy(os.path.join(cfg['gmm'], fn), out)
IMP.random_number_generator.seed(a.seed)
np.random.seed(a.seed)
os.chdir(cfg['wd'])
sys.argv = [cfg['script'], '-d', cfg['data'], '-o', out]
sys.path.insert(0, os.path.dirname(cfg['script']))
say(f'IMP {IMP.get_module_version()}, JAX {jax.__version__} on '
    f'{jax.devices()[0]}; {a.system}: {N} chains, rex {K}; session {SESSION}; '
    f'XLA_FLAGS={os.environ.get("XLA_FLAGS", "")}; compile cache '
    f'{jax.config.jax_compilation_cache_dir}')
t0 = time.time()
try:
    import runpy
    import contextlib
    with open(os.path.join(a.out, 'script_stdout.txt'),
              'w' if SESSION == 0 and not a.to_rmf else 'a') as so, \
            contextlib.redirect_stdout(so):
        runpy.run_path(cfg['script'], run_name='__main__')
except _Stop:
    pass
say(f'total wall time {time.time() - t0:.1f}s')
