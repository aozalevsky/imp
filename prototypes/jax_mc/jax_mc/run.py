"""Production runs for any sampler that follows the protocol (DESIGN.md).

  python -m jax_mc.run polii --chains 1024 --frames 1000 --out runs/x
      [--sweeps 10] [--kT 1 | --ladder [1,1.5,...] | --anneal 1,2.5,200,20]
      [--rescore-every K] [--thin T] [--block F] [--seed S]
      [--sampler my.module:factory] [--float64]
      [--cpi] [--cpt MINUTES] [--maxh HOURS] [--jax-cache DIR]

A frame is --sweeps sweeps (collective samplers: steps). Frames run in
blocks of --block inside one jitted call, and the next block is dispatched
before the previous block's records are written. Per frame:
  * temperatures: fixed (--kT), PMI's simulated annealing (--anneal tmin,
    tmax,nmin,nmax: tmin for nmin frames, then tmax for nmax), or replica
    exchange (--ladder: the chains form C / len(ladder) independent
    ladders of consecutive chains; after each frame neighbouring rungs try
    to swap, even and odd pairs on alternate frames);
  * every K-th frame the energies are rescored from scratch (the sampler's
    resync, R10) and the largest |tracked - rescored| is recorded;
  * records: frame, drift, then the total energy and kT of every chain.
Every --thin frames the degrees of freedom of every chain are written.

Outputs in OUT/ (append-only, so runs can be open-ended):
  run.json       settings and the record layouts
  records.f64    per frame: frame, drift, energy (C), kT (C)
  dofs.f32       per snapshot: frame, then per chain quat (nb, 4),
                 trans (nb, 3), bead (nbead, 3), nuis (N)
  state.cpt      the last checkpoint (sampler state, frame, ladder)
read with jax_mc.run.records(OUT) and jax_mc.run.snapshots(OUT).

Checkpoints (GROMACS-style): --cpt minutes between checkpoints, written
by a thread from a device copy while sampling goes on; one is always
written at the end. --cpi continues from OUT/state.cpt if it exists (the
outputs are cut back to it); --maxh stops after 99% of the hours; a STOP
file in OUT, SIGINT or SIGTERM stop after the current block. Every program
is kept in a persistent compile cache (OUT/jax_cache, or --jax-cache), so
a restarted session runs the same executables: XLA does not always compile
a program bit-identically twice. The Python hash seed is pinned to 0 (PMI
builds systems in set order).

The sampler's state must be a dict with 'kT' (C,).
"""
import argparse
import json
import math
import os
import pickle
import signal
import threading
import time

import numpy as np

REX_TAG = 0xFFFFFFFF            # fold_in data for the exchange draws


# ---------------- temperature policies (device side) -------------------------
def rex_swap(E, rung, ladder, frame, key):
    """one exchange round. E (C,) energies, rung (C,) each chain's place
       in its ladder (chains c // R form ladder c // R, R = len(ladder));
       returns the new rungs. Swap (r, r + 1) with r % 2 == frame % 2 is
       accepted if u < exp((1/kT_r - 1/kT_{r+1}) (E_at_r - E_at_r+1))."""
    import jax
    import jax.numpy as jnp
    C, R = E.shape[0], ladder.shape[0]
    L = C // R
    grp = jnp.arange(C) // R
    inv = jnp.zeros((L, R), jnp.int32).at[grp, rung].set(
        jnp.arange(C, dtype=jnp.int32))
    r = jnp.arange(R - 1)
    a, b = inv[:, r], inv[:, r + 1]                         # (L, R - 1)
    db = 1. / ladder[r] - 1. / ladder[r + 1]
    u = jax.random.uniform(jax.random.fold_in(key, frame), (L, R - 1),
                           E.dtype)
    acc = (r % 2 == frame % 2) & (jnp.log(u) < db * (E[a] - E[b]))
    rr = jnp.broadcast_to(r, a.shape).astype(rung.dtype)
    # only accepted pairs are written: active pairs are disjoint, but an
    # inactive pair shares a chain with its active neighbours
    ia, ib = jnp.where(acc, a, C).ravel(), jnp.where(acc, b, C).ravel()
    rung = rung.at[ia].set((rr + 1).ravel(), mode='drop')
    return rung.at[ib].set(rr.ravel(), mode='drop')


def anneal_kT(frame, tmin, tmax, nmin, nmax):
    """IMP.pmi.samplers._SamplerBase.temp_simulated_annealing"""
    import jax.numpy as jnp
    return jnp.where(frame % (nmin + nmax) < nmin, tmin, tmax)


# ---------------- the jitted block -------------------------------------------
def make_block(smp, sweeps, n_frames, rescore_every=1, ladder=None,
               anneal=None, seed=0, snapshot=True):
    """f(st, rung, frame0) -> st, rung, records, dofs (of the last frame).
       records: drift (F,), energy (F, C), kT (F, C)."""
    import jax
    import jax.numpy as jnp
    K = rescore_every
    can_rescore = hasattr(smp, 'resync')
    lad = None if ladder is None else jnp.asarray(ladder)
    rkey = jax.random.fold_in(jax.random.key(seed), REX_TAG)

    def frame(carry, f):
        st, rung = carry
        if anneal is not None:
            st = dict(st, kT=jnp.full_like(st['kT'], anneal_kT(f, *anneal)))
        st = smp.run(st, sweeps)
        e = smp.total(st)
        drift = jnp.array(jnp.nan, e.dtype)
        if can_rescore and K > 0:
            def rescore(st):
                st = smp.resync(st)
                e2 = smp.total(st)
                return st, e2, jnp.max(jnp.abs(e2 - e))
            st, e, drift = jax.lax.cond(
                f % K == K - 1, rescore, lambda st: (st, e, drift), st)
        kT = st['kT']
        if lad is not None:
            rung = rex_swap(e, rung, lad, f, rkey)
            st = dict(st, kT=lad[rung].astype(kT.dtype))
        return (st, rung), (drift, e, kT)

    def block(st, rung, f0):
        (st, rung), rec = jax.lax.scan(
            frame, (st, rung), f0 + jnp.arange(n_frames))
        d = smp.dofs(st) if snapshot else None
        return st, rung, rec, d
    return jax.jit(block, donate_argnums=0)


# ---------------- outputs ----------------------------------------------------
def dof_layout(dofs):
    """(name, per-chain shape) in output order, from chain-last dofs"""
    out = []
    for k in ('quat', 'trans', 'bead', 'nuis'):
        sh = tuple(dofs[k].shape)
        out.append((k, (sh[1], sh[0]) if len(sh) == 3 else (sh[0],)))
    return out


def _flat_dofs(frame, d):
    """one snapshot record: frame, then per chain its dofs (float32)"""
    C = d['quat'].shape[-1]
    parts = []
    for k in ('quat', 'trans', 'bead', 'nuis'):
        a = np.asarray(d[k], np.float32)
        a = np.moveaxis(a, -1, 0)                          # chain first
        if a.ndim == 3:
            a = np.swapaxes(a, 1, 2)                       # (C, n, 3|4)
        parts.append(a.reshape(C, -1))
    return np.concatenate([np.array([frame], np.float32),
                           np.concatenate(parts, 1).ravel()])


def records(out):
    """frame (F,), drift (F,), energy (F, C), kT (F, C)"""
    meta = json.load(open(os.path.join(out, 'run.json')))
    C = meta['chains']
    r = np.fromfile(os.path.join(out, 'records.f64')).reshape(-1, 2 + 2 * C)
    return r[:, 0].astype(np.int64), r[:, 1], r[:, 2:2 + C], r[:, 2 + C:]


def snapshots(out):
    """frames (S,), {name: (S, C, *shape)}"""
    meta = json.load(open(os.path.join(out, 'run.json')))
    C, lay = meta['chains'], meta['dof_layout']
    n = sum(int(np.prod(s)) for _, s in lay)
    a = np.fromfile(os.path.join(out, 'dofs.f32'), np.float32).reshape(
        -1, 1 + C * n)
    S = a.shape[0]
    frames, body = a[:, 0].astype(np.int64), a[:, 1:].reshape(S, C, n)
    res, o = {}, 0
    for k, sh in lay:
        m = int(np.prod(sh))
        res[k] = body[:, :, o:o + m].reshape((S, C) + tuple(sh))
        o += m
    return frames, res


# ---------------- checkpoints ------------------------------------------------
def _to_host(tree):
    import jax
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = _to_host(v)
        elif hasattr(v, 'dtype') and jax.dtypes.issubdtype(
                v.dtype, jax.dtypes.prng_key):
            out[k + '@key'] = np.asarray(jax.random.key_data(v))
        else:
            out[k] = jax.device_get(v)
    return out


def _to_device(tree):
    import jax
    import jax.numpy as jnp
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = _to_device(v)
        elif k.endswith('@key'):
            out[k[:-4]] = jax.random.wrap_key_data(jnp.asarray(v))
        else:
            out[k] = jax.tree.map(jnp.asarray, v)
    return out


def _device_copy(tree):
    """a device-side copy of a state (the original gets donated)"""
    import jax
    import jax.numpy as jnp

    def cp(x):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            return jax.random.wrap_key_data(jnp.copy(jax.random.key_data(x)))
        return jnp.copy(x)
    return jax.tree.map(cp, tree)


def write_checkpoint(path, st, extra):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(dict(state=_to_host(st), **extra), fh, protocol=5)
    os.replace(tmp, path)


def read_checkpoint(path):
    with open(path, 'rb') as fh:
        ck = pickle.load(fh)
    ck['state'] = _to_device(ck['state'])
    return ck


class _Stop:
    """SIGINT/SIGTERM, a STOP file in out, or the --maxh budget"""

    def __init__(self, out, t0, maxh):
        self.out, self.t0, self.maxh = out, t0, maxh
        self.reason = None
        stop = os.path.join(out, 'STOP')
        if os.path.exists(stop):
            os.remove(stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal)

    def _signal(self, num, _):
        if self.reason is not None:
            raise KeyboardInterrupt
        self.reason = signal.Signals(num).name

    def __call__(self):
        if self.reason is None:
            if os.path.exists(os.path.join(self.out, 'STOP')):
                self.reason = 'STOP file'
            elif self.maxh > 0 and time.time() - self.t0 >= \
                    0.99 * self.maxh * 3600:
                self.reason = f'--maxh {self.maxh}'
        return self.reason


# ---------------- the driver -------------------------------------------------
def run(smp, out, chains, frames, sweeps=10, kT=None, ladder=None,
        anneal=None, rescore_every=1, thin=10, block=10, seed=0, cpi=False,
        cpt=15., maxh=-1., settings=None, say=print):
    """Run `frames` frames (-1: until stopped). Returns a summary dict."""
    import jax
    import jax.numpy as jnp
    t_launch = time.time()
    os.makedirs(out, exist_ok=True)
    if ladder is not None and chains % len(ladder):
        raise ValueError('--chains must be a multiple of the ladder length')
    block = math.gcd(block, thin) if thin > 0 else block
    p_rec, p_dof = (os.path.join(out, x) for x in ('records.f64', 'dofs.f32'))
    p_cpt, p_meta = (os.path.join(out, x) for x in ('state.cpt', 'run.json'))
    cfg = dict(settings or {}, chains=chains, sweeps=sweeps, ladder=ladder,
               anneal=anneal, rescore_every=rescore_every, thin=thin,
               seed=seed)
    if cpi and os.path.exists(p_cpt):
        ck = read_checkpoint(p_cpt)
        old = {k: ck['cfg'].get(k) for k in cfg}
        if {k: v for k, v in old.items() if k != 'frames'} != \
                {k: v for k, v in cfg.items() if k != 'frames'}:
            raise ValueError(f'--cpi: settings differ from the checkpoint: '
                             f'{old} vs {cfg}')
        st, rung, f0 = ck['state'], jnp.asarray(ck['rung']), ck['frame']
        for p, n in ((p_rec, ck['rec_bytes']), (p_dof, ck['dof_bytes'])):
            with open(p, 'r+b') as fh:
                fh.truncate(n)
        say(f'continuing from frame {f0} ({p_cpt})')
    else:
        kTs = ladder if ladder is not None else kT
        kTs = None if kTs is None else np.resize(np.atleast_1d(kTs), chains)
        st = smp.init(chains, seed, kTs)
        rung = jnp.asarray(np.arange(chains) % len(ladder) if ladder
                           else np.zeros(chains), jnp.int32)
        f0 = 0
        for p in (p_rec, p_dof):
            open(p, 'wb').close()
    lay = dof_layout(jax.eval_shape(smp.dofs, st))
    meta = dict(cfg, dof_layout=lay, sampler=type(smp).__name__,
                record='frame, drift, energy (C), kT (C); float64',
                dofs='frame, then per chain the dof_layout arrays; float32')
    json.dump(meta, open(p_meta, 'w'), indent=1)

    stop = _Stop(out, t_launch, maxh)
    fns = {}

    def get_block(n):
        if n not in fns:
            fns[n] = make_block(smp, sweeps, n, rescore_every, ladder,
                                anneal, seed)
        return fns[n]
    writer = None
    t_cpt = time.time()
    f_end = None if frames < 0 else frames
    pending = None              # (records, dofs, frames, f_last) on device
    fh_rec, fh_dof = open(p_rec, 'ab'), open(p_dof, 'ab')
    t_start, steps0 = time.time(), f0

    def flush(p):
        rec, d, f_first, f_last = p
        drift, e, kTv = jax.device_get(rec)
        n = len(drift)
        r = np.concatenate([np.arange(f_first, f_first + n)[:, None],
                            drift[:, None], e, kTv], 1).astype(np.float64)
        fh_rec.write(r.tobytes())
        if thin > 0 and (f_last + 1) % thin == 0:
            fh_dof.write(_flat_dofs(f_last, jax.device_get(d)).tobytes())
        return float(np.nanmax(drift)) if np.isfinite(drift).any() else None

    last_drift, t_say = None, time.time()
    while (f_end is None or f0 < f_end) and not stop():
        n = block if f_end is None else min(block, f_end - f0)
        st, rung, rec, d = get_block(n)(st, rung, f0)
        if pending is not None:
            last_drift = flush(pending) or last_drift
        pending = (rec, d, f0, f0 + n - 1)
        f0 += n
        if time.time() - t_say > 60:
            t_say = time.time()
            say(f'frame {f0}; drift {last_drift}; '
                f'{(f0 - steps0) / (t_say - t_start):.3g} frames/s')
        if cpt > 0 and time.time() - t_cpt >= 60 * cpt:
            flush(pending)
            pending = None
            fh_rec.flush()
            fh_dof.flush()
            if writer is not None:
                writer.join()
            extra = dict(cfg=cfg, rung=np.asarray(rung), frame=f0,
                         rec_bytes=fh_rec.tell(), dof_bytes=fh_dof.tell())
            snap = _device_copy(st)
            writer = threading.Thread(target=write_checkpoint,
                                      args=(p_cpt, snap, extra))
            writer.start()
            t_cpt = time.time()
    if pending is not None:
        last_drift = flush(pending) or last_drift
    fh_rec.close()
    fh_dof.close()
    if writer is not None:
        writer.join()
    write_checkpoint(p_cpt, st, dict(
        cfg=cfg, rung=np.asarray(rung), frame=f0,
        rec_bytes=os.path.getsize(p_rec), dof_bytes=os.path.getsize(p_dof)))
    el = time.time() - t_start
    steps = (f0 - steps0) * sweeps * getattr(smp, 'steps_per_sweep', 1)
    summ = dict(frames=f0, stopped=stop.reason, seconds=round(el, 1),
                mc_steps_per_s=steps * chains / el if el > 0 else None,
                last_drift=last_drift)
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    say(f'{f0} frames; {summ["mc_steps_per_s"] or 0:.4g} MC steps/s '
        f'(incl. compile); stopped by {stop.reason or "end"}')
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('system')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chains', type=int, default=1024)
    ap.add_argument('--frames', type=int, default=100,
                    help='-1: until stopped')
    ap.add_argument('--sweeps', type=int, default=10)
    ap.add_argument('--kT', type=float, default=None)
    ap.add_argument('--ladder', nargs='?', const='protocol', default=None,
                    help="REX temperatures a,b,...; alone: the protocol's")
    ap.add_argument('--anneal', default=None, help='tmin,tmax,nmin,nmax')
    ap.add_argument('--rescore-every', type=int, default=1)
    ap.add_argument('--thin', type=int, default=10)
    ap.add_argument('--block', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sampler', default=None)
    ap.add_argument('--float64', action='store_true')
    ap.add_argument('--cpi', action='store_true')
    ap.add_argument('--cpt', type=float, default=15.)
    ap.add_argument('--maxh', type=float, default=-1.)
    ap.add_argument('--cov', default='frozen')
    ap.add_argument('--jax-cache', default=None,
                    help='compile cache (default OUT/jax_cache)')
    a = ap.parse_args()
    from . import pin_hash_seed, xla_defaults
    pin_hash_seed()
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', a.float64)
    # every session of a run uses the same executables: XLA doesn't always
    # compile a program the same way twice, and restarts should continue
    # exactly
    jax.config.update('jax_compilation_cache_dir',
                      a.jax_cache or os.path.join(a.out, 'jax_cache'))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    from . import sources, system
    from .check import _factory
    from .engine import make
    s = system.extract(sources.load(a.system), cov_mode=a.cov)
    smp = _factory(a.sampler)(s) if a.sampler else make(s)
    ladder = None
    if a.ladder == 'protocol':
        ladder = s.source.ladder
        if ladder is None:
            raise SystemExit(f'{a.system} has no protocol ladder')
    elif a.ladder:
        ladder = [float(x) for x in a.ladder.split(',')]
    anneal = None
    if a.anneal:
        t1, t2, n1, n2 = a.anneal.split(',')
        anneal = (float(t1), float(t2), int(n1), int(n2))
    run(smp, a.out, a.chains, a.frames, a.sweeps, a.kT, ladder, anneal,
        a.rescore_every, a.thin, a.block, a.seed, a.cpi, a.cpt, a.maxh,
        settings=dict(system=a.system, sampler=a.sampler, cov=a.cov,
                      float64=a.float64))


if __name__ == '__main__':
    main()
