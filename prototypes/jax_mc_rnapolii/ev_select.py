"""Choose how a run scores excluded volume (EV) against the moving rigid
bodies:

* 'dense': each moved row against every other dynamic EV row;
* 'body': per-body grids (TutSamplerSoA(body_grid=True)), each body's EV
  rows in a grid fixed in the body's own frame.

Both are exact; which is faster depends on the body sizes and the chain
count. select() decides:

* forced: --ev dense or --ev body;
* heuristic: if the largest rigid body has fewer than MIN_BODY_ROWS EV
  rows, dense without a trial (the tutorial's 18-19-row bodies run 25%
  slower with grids);
* cached: an earlier trial for the same system, chain count, block, GPU
  model, JAX version and sampler code;
* trial: the production block is compiled for both and timed at the run's
  chain count on copies of the starting state. The work of every kernel is
  fixed by the shapes, not by the configuration, so the starting state is
  representative. 'body' must be at least MARGIN faster. With several
  ranks under Open MPI, every rank times both and they agree on the summed
  times through files in a shared directory.

The compiled block of the chosen mode is returned, so the trial costs one
extra compilation plus a few seconds of timing.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

MIN_BODY_ROWS = 30
MARGIN = 0.02


def system_key(smp):
    """Hash of the tables that decide the EV work"""
    h = hashlib.sha1()
    for x in (np.asarray(smp.body_ev), smp.r, smp.local, smp.static_xyz,
              np.asarray(smp.g_table.shape), np.asarray([smp.n_movers])):
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()[:16]


def code_key():
    """Hash of the sampler sources and the running script"""
    h = hashlib.sha1()
    for name in ('sampler', 'sampler_soa', 'ev_select', '__main__'):
        f = getattr(sys.modules.get(name), '__file__', None)
        if f and os.path.exists(f):
            with open(f, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()[:8]


def heuristic(smp):
    rows = max(n for _, n in smp.body_ev)
    if rows < MIN_BODY_ROWS:
        return 'dense', (f'heuristic: largest rigid body has {rows} EV rows '
                         f'(< {MIN_BODY_ROWS})')
    return None, f'largest rigid body has {rows} EV rows'


def clone(st):
    """A copy of a state (blocks donate their input)"""
    import jax
    import jax.numpy as jnp

    def c(x):
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return jax.random.wrap_key_data(
                jnp.copy(jax.random.key_data(x)),
                impl=jax.random.key_impl(x))
        return jnp.copy(x)
    return jax.tree.map(c, st)


def time_block(block, st, kTs, min_s=1.0, min_reps=2):
    """Seconds per call of a compiled block after a warm-up call (st is
       donated)"""
    import jax
    st, _ = block(st, kTs)
    jax.block_until_ready(st.energy)
    n, t0 = 0, time.perf_counter()
    while n < min_reps or time.perf_counter() - t0 < min_s:
        st, _ = block(st, kTs)
        jax.block_until_ready(st.energy)
        n += 1
    return (time.perf_counter() - t0) / n


def _read_json(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _agree(share, rank, size, times, timeout=300.):
    """Sum every rank's trial times (files in the directory share, tagged
       with the MPI job so stale files are ignored); None on timeout or
       without an Open MPI job tag (each rank then decides on its own
       times; both schemes are exact)"""
    job = (os.environ.get('PMIX_NAMESPACE')
           or os.environ.get('OMPI_MCA_ess_base_jobid', ''))
    if not job:
        return None
    _write_json(os.path.join(share, f'rank{rank}.json'),
                dict(job=job, times=times))
    t0 = time.time()
    while time.time() - t0 < timeout:
        got = [_read_json(os.path.join(share, f'rank{r}.json'), None)
               for r in range(size)]
        if all(g is not None and g['job'] == job for g in got):
            return {m: sum(g['times'][m] for g in got) for m in times}
        time.sleep(0.5)
    return None


def select(smp, st, kTs, compile_block, mode='auto', cache=None, rank=0,
           size=1, share=None, say=print):
    """smp: a TutSamplerSoA; st: the run's starting state (not modified);
       kTs: a block's temperatures; compile_block(sampler) -> compiled
       block. Returns (sampler variant, compiled block, info dict)."""
    import jax
    B = int(st.energy.shape[0])
    dev = jax.devices()[0]
    key = (f'{system_key(smp)}|chains {B}|block {len(kTs)}|'
           f'{dev.device_kind}|jax {jax.__version__}|code {code_key()}')
    info = dict(requested=mode, key=key)
    if mode in ('dense', 'body'):
        choice, info['reason'] = mode, 'forced'
    else:
        choice, info['reason'] = heuristic(smp)
        if choice is None:
            rec = _read_json(cache, {}).get(key) if cache else None
            if rec is not None:
                choice = rec['mode']
                info.update(reason='cached trial', trial_s=rec['trial_s'])
    if choice is not None:
        v = smp.variant(choice == 'body')
        t0 = time.time()
        block = compile_block(v)
        info.update(mode=choice, compile_s={choice: time.time() - t0})
        say(f'EV scheme: {choice} ({info["reason"]})')
        return v, block, info

    # timed trial
    blocks, comp, times = {}, {}, {}
    for m in ('dense', 'body'):
        v = smp.variant(m == 'body')
        t0 = time.time()
        blocks[m] = (v, compile_block(v))
        comp[m] = time.time() - t0
        times[m] = time_block(blocks[m][1], clone(st), kTs)
    agreed = _agree(share, rank, size, times) if size > 1 and share \
        else None
    tot = agreed or times
    choice = 'body' if tot['body'] < (1. - MARGIN) * tot['dense'] \
        else 'dense'
    info.update(mode=choice, compile_s=comp, trial_s=times,
                reason='trial' + (f' (summed over {size} ranks)' if agreed
                                  else ''))
    if cache and rank == 0:
        c = _read_json(cache, {})
        c[key] = dict(mode=choice, trial_s=tot, date=time.strftime('%F %T'))
        _write_json(cache, c)
    sweeps = 10 * len(kTs)
    say(f'EV scheme: {choice} ({info["reason"]}: dense '
        f'{1e3 * times["dense"] / sweeps:.3f} ms, body grids '
        f'{1e3 * times["body"] / sweeps:.3f} ms per sweep incl. rescores on '
        f'this rank; compile {comp["dense"]:.0f} + {comp["body"]:.0f} s)')
    v, block = blocks[choice]
    del blocks
    return v, block, info
