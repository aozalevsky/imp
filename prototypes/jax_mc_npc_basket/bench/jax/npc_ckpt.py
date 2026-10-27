"""GROMACS-style checkpoint/restart for run_npc_jax.py.

* state.cpt holds everything needed to continue the run exactly: every
  chain's sampler state (with its random key), the REX ladders and their
  random key, the stage and frame counter, and the byte offset of every
  output file at that frame. It is written atomically (temporary file,
  fsync, rename); the previous checkpoint is kept as state_prev.cpt.
* Output files are append-only. On restart they are checked (MD5 of the
  last MiB before the checkpointed offset, as mdrun -append does) and cut
  back to the checkpointed offset, so frames written after the checkpoint
  are dropped and then written again by the continued run.
* traj.dof is the compact trajectory: per saved frame and per ladder (the
  T_min replica, as for the RMF files) the score and the degrees of
  freedom (moved coordinates, rigid-body quaternions, moved nuisances),
  in fixed-size records after a JSON header. layout.npz next to it has
  what is needed to rebuild a full model state (the constant rows).
"""
import dataclasses
import hashlib
import json
import os
import signal
import time

import numpy as np

CPT, CPT_PREV = 'state.cpt', 'state_prev.cpt'
TAIL = 1 << 20          # bytes checksummed before each output offset
DOF_MAGIC = b'NPCDOF1\n'


# ---------------------------------------------------------------- checkpoint
def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(out, meta, arrays):
    """Write out/state.cpt atomically; the old one becomes state_prev.cpt."""
    path = os.path.join(out, CPT)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez(fh, __meta__=np.array(json.dumps(meta)), **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        os.replace(path, os.path.join(out, CPT_PREV))
    os.replace(tmp, path)
    _fsync_dir(out)


def load(out):
    """(meta, arrays) of the newest readable checkpoint in out, or None.
       Falls back to state_prev.cpt if state.cpt is missing (a crash
       between the two renames) or unreadable."""
    for name in (CPT, CPT_PREV):
        path = os.path.join(out, name)
        if not os.path.exists(path):
            continue
        try:
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads(str(arrays.pop('__meta__')))
            meta['_file'] = name
            return meta, arrays
        except Exception as e:                       # noqa: BLE001
            print(f'checkpoint {path} unreadable ({e}); trying the previous '
                  'one', flush=True)
    return None


def load_meta(out):
    """meta of the newest readable checkpoint in out (arrays not read)"""
    for name in (CPT, CPT_PREV):
        path = os.path.join(out, name)
        try:
            with np.load(path) as z:
                return json.loads(str(z['__meta__']))
        except Exception:                            # noqa: BLE001
            continue
    return None


def state_arrays(sb, prefix='state/'):
    """Host copies of a sampler state dataclass (batched or not); JAX
       random keys are stored as their raw key data."""
    import jax
    out = {}
    for f in dataclasses.fields(sb):
        v = getattr(sb, f.name)
        if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key):
            out[prefix + f.name + '@key'] = np.asarray(
                jax.random.key_data(v))
        else:
            out[prefix + f.name] = np.asarray(v)
    return out


def state_from_arrays(cls, arrays, prefix='state/'):
    import jax
    import jax.numpy as jnp
    kw = {}
    for f in dataclasses.fields(cls):
        if prefix + f.name + '@key' in arrays:
            kw[f.name] = jax.random.wrap_key_data(
                jnp.asarray(arrays[prefix + f.name + '@key']))
        else:
            kw[f.name] = jnp.asarray(arrays[prefix + f.name])
    return cls(**kw)


def key_arrays(name, key):
    import jax
    return {name + '@key': np.asarray(jax.random.key_data(key))}


def key_from_arrays(name, arrays):
    import jax
    import jax.numpy as jnp
    return jax.random.wrap_key_data(jnp.asarray(arrays[name + '@key']))


def file_md5(paths):
    h = hashlib.md5()
    for p in paths:
        with open(p, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


# ------------------------------------------------------ append-only outputs
def _tail_md5(path, offset):
    with open(path, 'rb') as fh:
        fh.seek(max(0, offset - TAIL))
        return hashlib.md5(fh.read(offset - max(0, offset - TAIL))) \
            .hexdigest()


class AppendFile:
    """Binary append-only output file whose position goes into the
       checkpoint. resume=(offset, md5) checks the file and cuts it back to
       offset; resume=None starts it empty."""

    def __init__(self, path, resume=None):
        self.path = path
        if resume is None:
            self.fh = open(path, 'wb')
            return
        offset, md5 = resume
        size = os.path.getsize(path) if os.path.exists(path) else -1
        if size < offset:
            raise RuntimeError(f'{path}: {size} bytes, the checkpoint needs '
                               f'{offset}; the output was lost or replaced')
        if _tail_md5(path, offset) != md5:
            raise RuntimeError(f'{path}: content before the checkpointed '
                               'offset differs from the checkpoint')
        with open(path, 'r+b') as fh:
            fh.truncate(offset)
        self.fh = open(path, 'ab')

    def write(self, b):
        self.fh.write(b)

    def mark(self):
        """(offset, md5 of the tail) after flushing to disk"""
        self.fh.flush()
        os.fsync(self.fh.fileno())
        off = self.fh.tell()
        return [off, _tail_md5(self.path, off)]

    def close(self):
        self.fh.close()


class TextLog(AppendFile):
    def write(self, s):
        self.fh.write(s.encode())


# ------------------------------------------------------ compact trajectory
def dof_layout(smp, states, nuis_rows):
    """Rows stored per frame, from the sampler's mover tables, plus the
       constant rows of the first chain. Rows that are not moved but differ
       between chains (e.g. placed by a per-chain shuffle without a mover)
       are stored per frame as well. states: batched initial state."""
    xyz = np.asarray(states.xyz)                     # (N, K+2, 3)
    nuis = np.asarray(states.nuis)
    free = set(int(r) for r in smp.dof_rows)
    member = set(int(r) for r in smp.member_rows)
    rest = np.array(sorted(set(range(smp.K)) - free - member), np.int64)
    if len(rest):
        diff = np.any(xyz[:, rest] != xyz[:1, rest], axis=(0, 2))
        free |= set(int(r) for r in rest[diff])
    nrows = set(int(r) for r in nuis_rows)
    nrest = np.array(sorted(set(range(smp.K)) - nrows), np.int64)
    if len(nrest):
        d = np.any((nuis[:, nrest] != nuis[:1, nrest])
                   & ~(np.isnan(nuis[:, nrest]) & np.isnan(nuis[:1, nrest])),
                   axis=0)
        nrows |= set(int(r) for r in nrest[d])
    return dict(free_rows=np.array(sorted(free), np.int64),
                nuis_rows=np.array(sorted(nrows), np.int64),
                member_rows=np.array(sorted(member), np.int64),
                nb=int(smp.nb), K=int(smp.K),
                xyz0=xyz[0], nuis0=nuis[0], quat0=np.asarray(states.quat[0]))


def dof_dtype(L, nb, nf, nn):
    return np.dtype([('frame', '<i8'), ('chain', '<i4', (L,)),
                     ('score', '<f8', (L,)), ('quat', '<f4', (L, nb, 4)),
                     ('xyz', '<f4', (L, nf, 3)), ('nuis', '<f4', (L, nn))])


def dof_header(layout, L, stage, extra=None):
    h = dict(format='npc_dof', version=1, stage=stage, ladders=L,
             nb=layout['nb'], K=layout['K'],
             n_free=len(layout['free_rows']),
             n_nuis=len(layout['nuis_rows']))
    h.update(extra or {})
    js = json.dumps(h).encode()
    n = len(DOF_MAGIC) + 8 + len(js)
    pad = (-n) % 4096
    return DOF_MAGIC + np.uint64(len(js) + pad).tobytes() + js + b' ' * pad


def dof_record(dt, frame, chains, scores, quat, xyz, nuis):
    r = np.zeros((), dt)
    r['frame'], r['chain'], r['score'] = frame, chains, scores
    r['quat'], r['xyz'], r['nuis'] = quat, xyz, nuis
    return r.tobytes()


def read_dof(path):
    """(header, records): records is a read-only memmap with fields frame,
       chain, score, quat, xyz, nuis (one record per saved frame). A
       partial last record (a run killed while writing) is ignored."""
    with open(path, 'rb') as fh:
        if fh.read(len(DOF_MAGIC)) != DOF_MAGIC:
            raise ValueError(f'{path}: not a traj.dof file')
        n = int(np.frombuffer(fh.read(8), np.uint64)[0])
        h = json.loads(fh.read(n).decode())
    off = len(DOF_MAGIC) + 8 + n
    dt = dof_dtype(h['ladders'], h['nb'], h['n_free'], h['n_nuis'])
    nrec = (os.path.getsize(path) - off) // dt.itemsize
    if nrec == 0:
        return h, np.zeros(0, dt)
    return h, np.memmap(path, dt, 'r', offset=off, shape=(nrec,))


def full_state_arrays(layout, rec, i):
    """xyz (K+2, 3), nuis (K+2,), quat (nb+1, 4) of ladder slot i of one
       record. Rigid-body member rows keep the layout's (first chain's
       initial) values: write_model() with update=True recomputes them
       from the bodies."""
    xyz = layout['xyz0'].copy()
    nuis = layout['nuis0'].copy()
    quat = layout['quat0'].copy()
    xyz[layout['free_rows']] = rec['xyz'][i]
    nuis[layout['nuis_rows']] = rec['nuis'][i]
    quat[:layout['nb']] = rec['quat'][i]
    return xyz, nuis, quat


# ------------------------------------------------------------- stopping
class StopRequest:
    """First SIGTERM / SIGINT / SIGUSR1 / SIGUSR2: finish the current
       frame, write a checkpoint and stop. A second SIGINT stops at once.
       Also a STOP file in out (made after the start) and the --maxh budget
       (stop at 99% of it, as mdrun does)."""

    def __init__(self, out, t_launch, maxh=-1.):
        self.reason = None
        self.stop_file = os.path.join(out, 'STOP')
        self.t_launch, self.maxh = t_launch, maxh
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1,
                  signal.SIGUSR2):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        name = signal.Signals(signum).name
        if self.reason is not None and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.reason = self.reason or f'signal {name}'

    def check(self):
        if self.reason is None:
            if (os.path.exists(self.stop_file)
                    and os.path.getmtime(self.stop_file) >= self.t_launch):
                self.reason = 'STOP file'
            elif (self.maxh > 0 and time.time() - self.t_launch
                    >= 0.99 * self.maxh * 3600):
                self.reason = f'--maxh {self.maxh}'
        return self.reason
