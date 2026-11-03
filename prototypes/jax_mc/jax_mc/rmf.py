"""RMF files from a jax_mc.run output.

  python -m jax_mc.rmf OUT [--chains 0,1,2] [--rung R] [--best N]
      [--every K]

Rebuilds the system with the run's settings (OUT/run.json), then writes
into OUT/rmf/:
  chain<c>.rmf3   one frame per snapshot of chain c (--chains);
  rung<r>.rmf3    one frame per snapshot, from whichever chain was at
                  ladder rung r then (--rung, replica-exchange runs);
  best.rmf3       the N lowest-energy snapshots over all chains (--best).
Frames are named 'frame F chain C E ...' with the energy recorded for that
frame. --every K keeps every K-th snapshot.
"""
import argparse
import json
import os

import numpy as np

from . import run as R


def snapshot_dofs(d, k):
    """chain-last degrees of freedom (System layout) of snapshot k, from
       run.snapshots()'s chain-first arrays"""
    out = {}
    for name, a in d.items():
        a = np.asarray(a[k], np.float64)                    # (C, ...)
        out[name] = np.moveaxis(a, 0, -1) if a.ndim == 2 else \
            np.transpose(a, (2, 1, 0))                      # (3|4, n, C)
    return out


def write(out, chains=(), rung=None, best=0, every=1, say=print):
    import IMP
    import IMP.rmf
    import RMF
    from . import sources, system
    meta = json.load(open(os.path.join(out, 'run.json')))
    s = system.extract(sources.load(meta['system']),
                       cov_mode=meta.get('cov', 'frozen'))
    if s.source.hier is None:
        raise SystemExit(f'{meta["system"]}: the source has no hierarchy')
    frames, d = R.snapshots(out)
    fr, _, E, kT = R.records(out)
    at = np.searchsorted(fr, frames)                        # record rows
    keep = np.arange(0, len(frames), every)
    os.makedirs(os.path.join(out, 'rmf'), exist_ok=True)
    cache = {}

    def dofs(k):
        if k not in cache:
            cache.clear()
            cache[k] = snapshot_dofs(d, k)
        return cache[k]

    def dump(name, items):
        """items: (snapshot k, chain c) in order"""
        path = os.path.join(out, 'rmf', name)
        fh = RMF.create_rmf_file(path)
        IMP.rmf.add_hierarchy(fh, s.source.hier)
        for k, c in items:
            s.set_imp_dofs(dofs(k), c)
            IMP.rmf.save_frame(fh, f'frame {frames[k]} chain {c} '
                               f'E {E[at[k], c]:.6g}')
        del fh
        say(f'{path}: {len(items)} frames')

    for c in chains:
        dump(f'chain{c}.rmf3', [(k, c) for k in keep])
    if rung is not None:
        lad = meta.get('ladder')
        if not lad:
            raise SystemExit('--rung needs a replica-exchange run')
        t = lad[rung]
        # nearest: float32 runs store the ladder rounded
        dump(f'rung{rung}.rmf3', [(k, int(np.argmin(np.abs(kT[at[k]] - t))))
                                  for k in keep])
    if best:
        flat = [(E[at[k], c], k, c) for k in keep for c in range(E.shape[1])]
        flat.sort()
        dump('best.rmf3', sorted((k, c) for _, k, c in flat[:best]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--chains', default='')
    ap.add_argument('--rung', type=int, default=None)
    ap.add_argument('--best', type=int, default=0)
    ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args()
    from . import pin_hash_seed
    pin_hash_seed()
    write(a.out, [int(x) for x in a.chains.split(',') if x], a.rung, a.best,
          a.every)


if __name__ == '__main__':
    main()
