"""Collect the best-scoring models of run_npc_jax.py runs into one RMF file.

For every run (chain) the saved frame with the lowest score is taken; the
best --n runs go into one RMF file, best first, each frame named
<run dir>_chain<i>_frame<f>_score<s>. Scores are the sampler's totals from
trace.jsonl. --csv lists every saved frame of every run.

  python best_models.py runs/poc_g0 runs/poc_g1 --n 100 --out best_models.rmf3

--trajectory FILE instead copies every frame of one chain file, all data
included. Frames are copied at the RMF level, values unchanged: positions,
rigid-body frames, Gaussian variances, nuisances (psi, sigma) and
restraint scores. The RMF files of different runs differ in the
excluded-volume pair features (IMP adds a node for a pair the first time
it scores), so when merging runs, nodes are matched by their path of
names and the merged file keeps the excluded-volume total but not its
pair features. All files must come from the same script and
options. Structure provenance paths are rewritten to NPC_Basket/... next
to the output file.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
import RMF

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='*')
ap.add_argument('--stage', default='rex2', help='rex1, rex2 or one_stage')
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--out', required=True)
ap.add_argument('--csv', default=None)
ap.add_argument('--trajectory', default=None)
a = ap.parse_args()


# static file names that RMF stores relative to the RMF file
PATH_KEYS = [('provenance', 'structure filename'),
             ('provenance', 'script filename'),
             ('IMP restraint files', 'filename')]


def path_keys(fh):
    cats = {fh.get_name(c): c for c in fh.get_categories()}
    return {(c, k): fh.get_key(cats[c], k, RMF.StringTag())
            for c, k in PATH_KEYS if c in cats}


def clean_provenance(src, out, src_path):
    """Point provenance and restraint data files at NPC_Basket/... beside
       the output file. RMF stores these paths relative to the file."""
    sdir = os.path.dirname(os.path.abspath(src_path))
    kin, kout = path_keys(src), path_keys(out)
    kept = 0
    for nid in src.get_node_ids():
        for name, k in kin.items():
            rel = src.get_node(nid).get_static_value(k)
            if rel is None:
                continue
            fn = os.path.normpath(os.path.join(sdir, rel))
            i = fn.find('NPC_Basket' + os.sep)
            if i >= 0:
                out.get_node(nid).set_static_value(kout[name], fn[i:])
            else:
                kept += 1
    if kept:
        print(f'{kept} provenance paths without NPC_Basket/ left as they are')


# the values RMF files from run_npc_jax.py hold per frame
FRAME_KEYS = [('physics', 'coordinates', RMF.Vector3Tag()),
              ('physics', 'translation', RMF.Vector3Tag()),
              ('physics', 'rotation', RMF.Vector4Tag()),
              ('physics', 'variances', RMF.Vector3Tag()),
              ('uncertainty', 'scale', RMF.FloatTag()),
              ('feature', 'score', RMF.FloatTag())]


def node_paths(fh):
    """{path: node id}, a path being ((name, n-th child of that name), ...)
       from the root; below excluded-volume restraints only the top node."""
    out = {}

    def walk(nd, path):
        seen = collections.Counter()
        for c in nd.get_children():
            nm = c.get_name()
            q = path + ((nm, seen[nm]),)
            seen[nm] += 1
            if q not in out:
                out[q] = c.get_id()
                if not (len(q) == 1 and c.get_type() == RMF.FEATURE
                        and nm.startswith('ExcludedVolume')):
                    walk(c, q)
    walk(fh.get_root_node(), ())
    return out


def copy_frames(selection):
    """selection: (RMF file, frame index, output frame name), in order."""
    first = RMF.open_rmf_file_read_only(selection[0][0])
    out = RMF.create_rmf_file(a.out)
    RMF.clone_file_info(first, out)
    RMF.clone_hierarchy(first, out)
    RMF.clone_static_frame(first, out)
    clean_provenance(first, out, selection[0][0])
    out_paths = node_paths(out)
    kout = [out.get_key(out.get_category(c), k, tag)
            for c, k, tag in FRAME_KEYS]
    single = len(set(p for p, _, _ in selection)) == 1
    path = src = pairs = None
    dropped = 0
    for p, k, name in selection:
        if single:                  # one file: same nodes, copy everything
            if src is None:
                src = RMF.open_rmf_file_read_only(p)
            src.set_current_frame(RMF.FrameID(k))
            out.add_frame(name, RMF.FRAME)
            RMF.clone_loaded_frame(src, out)
            continue
        if p != path:
            path, src = p, RMF.open_rmf_file_read_only(p)
            cats = {src.get_name(c): c for c in src.get_categories()}
            kin = [src.get_key(cats[c], kk, tag) if c in cats else None
                   for c, kk, tag in FRAME_KEYS]
            keys = [(ki, ko) for ki, ko in zip(kin, kout) if ki is not None]
            src_paths = node_paths(src)
            pairs = [(src.get_node(i), out.get_node(out_paths[q]))
                     for q, i in src_paths.items() if q in out_paths]
            missing = [src.get_node(i) for q, i in src_paths.items()
                       if q not in out_paths]
        src.set_current_frame(RMF.FrameID(k))
        out.add_frame(name, RMF.FRAME)
        for sn, on in pairs:
            for ki, ko in keys:
                v = sn.get_frame_value(ki)
                if v is not None:
                    on.set_frame_value(ko, v)
        dropped += sum(1 for sn in missing for ki, _ in keys
                       if sn.get_frame_value(ki) is not None)
    del out
    print(f'wrote {len(selection)} frames to {a.out}'
          + (f'; {dropped} values of nodes missing from the first file '
             'dropped' if dropped else ''))


if a.trajectory:
    src = RMF.open_rmf_file_read_only(a.trajectory)
    copy_frames([(a.trajectory, f.get_index(), src.get_name(f))
                 for f in src.get_frames()])
    raise SystemExit

rows = []           # (run, chain, frame, score, file, frame index)
for r in a.runs:
    sdir = os.path.join(r, a.stage)
    label = os.path.basename(os.path.normpath(r))
    with open(os.path.join(sdir, 'trace.jsonl')) as fh:
        trace = [json.loads(x) for x in fh]
    scores = np.array([t['score'] for t in trace])
    tmin = np.array([t['tmin_chain'] for t in trace])
    fh = RMF.open_rmf_file_read_only(os.path.join(sdir, 'chain_0.rmf3'))
    saved = [int(fh.get_name(f)) for f in fh.get_frames()]
    del fh
    for i in range(tmin.shape[1]):
        for k, f in enumerate(saved):
            if f < len(trace):
                rows.append((label, i, f, float(scores[f, tmin[f, i]]),
                             os.path.join(sdir, f'chain_{i}.rmf3'), k))
    print(f'{label}: {tmin.shape[1]} runs, {len(trace)} frames, '
          f'{len(saved)} saved per run')

if a.csv:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'chain', 'frame', 'score'])
        for row in rows:
            w.writerow([row[0], row[1], row[2], f'{row[3]:.2f}'])

best = {}
for row in rows:
    key = (row[0], row[1])
    if key not in best or row[3] < best[key][3]:
        best[key] = row
sel = sorted(best.values(), key=lambda x: x[3])[:a.n]
print(f'{len(best)} runs; best {sel[0][3]:.1f}, '
      f'{len(sel)}th best {sel[-1][3]:.1f}')
copy_frames([(path, k, f'{label}_chain{i}_frame{f}_score{s:.1f}')
             for label, i, f, s, path, k in sel])
