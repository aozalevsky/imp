"""Coordinates of every particle in two RMF files, frame by frame (e.g.
chain_0.rmf3 written while sampling vs ladder_0.rmf3 made from traj.dof).

    python compare_rmf.py A.rmf3 B.rmf3
"""
import sys

import numpy as np
import IMP
import IMP.atom
import IMP.core
import IMP.rmf
import RMF


def frames(path):
    fh = RMF.open_rmf_file_read_only(path)
    m = IMP.Model()
    h = IMP.rmf.create_hierarchies(fh, m)
    ps = [p for x in h for p in IMP.atom.get_leaves(x)
          if IMP.core.XYZ.get_is_setup(p)]
    out, names = [], []
    for i in range(fh.get_number_of_frames()):
        IMP.rmf.load_frame(fh, RMF.FrameID(i))
        out.append(np.array([IMP.core.XYZ(p).get_coordinates() for p in ps]))
        names.append(fh.get_name(RMF.FrameID(i)))
    return np.array(out), names, len(ps)


A, na, pa = frames(sys.argv[1])
B, nb, pb = frames(sys.argv[2])
print(f'{sys.argv[1]}: {len(A)} frames x {pa} particles; '
      f'{sys.argv[2]}: {len(B)} frames x {pb} particles')
print('frame names', na[:3], '/', nb[:3])
n = min(len(A), len(B))
d = np.abs(A[:n] - B[:n])
print(f'max |coordinate difference| over {n} frames: {d.max():.3g} A '
      f'(identical: {np.array_equal(A[:n], B[:n])})')
