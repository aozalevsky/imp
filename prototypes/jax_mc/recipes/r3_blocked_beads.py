"""Recipe R3: Blocked bead steps (score B consecutive beads in one reduction,
then apply exact within-block corrections).

Specialization level:
  Level 3 (Blocked bead execution segments and block scorers in
  `few_bodies_beads`; GPU-only by default since CPU prefers unblocked 1-bead
  steps).

Preconditions (asserted):
  1. Consecutive `ball` movers in a block move consecutive bead rows
     `[o, o + B)` in `X`.
  2. No EM Gaussians lie on bead rows.
  3. Every term reading bead rows has an exact within-block `m2 < m`
     correction (`SoftSphereEV`, `Pairs`, unambiguous `CrossLinks` with
     `cmax == 1`).
"""
import numpy as np

from jax_mc.system import BEAD
from jax_mc.terms import basic, em


def check_preconditions(s, steps, bead_off, n_de):
    """Assert R3 preconditions on System s and its step table."""
    bead_sys_rows = np.nonzero(s.row_kind == BEAD)[0]
    assert len(bead_sys_rows) == s.nbead and s.nbead > 0, (
        'R3 requires bead rows moved by BallMovers')
    assert bead_off + s.nbead == n_de, (
        f'R3 requires bead rows [{bead_off}, {bead_off + s.nbead}) to end '
        f'the dynamic EV slice [0, {n_de})')
    for t in s.terms:
        if isinstance(t, em._EM):
            assert not np.isin(t.rows_read, bead_sys_rows).any(), (
                f'R3 requires no EM Gaussians on beads, but {t.label} reads '
                f'bead rows')
        elif t.rows_read is not None and np.isin(
                t.rows_read, bead_sys_rows).any():
            assert isinstance(
                t, (basic.SoftSphereEV, basic.Pairs, basic.CrossLinks)), (
                f'R3 requires every term reading beads to have a within-block '
                f'correction, got {t.label} ({type(t).__name__})')
            if isinstance(t, basic.CrossLinks):
                assert t.con.shape[1] == 1, (
                    f'R3 within-block cross-link correction requires '
                    f'unambiguous cross-links (cmax == 1), got '
                    f'{t.con.shape[1]}')
    bead_steps = [p for p in steps if p['kind'] == 'bead']
    assert len(bead_steps) == s.nbead
    assert np.array_equal(
        [p['row'] for p in bead_steps],
        np.arange(bead_off, bead_off + s.nbead)), (
        'R3 requires BallMovers in contiguous row order')


def build_exec_segs(steps, n_de, max_blk):
    """Group consecutive bead steps with consecutive rows into blocks of up to
       `max_blk` beads, with `mask_ext[m, o:o+m+1] = False` so within-block
       bead pairs `m2 < m` are excluded from the external reduction and
       scored against the updated positions `(xcx[m2], xcy[m2], xcz[m2])`."""
    assert max_blk >= 1
    segs = []
    cur = []
    for i, p in enumerate(steps):
        if p['kind'] == 'bead':
            if cur and (p['row'] != steps[cur[-1]]['row'] + 1
                        or len(cur) >= max_blk):
                segs.append(('bead_blk', cur))
                cur = [i]
            else:
                cur.append(i)
        else:
            if cur:
                segs.append(('bead_blk', cur))
                cur = []
            segs.append((p['kind'], i))
    if cur:
        segs.append(('bead_blk', cur))
    exec_segs = []
    for kind, val in segs:
        if kind != 'bead_blk':
            exec_segs.append((kind, val, None))
        else:
            idxs = val
            B = len(idxs)
            o = steps[idxs[0]]['row']
            mask_ext = np.zeros((B, n_de), dtype=bool)
            for m, i in enumerate(idxs):
                mask_ext[m] = steps[i]['evmask_dyn'].copy()
                mask_ext[m, o:o + m + 1] = False
            exec_segs.append(('bead_blk', idxs, mask_ext))
    return exec_segs


def factory(s):
    """Return FewBodiesBeads with R3 blocked beads enabled (`blocked=True`)."""
    from templates.few_bodies_beads.sampler import FewBodiesBeads
    return FewBodiesBeads(s, blocked=True)
