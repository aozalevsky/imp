"""Recipe R6: Sweep-wide precompute of bead proposals' static terms.

Specialization level:
  Level 3 (Sweep-level precompute helper `precompute_bead_static_ev`, used in
  `few_bodies_beads`).

Preconditions (asserted):
  1. Each bead is moved at most once per sweep by a `BallMover`
     (`len(bead_steps) == s.nbead`).
  2. Bead movers visit the beads in contiguous row order
     (`bead_rows == arange(bead_off, bead_off + s.nbead)`), and no earlier
     mover in the sweep moves any bead row.
  3. The static particle set is non-empty (`len(s.static_rows) > 0`) and never
     moves during the sweep.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .r1_static_ev_grid import grid_ev


def check_preconditions(s, bead_steps, bead_rows, bead_off):
    """Assert R6 preconditions on System s and its bead mover layout."""
    assert len(s.static_rows) > 0, (
        'R6 requires a non-empty static particle set')
    assert len(bead_steps) == s.nbead and s.nbead > 0, (
        f'R6 requires one BallMover per bead ({s.nbead}), got '
        f'{len(bead_steps)}')
    assert np.array_equal(
        bead_rows, np.arange(bead_off, bead_off + s.nbead)), (
        'R6 requires BallMovers in contiguous bead row order')
    # Verify no non-ball mover touches any bead row
    for i, mv in enumerate(s.movers):
        if mv.kind != 'ball':
            assert not len(mv.beads), (
                f'R6 requires beads to be moved only by BallMovers, but '
                f'mover {i} ({mv.kind}) moves beads {mv.beads}')


def precompute_bead_static_ev(X0, bead_off, nbead, bead_steps, bead_radii,
                              r_beads, nrm, uni, grid):
    """Evaluate the static-grid EV of all `nbead` proposed bead positions in a
       single batched `(nbead, C, K)` pass before the mover loop.

       X0:          (3, n_dyn, C) positions at the start of the sweep
       bead_off:    start row of beads in X0
       nbead:       number of beads
       bead_steps:  (nbead,) mover indices of the BallMovers
       bead_radii:  (nbead,) BallMover step radii
       r_beads:     (nbead,) EV sphere radii of the beads
       nrm:         (n_movers, >=3, C) normal draws Z[:, :3]
       uni:         (n_movers, >=1, C) uniform draws with U[:, 0] at index 0
       grid:        R1 static EV grid dict
    """
    dt = X0.dtype
    xo_all = X0[:, bead_off:bead_off + nbead]
    db = jnp.moveaxis(nrm[bead_steps, :3], 1, 0)
    ub = uni[bead_steps, 0]
    norm_db = jnp.sqrt(
        db[0] * db[0] + db[1] * db[1] + db[2] * db[2])
    step_b = jnp.asarray(bead_radii, dt)[:, None] * jnp.cbrt(ub)
    xn_all = xo_all + (db / norm_db[None]) * step_b[None]
    return jax.lax.optimization_barrier(grid_ev(grid, xn_all, r_beads, dt))


def factory(s):
    """Return FewBodiesBeads with R6 enabled (and R3 unblocked so R6 is
       tested directly against the reference Engine)."""
    from templates.few_bodies_beads.sampler import FewBodiesBeads
    return FewBodiesBeads(s, blocked=False, precompute=True)
