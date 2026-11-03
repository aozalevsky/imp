"""Recipe R7: Contract-preserving sliced RNG draws.

Specialization level:
  Level 2 / Level 3 (RNG consumption helper `sliced_draws` and
  `SlicedRNGEngine(Engine)`).

Preconditions (asserted):
  1. All movers in `s.movers` are in `{'ball', 'rb', 'transform', 'normal'}`.
  2. Follows the `jax_mc` random-number contract (`DESIGN.md`): generates the
     full `(n_movers, 3)` uniform and `(n_movers, 6)` normal arrays per chain
     and sweep from `fold_in(chain_key[c], sweep)`, then slices out the unused
     rotation columns before transposing to chain-last `(..., C)`.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jax_mc import engine as E


def check_preconditions(s):
    """Assert R7 preconditions on System s."""
    kinds = {mv.kind for mv in s.movers}
    allowed = {'ball', 'rb', 'transform', 'normal'}
    assert kinds <= allowed, (
        f'R7 requires mover kinds in {allowed}, got {kinds - allowed}')


def sliced_draws(key, sweep, n_movers, rot_movers, dtype):
    """Generate bitwise-identical draws to `engine.draws(key, sweep, n_movers,
       dtype)`, but slice out unused rotation columns inside `vmap` before the
       chain-last `moveaxis` transpose.

       Returns:
         nrm3:   (n_movers, 3, C)  Z[:, 0:3] (ball/translation/nuisance)
         uni02:  (n_movers, 2, C)  stack([U[:, 0], U[:, 2]], axis=1)
                                   (col 0: ball cbrt, col 1: Metropolis accept)
         rb_ax:  (n_rot, 3, C)     Z[rot_movers, 3:6] (rotation axis)
         rb_u1:  (n_rot, C)        U[rot_movers, 1] (rotation angle)
    """
    rot_idx = np.asarray(rot_movers, np.int32)

    def one(k):
        k = jax.random.fold_in(k, sweep)
        u = jax.random.uniform(k, (n_movers, 3), dtype)
        z = jax.random.normal(jax.random.fold_in(k, 1), (n_movers, 6), dtype)
        u02 = jnp.stack([u[:, 0], u[:, 2]], axis=1)
        return z[:, :3], u02, z[rot_idx, 3:6], u[rot_idx, 1]

    nrm3, uni02, rb_ax, rb_u1 = jax.vmap(one)(key)
    return (jnp.moveaxis(nrm3, 0, -1),
            jnp.moveaxis(uni02, 0, -1),
            jnp.moveaxis(rb_ax, 0, -1),
            jnp.moveaxis(rb_u1, 0, -1))


class SlicedRNGEngine(E.Engine):
    """Engine subclass demonstrating R7: slices unused rotation draws inside
       `vmap` before transposing to chain-last, then reconstructs the exact
       per-step `(u, z)` views."""

    def __init__(self, s, incremental=True):
        check_preconditions(s)
        super().__init__(s, incremental=incremental)
        self.rot_movers = np.array(
            [i for i, mv in enumerate(s.movers)
             if mv.kind in ('rb', 'transform')], np.int32)
        self.rot_map = np.full(len(s.movers), -1, np.int32)
        for r_i, m_i in enumerate(self.rot_movers):
            self.rot_map[m_i] = r_i

    def draws(self, key, sweep, dtype):
        n = len(self.s.movers)
        nrm3, uni02, rb_ax, rb_u1 = sliced_draws(
            key, sweep, n, self.rot_movers, dtype)
        C = key.shape[0]
        z_pad = jnp.zeros((n, 3, C), dtype)
        u1_pad = jnp.zeros((n, 1, C), dtype)
        if len(self.rot_movers):
            z_pad = z_pad.at[self.rot_movers].set(rb_ax)
            u1_pad = u1_pad.at[self.rot_movers, 0].set(rb_u1)
        U = jnp.concatenate([uni02[:, 0:1], u1_pad, uni02[:, 1:2]], axis=1)
        Z = jnp.concatenate([nrm3, z_pad], axis=1)
        return U, Z


def factory(s):
    return SlicedRNGEngine(s)
