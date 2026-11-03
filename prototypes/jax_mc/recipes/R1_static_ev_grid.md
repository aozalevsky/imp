# R1: Static/movable split and static EV spatial hash grid

## What it does
Splits excluded-volume (EV) particles into a moving set (`n_de`) and a static
set (`n_se`), precomputes a 3D uniform spatial hash grid over the static EV
spheres on the host (`build_grid`), and scores moving query spheres against
at most `K` candidate static spheres in their cell (`grid_ev`) instead of
scanning all `n_se` static spheres on every MC step.

- **Cell table (`table_u16`)**: `(n_cells + 1, K)` `uint16` indices into the
  static EV set, padded with a far-away dummy sphere at `(1e4, 1e4, 1e4)` with
  radius `0.0` (the last cell row `n_cells` is the out-of-bounds sentinel).
- **Point table (`PR`)**: `(n_se + 1, 4)` contiguous `(x, y, z, r)` array so
  each candidate gather loads one 16-byte `float4` per candidate.
- **Specialization level**:
  - **Level 1 (Term subclass)**: [`StaticGridEV(SoftSphereEV)`](r1_static_ev_grid.py)
    replaces `SoftSphereEV` in `Engine(s)` via `replace_ev(s)`, which rebuilds the system with `extract(source, replace={SoftSphereEV: StaticGridEV})`.
  - **Level 3 (Template)**: [`FewBodiesBeads`](../templates/few_bodies_beads/sampler.py)
    additionally caches each bead's (`es_bead`) and each rigid body's
    (`es_body`) static EV energy in the state so only the proposed position is
    queried against the grid.

## Preconditions (enforced by `assert`)
1. `n_static > 0` and `n_static < len(rows)`: the EV restraint has both static
   and moving particles (`assert len(P) > 0`, `assert ns > 0`, `assert ns < n`).
2. `n_static <= np.iinfo(np.uint16).max` (`65,535`): static sphere indices
   including the sentinel index `n_static` fit in `uint16`
   (`assert len(P) <= np.iinfo(np.uint16).max`).
3. Positive radii and cell width (`assert rq > 0. and float(R.max()) > 0. and h > 0.`).
4. Disjoint exclusion labels between static and moving EV particles so cell
   lookup needs no per-pair label comparison
   (`assert not (set(self.lab[:ns]) & set(self.lab[ns:]))`).

## Measured gain
- **RNA Polymerase II (`polii`)**: reduces static EV queries from `369` spheres
  to `K = 21` candidates per cell (`h = 6.0 Å`, `17.6×` fewer pairs per query).
  - On **1 NVIDIA L40 GPU (`float32`, 32,768 chains)**: part of the `2×`
    speedup in `few_bodies_beads` (`~1.16 bn` MC steps/s).
  - On **1 CPU core (`float64`, 1 chain)**: combined with R4–R7, improves
    throughput by `+23%` (`49.9k -> 61.4k` MC steps/s).

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing Metropolis decisions against
  `Engine(s)` in `float64` across `<= 2,048` and `> 4,096` chains at `kT = 1`
  and `kT = 10`). A `0.01 Å` cell-face margin in `build_grid` ensures
  `float32` floor/clip cell indexing never misses a touching sphere.
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r1_static_ev_grid:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r1_static_ev_grid:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r1_static_ev_grid.py`](r1_static_ev_grid.py):

```python
def build_grid(P, R, rq, h=6.0):
    P = np.asarray(P, np.float64)
    R = np.asarray(R, np.float64)
    assert P.ndim == 2 and P.shape[1] == 3 and len(P) > 0
    assert len(P) <= np.iinfo(np.uint16).max
    assert rq > 0. and float(R.max()) > 0. and h > 0.
    pad = rq + float(R.max()) + h
    lo = P.min(0) - pad
    n = np.ceil((P.max(0) + pad - lo) / h).astype(np.int64)
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'),
                   -1).reshape(-1, 3)
    cen = lo + h * (ijk + 0.5)
    dd = np.maximum(np.abs(cen[:, None, :] - P[None]) - h / 2, 0.)
    cand = np.sqrt((dd ** 2).sum(-1)) < rq + R[None] + 0.01
    K = max(int(cand.sum(1).max()), 1)
    table = np.full((len(cen) + 1, K), len(P), np.int32)
    for c in np.nonzero(cand.any(1))[0]:
        j = np.nonzero(cand[c])[0]
        table[c, :len(j)] = j
    P_pad = np.vstack([P, [1e4, 1e4, 1e4]]).T
    R_pad = np.r_[R, 0.]
    return dict(lo=lo, h=float(h), n=n, table=table, K=K, P=P_pad, R=R_pad,
                table_u16=table.astype(np.uint16),
                PR=np.ascontiguousarray(np.column_stack([P_pad.T, R_pad])))


def grid_ev(g, xq, rq, dt=None, sum_rows=False):
    if dt is None:
        dt = xq.dtype
    n = g['n']
    idx = jnp.floor((xq - jnp.asarray(g['lo'], dt)[:, None, None]) / g['h'])
    idx = jnp.clip(idx, -1, jnp.asarray(n, dt)[:, None, None]).astype(jnp.int32)
    idx0, idx1, idx2 = idx[0], idx[1], idx[2]
    n0, n1, n2 = int(n[0]), int(n[1]), int(n[2])
    inside = ((idx0 >= 0) & (idx0 < n0)
              & (idx1 >= 0) & (idx1 < n1)
              & (idx2 >= 0) & (idx2 < n2))
    flat = jnp.where(inside, (idx0 * n1 + idx1) * n2 + idx2,
                     len(g['table']) - 1)
    cand = jnp.asarray(g['table_u16'])[flat.astype(jnp.uint32)]
    pr = jnp.asarray(g['PR'], dt)[cand]
    d = dist3(pr[..., 0], pr[..., 1], pr[..., 2],
              xq[0, ..., None], xq[1, ..., None], xq[2, ..., None])
    e = soft_pair(d, jnp.asarray(rq, dt)[:, None, None] + pr[..., 3])
    return jnp.sum(e, axis=(0, -1) if sum_rows else -1)
```
