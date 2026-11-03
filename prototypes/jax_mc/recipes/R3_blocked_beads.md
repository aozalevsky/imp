# R3: Blocked bead steps with exact within-block corrections

## What it does
Groups `B` consecutive `BallMover` steps on contiguous bead rows
`[o, o + B)` into a single block (`build_exec_segs`):
1. Scores all `B` proposed and old bead positions against external dynamic EV
   rows (`mask_ext[m, o:o+m+1] = False`) before resolving the sequential
   Metropolis decisions:
   - For `C <= 4,096` chains (`_bead_block_batched`), reduces `(B, n_de, C)` in
     one 3D kernel to saturate GPU SMs.
   - For `C > 4,096` chains (`_bead_block_chained`), chains `B` single-output
     `(n_de, C)` reductions via data-dependent masks and `optimization_barrier`
     so XLA's `MultiOutputFusion` keeps each reduction as a fast 1-output warp
     reduction while fusing all `B` Metropolis decisions into a single kernel.
2. Resolves the `B` sequential Metropolis steps `m = 0 .. B - 1` inside one
   fused kernel, adding the exact within-block `m2 < m` EV, connectivity
   (`Pairs`), and cross-link (`CrossLinks`) interactions against the already
   accepted/rejected positions `(xcx[m2], xcy[m2], xcz[m2])`.
3. Writes all `B` updated bead rows `X[:, o:o + B]` back to `X` in a single
   in-place `dynamic_update_slice_in_dim`.

- **Specialization level**: **Level 3 (Sweep / block execution in
  `few_bodies_beads`)**. Enabled by default on GPU (`blocked=True`) and
  disabled on CPU (`blocked=False`).

## Preconditions (enforced by `assert`)
1. Every `BallMover` moves one bead, and bead movers appear in contiguous bead
   row order `[bead_off, bead_off + nbead)`
   (`assert np.array_equal([p['row'] for p in bead_steps], np.arange(bead_off, bead_off + s.nbead))`).
2. Bead rows lie at the end of the dynamic EV slice `[0, n_de)`
   (`assert bead_off + s.nbead == n_de`).
3. No EM Gaussians lie on bead rows
   (`assert not np.isin(t.rows_read, bead_sys_rows).any()`).
4. Every term reading bead rows has an exact within-block `m2 < m` correction
   (`SoftSphereEV`, `Pairs`, and unambiguous `CrossLinks` with `cmax == 1`).

## Measured gain
- **RNA Polymerase II (`polii`) on 1 NVIDIA L40 GPU (`float32`)**:
  - Cuts CUDA kernel launches per sweep from `~340` to `~28` (`12×` fewer
    launches per sweep).
  - Delivers `+57%` to `+90%` higher throughput across `1,024–65,536` chains
    on GPU (`~1.16 bn` MC steps/s peak on 1 L40).
  - On **1 CPU core**, unblocked 1-bead steps are `19%` faster (no GPU launch
    overhead to amortize, and blocked masks add scalar work), so `FewBodiesBeads`
    defaults to `blocked = (jax.default_backend() != 'cpu')`.

## Exactness level & `check` call
- **Exactness**: **Level D** (`0` differing Metropolis decisions against
  `Engine(s)` in `float64` at both `<= 2,048` and `> 4,096` chains).
- **Check command**:
  ```bash
  python -m jax_mc.check decisions polii \
      --sampler recipes.r3_blocked_beads:factory \
      --chains 1024 --ref-chains 256 --kT 1,10 --cov rotate

  python -m jax_mc.check decisions polii \
      --sampler recipes.r3_blocked_beads:factory \
      --chains 8192 --ref-chains 128 --kT 1,10 --cov rotate
  ```

## Reference code
From [`recipes/r3_blocked_beads.py`](r3_blocked_beads.py):

```python
def build_exec_segs(steps, n_de, max_blk):
    assert max_blk >= 1
    segs, cur = [], []
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
            B = len(val)
            o = steps[val[0]]['row']
            mask_ext = np.zeros((B, n_de), dtype=bool)
            for m, i in enumerate(val):
                mask_ext[m] = steps[i]['evmask_dyn'].copy()
                mask_ext[m, o:o + m + 1] = False
            exec_segs.append(('bead_blk', val, mask_ext))
    return exec_segs
```
