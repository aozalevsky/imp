# Tests of checkpoints and restarts

Each script compares a run that is stopped (SIGTERM, `kill -9`) and
continued with `--cpi` against the same run uninterrupted: trace, checks,
`traj.dof` and every array of the final `state.cpt` must be identical
(`compare_runs.py`). Set `NPC` to the NPC_Basket clone that has `bench/`
(see `../install.sh`) and optionally `ENV_SH` to a script that sets up IMP
and JAX.

| script | what |
|---|---|
| `run_tests.sh` | CPU, 8 runs in ladders of 2: one stage (SIGTERM, `kill -9`, nothing to do, extend 12 -> 16 frames), two stages (stop at the REX1/REX2 boundary and in REX2), RMF from `traj.dof` vs RMF written while sampling (`compare_rmf.py`) |
| `rex1_test.sh` | CPU: stop and continue inside REX1; `--to-rmf` with only `--out` |
| `gpu_tests.sh` | GPU (`GPU=i`): 256 runs, 300 frames (SIGTERM at 91, `kill -9` at 200); checkpoint cost at 2,048 runs |
| `gpu_tests2.sh` | GPU: the same with a shared compilation cache, and with autotuning off and separate caches; speed with autotuning on and off |
| `det_test.sh` | GPU: two identical runs per XLA setting (default, deterministic ops, autotuning off) |
| `perf_headroom.py` | time per MC step against the number of runs, kernels per step, XLA's operation counts per mover |

Results (2026-10-26, L40 and CPU): all comparisons identical once
compiled programs are shared (`--out/jax_cache`) or autotuning is off;
see the "Long runs" section of `../README.md`.
