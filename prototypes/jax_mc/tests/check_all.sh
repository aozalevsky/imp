#!/bin/bash
# Every exactness check of jax_mc, its templates and its recipes.
#
#   tests/check_all.sh WORKDIR [quick|cpu]
#
# Run from prototypes/jax_mc on a GPU node, with IMP, JAX and the
# JAX_MC_* system directories set (jax_mc/sources.py). WORKDIR holds the
# run-driver outputs. It is overwritten by the next run, never deleted.
# 'quick' skips the large-chain regime and the float32 reports; 'cpu' runs
# only the parts that need no GPU (terms and the run driver).
#
# Exit status 1 if any check fails. A new template or recipe adds its
# `check decisions` line to the templates section (GUIDE.md, step 5).
set -u
W=${1:?usage: tests/check_all.sh WORKDIR [quick|cpu]}
MODE=${2:-full}
mkdir -p "$W"
failed=()
run() {
    echo "== $*"
    "$@" || failed+=("$*")
}
cpu() { JAX_PLATFORMS=cpu "$@"; }
t0=$SECONDS

echo "### level T: every term against C++ (CPU, float64)"
for s in mzid polii npc pm; do
    run cpu python -m jax_mc.check terms $s
done

if [ "$MODE" != cpu ]; then
echo "### level D: the engine against its full rescore (float64)"
for s in mzid polii npc; do
    run python -m jax_mc.check decisions $s --chains 1024 --ref-chains 256 \
        --sweeps 5 --kT 1,10
    [ "$MODE" = full ] && run python -m jax_mc.check decisions $s \
        --chains 8192 --ref-chains 128 --sweeps 3 --kT 1
done

echo "### level D: templates against the reference engine (float64)"
run python -m jax_mc.check decisions pm \
    --sampler templates.collective.sampler:factory \
    --chains 256 --ref-chains 16 --sweeps 200 --kT 300,3000
# few_bodies_beads (Pol II) and recipes R1-R7: add their lines here

if [ "$MODE" = full ]; then
    echo "### level F: float32 drift, reported only"
    for s in mzid polii npc; do
        run python -m jax_mc.check drift $s --chains 1024 --sweeps 50 --float32
    done
fi
fi

echo "### run driver: restart and ranks equal one run (mzIdentML, CPU)"
O="mzid --sweeps 5 --ladder 1,2,4,8 --thin 10 --block 5 --cpt 0 --float64"
ok=1
cpu python -m jax_mc.run $O --chains 32 --frames 40 --out "$W/one" \
    > "$W/one.log" 2>&1 || ok=0
cpu python -m jax_mc.run $O --chains 32 --frames 20 --out "$W/restart" \
    > "$W/restart.log" 2>&1 || ok=0
cpu python -m jax_mc.run $O --chains 32 --frames 40 --out "$W/restart" --cpi \
    >> "$W/restart.log" 2>&1 || ok=0
for r in 0 1; do
    cpu python -m jax_mc.run $O --chains 16 --frames 40 --ranks 2 --rank $r \
        --out "$W/ranks" > "$W/ranks$r.log" 2>&1 || ok=0
done
[ $ok = 1 ] || failed+=("run driver: a run failed, see $W/*.log")
[ $ok = 1 ] && run cpu python - "$W" <<'PY'
import sys
import numpy as np
from jax_mc import run
w = sys.argv[1]
f1, _, E1, k1 = run.records(f'{w}/one')
_, D1 = run.snapshots(f'{w}/one')
bad = []
def same(name, f, E, k, D, sl):
    # positions and temperatures exactly; rescored energies can differ in
    # the last place between runs (DESIGN.md, run driver)
    dE = np.abs(E - E1[:, sl]).max() / np.abs(E1[:, sl]).max()
    dk = np.abs(k - k1[:, sl]).max()
    dD = max(np.abs(D[n] - D1[n][:, sl]).max() for n in D if D[n].size)
    ok = len(f) == len(f1) and dk == 0 and dD == 0 and dE < 1e-12
    print(f'  {name}: {len(f)} frames; kT {dk}, dofs {dD}, energy (rel) '
          f'{dE:.1e}: {"ok" if ok else "DIFFERENT"}')
    if not ok:
        bad.append(name)
f, _, E, k = run.records(f'{w}/restart')
same('restart at frame 20', f, E, k, run.snapshots(f'{w}/restart')[1],
     slice(None))
for r in (0, 1):
    f, _, E, k = run.records(f'{w}/ranks/rank{r}')
    same(f'rank {r} of 2', f, E, k, run.snapshots(f'{w}/ranks/rank{r}')[1],
         slice(16 * r, 16 * (r + 1)))
sys.exit(1 if bad else 0)
PY

echo "### $(( (SECONDS - t0) / 60 )) min"
if [ ${#failed[@]} -gt 0 ]; then
    printf 'FAILED: %s\n' "${failed[@]}"
    exit 1
fi
echo "all checks passed"
