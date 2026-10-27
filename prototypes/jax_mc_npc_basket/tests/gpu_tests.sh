#!/bin/bash
# GPU: bit-identity of continued runs (256 chains) and checkpoint/traj cost (2,048 chains)
[ -n "$ENV_SH" ] && source "$ENV_SH"   # optional: environment (IMP, JAX venv)
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=${GPU:-2} JAX_DEFAULT_MATMUL_PRECISION=highest
cd "${NPC:?set NPC to the NPC_Basket clone that has bench/}/bench/jax"
T=$(cd "$(dirname "$0")" && pwd); R=$T/gpu_runs; mkdir -p $R
wait_lines() { while kill -0 $3 2>/dev/null; do
    [ -f $1 ] && [ $(wc -l < $1) -ge $2 ] && return 0; sleep 0.5; done; return 1; }
stopped() { grep -E "stops after|continuing|frames done|Traceback|Error" $1 | tail -3; }
RUN="python -u -W ignore run_npc_jax.py yeast --chains 256 --rex 2 --thin 10 --check-every 50 --seed 7 --one-stage --shuffle-each"
echo "=== G1. 256 chains: reference 300 frames"
rm -rf $R/g_ref $R/g_cut
$RUN --frames2 300 --cpt 0 --out $R/g_ref > $R/g_ref.log 2>&1; stopped $R/g_ref.log; grep steps_per_s $R/g_ref.log | cut -c1-300
echo "=== G1. SIGTERM at >= 90, resume, kill -9 at >= 200 (checkpoint every 6 s), resume"
$RUN --frames2 300 --cpt 0.1 --cpi --out $R/g_cut > $R/g_cut_1.log 2>&1 &
P=$!; wait_lines $R/g_cut/one_stage/trace.jsonl 90 $P && kill -TERM $P; wait $P; stopped $R/g_cut_1.log
$RUN --frames2 300 --cpt 0.1 --cpi --out $R/g_cut > $R/g_cut_2.log 2>&1 &
P=$!; wait_lines $R/g_cut/one_stage/trace.jsonl 200 $P && kill -KILL $P; wait $P; echo "killed: trace has $(wc -l < $R/g_cut/one_stage/trace.jsonl) lines"; grep "checkpoint after" $R/g_cut_2.log | tail -1
$RUN --frames2 300 --cpt 0.1 --cpi --out $R/g_cut > $R/g_cut_3.log 2>&1; stopped $R/g_cut_3.log
python $T/compare_runs.py $R/g_ref $R/g_cut one_stage

RUN2="python -u -W ignore run_npc_jax.py yeast --chains 2048 --thin 10 --seed 7 --one-stage"
echo "=== G2. 2,048 chains, 300 frames: no checkpoint until the end"
rm -rf $R/c_nock $R/c_ck
$RUN2 --frames2 300 --cpt 0 --out $R/c_nock > $R/c_nock.log 2>&1; stopped $R/c_nock.log; grep -E "checkpoint after|steps_per_s" $R/c_nock.log | cut -c1-400
echo "=== G2. 2,048 chains, 300 frames: checkpoint every 30 s"
$RUN2 --frames2 300 --cpt 0.5 --out $R/c_ck > $R/c_ck.log 2>&1; stopped $R/c_ck.log; grep -E "checkpoint after|steps_per_s" $R/c_ck.log | cut -c1-400
ls -la $R/c_ck/state.cpt $R/c_ck/one_stage/
python - $R/c_ck/one_stage/traj.dof <<'PY'
import sys; sys.path.insert(0, '.')
import npc_ckpt
h, r = npc_ckpt.read_dof(sys.argv[1])
print('traj.dof:', len(r), 'records of', r.dtype.itemsize, 'bytes =', r.dtype.itemsize / h['ladders'], 'bytes per chain and saved frame;', h['n_free'], 'free rows,', h['nb'], 'bodies,', h['n_nuis'], 'nuisances')
PY
echo GPU_TESTS_DONE
