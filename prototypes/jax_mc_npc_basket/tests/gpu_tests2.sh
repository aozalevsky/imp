#!/bin/bash
# GPU identity with the compilation cache (A), with autotuning off (B), and its speed cost (C)
[ -n "$ENV_SH" ] && source "$ENV_SH"   # optional: environment (IMP, JAX venv)
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=${GPU:-2} JAX_DEFAULT_MATMUL_PRECISION=highest
cd "${NPC:?set NPC to the NPC_Basket clone that has bench/}/bench/jax"
T=$(cd "$(dirname "$0")" && pwd); R=$T/gpu_runs2; rm -rf $R; mkdir -p $R
wait_lines() { while kill -0 $3 2>/dev/null; do
    [ -f $1 ] && [ $(wc -l < $1) -ge $2 ] && return 0; sleep 0.5; done; return 1; }
stopped() { grep -E "stops after|continuing|frames done|Traceback|Error" $1 | tail -2; }
RUN="python -u -W ignore run_npc_jax.py yeast --chains 256 --rex 2 --thin 10 --check-every 50 --seed 7 --one-stage --shuffle-each --frames2 300"
scenario() {   # name, extra args for every session
  n=$1; shift
  $RUN --cpt 0 "$@" --out $R/${n}_ref > $R/${n}_ref.log 2>&1; stopped $R/${n}_ref.log
  $RUN --cpt 0.1 --cpi "$@" --out $R/${n}_cut > $R/${n}_cut1.log 2>&1 &
  P=$!; wait_lines $R/${n}_cut/one_stage/trace.jsonl 90 $P && kill -TERM $P; wait $P; stopped $R/${n}_cut1.log
  $RUN --cpt 0.1 --cpi "$@" --out $R/${n}_cut > $R/${n}_cut2.log 2>&1 &
  P=$!; wait_lines $R/${n}_cut/one_stage/trace.jsonl 200 $P && kill -KILL $P; wait $P; echo "  killed at $(wc -l < $R/${n}_cut/one_stage/trace.jsonl) trace lines"
  $RUN --cpt 0.1 --cpi "$@" --out $R/${n}_cut > $R/${n}_cut3.log 2>&1; stopped $R/${n}_cut3.log
  python $T/compare_runs.py $R/${n}_ref $R/${n}_cut one_stage | tail -4
}
echo "=== A. reference and interrupted run share one compilation cache"
scenario A --jax-cache $R/A_cache
echo "=== B. autotuning off, separate caches"
XLA_FLAGS=--xla_gpu_autotune_level=0 scenario B
echo "=== C. speed at 2,048 chains, 150 frames: autotuning on / off"
RUN2="python -u -W ignore run_npc_jax.py yeast --chains 2048 --thin 10 --seed 7 --one-stage --frames2 150 --cpt 0"
$RUN2 --out $R/C_on > $R/C_on.log 2>&1; grep -h steps_per_s $R/C_on.log | python3 -c "import sys,json; [print('  on ', round(json.loads(l.strip())['steps_per_s_wall'])) for l in sys.stdin]"
XLA_FLAGS=--xla_gpu_autotune_level=0 $RUN2 --out $R/C_off > $R/C_off.log 2>&1; grep -h steps_per_s $R/C_off.log | python3 -c "import sys,json; [print('  off', round(json.loads(l.strip())['steps_per_s_wall'])) for l in sys.stdin]"
echo "=== restart time with a warm cache (session setup to first frame)"
grep -h "setup\|frame 30/" $R/A_cut3.log | head -2 | cut -c1-200
echo GPU2_DONE
