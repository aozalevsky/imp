#!/bin/bash
# GPU run-to-run reproducibility: two identical uninterrupted runs per XLA setting
[ -n "$ENV_SH" ] && source "$ENV_SH"   # optional: environment (IMP, JAX venv)
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=${GPU:-2} JAX_DEFAULT_MATMUL_PRECISION=highest
T=$(cd "$(dirname "$0")" && pwd); R=$T/det_runs; rm -rf $R; mkdir -p $R
cd "${NPC:?set NPC to the NPC_Basket clone that has bench/}/bench/jax"
echo "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-unset}"
python -c "import jax; print('jax', jax.__version__, 'cache dir', jax.config.jax_compilation_cache_dir)"
RUN="python -u -W ignore run_npc_jax.py yeast --chains 256 --rex 2 --thin 10 --seed 7 --one-stage --shuffle-each --frames2 20 --cpt 0"
for cfg in default det autotune0; do
  case $cfg in
    default) X="";;
    det) X="--xla_gpu_deterministic_ops=true";;
    autotune0) X="--xla_gpu_autotune_level=0";;
  esac
  for k in 1 2; do
    XLA_FLAGS="$X" $RUN --out $R/${cfg}_$k > $R/${cfg}_$k.log 2>&1
    grep -E "Traceback|Error" $R/${cfg}_$k.log | head -3
  done
  python - $R/${cfg}_1 $R/${cfg}_2 $cfg <<'PY'
import json, sys
A = [json.loads(l)['score'] for l in open(sys.argv[1] + '/one_stage/trace.jsonl')]
B = [json.loads(l)['score'] for l in open(sys.argv[2] + '/one_stage/trace.jsonl')]
nd = [sum(x != y for x, y in zip(a, b)) for a, b in zip(A, B)]
first = next((i for i, v in enumerate(nd) if v), None)
print(f'{sys.argv[3]}: first differing frame {first}; chains differing at frames 0, 5, 19: {nd[0]}, {nd[5]}, {nd[19]}')
PY
  grep -h steps_per_s $R/${cfg}_1.log | python3 -c "import sys,json; [print('  ', json.loads(l)['steps_per_s_wall']) for l in sys.stdin]" 2>/dev/null
done
echo DET_DONE
