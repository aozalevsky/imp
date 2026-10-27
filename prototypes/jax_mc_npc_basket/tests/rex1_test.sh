#!/bin/bash
# stop and continue inside REX1 (CPU, small)
[ -n "$ENV_SH" ] && source "$ENV_SH"   # optional: environment (IMP, JAX venv)
export JAX_PLATFORMS=cpu
cd "${NPC:?set NPC to the NPC_Basket clone that has bench/}/bench/jax"
T=$(cd "$(dirname "$0")" && pwd); R=$T/runs; mkdir -p $R
RUN="nice -n 10 python -u -W ignore run_npc_jax.py yeast --chains 8 --rex 2 --thin 2 --check-every 3 --seed 5"
wait_lines() { while kill -0 $3 2>/dev/null; do
    [ -f $1 ] && [ $(wc -l < $1) -ge $2 ] && return 0; sleep 1; done; return 1; }
stopped() { grep -E "stops after|continuing|frames done|Traceback|Error" $1 | tail -3; }
rm -rf $R/r1_ref $R/r1_cut
$RUN --frames1 14 --frames2 4 --cpt 0 --out $R/r1_ref > $R/r1_ref.log 2>&1; stopped $R/r1_ref.log
$RUN --frames1 14 --frames2 4 --cpt 0.05 --cpi --out $R/r1_cut > $R/r1_cut_1.log 2>&1 &
P=$!; wait_lines $R/r1_cut/rex1/trace.jsonl 3 $P && kill -TERM $P; wait $P; stopped $R/r1_cut_1.log
$RUN --frames1 14 --frames2 4 --cpt 0.05 --cpi --out $R/r1_cut > $R/r1_cut_2.log 2>&1; stopped $R/r1_cut_2.log
python $T/compare_runs.py $R/r1_ref $R/r1_cut rex1 rex2
echo "=== --to-rmf with only --out (settings from the checkpoint)"
python -u -W ignore run_npc_jax.py --out $R/r1_cut --to-rmf rex2 --best 3 > $R/r1_rmf.log 2>&1; grep -E "wrote|Traceback|Error" $R/r1_rmf.log
echo REX1_DONE
