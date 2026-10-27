#!/bin/bash
# checkpoint/restart tests of run_npc_jax.py on CPU (small system)
[ -n "$ENV_SH" ] && source "$ENV_SH"   # optional: environment (IMP, JAX venv)
export JAX_PLATFORMS=cpu
T=$(cd "$(dirname "$0")" && pwd); R=$T/runs; mkdir -p $R
cd "${NPC:?set NPC to the NPC_Basket clone that has bench/}/bench/jax"
RUN="nice -n 10 python -u -W ignore run_npc_jax.py yeast --chains 8 --rex 2 --thin 2 --check-every 3 --seed 3"
wait_lines() {   # file n pid: until file has n lines (or pid exits)
  while kill -0 $3 2>/dev/null; do
    [ -f $1 ] && [ $(wc -l < $1) -ge $2 ] && return 0; sleep 1; done; return 1; }
stopped() { grep -E "stops after|checkpoint after|continuing|frames done|Traceback|Error" $1 | tail -4; }

echo "=== 1. one stage: reference, 16 frames uninterrupted (+ live RMF)"
rm -rf $R/one_ref $R/one_cut
$RUN --one-stage --shuffle-each --frames2 16 --cpt 0 --rmf-live --out $R/one_ref > $R/one_ref.log 2>&1
grep -E "Traceback|Error" $R/one_ref.log; tail -2 $R/one_ref.log | cut -c1-200

echo "=== 1. one stage: SIGTERM at frame >= 4"
$RUN --one-stage --shuffle-each --frames2 12 --cpt 0.05 --cpi --out $R/one_cut > $R/one_cut_1.log 2>&1 &
P=$!; wait_lines $R/one_cut/one_stage/trace.jsonl 4 $P && kill -TERM $P; wait $P; echo "exit $?"; stopped $R/one_cut_1.log
echo "=== 1. resume, kill -9 at frame >= 9"
$RUN --one-stage --shuffle-each --frames2 12 --cpt 0.05 --cpi --out $R/one_cut > $R/one_cut_2.log 2>&1 &
P=$!; wait_lines $R/one_cut/one_stage/trace.jsonl 9 $P && kill -KILL $P; wait $P; echo "exit $?"; stopped $R/one_cut_2.log
wc -l $R/one_cut/one_stage/trace.jsonl
echo "=== 1. resume to 12 frames"
$RUN --one-stage --shuffle-each --frames2 12 --cpt 0.05 --cpi --out $R/one_cut > $R/one_cut_3.log 2>&1; echo "exit $?"; stopped $R/one_cut_3.log
echo "=== 1. same command again: nothing to do"
$RUN --one-stage --shuffle-each --frames2 12 --cpt 0.05 --cpi --out $R/one_cut > $R/one_cut_4.log 2>&1; echo "exit $?"; stopped $R/one_cut_4.log
echo "=== 1. extend to 16 frames"
$RUN --one-stage --shuffle-each --frames2 16 --cpt 0.05 --cpi --out $R/one_cut > $R/one_cut_5.log 2>&1; echo "exit $?"; stopped $R/one_cut_5.log
python $T/compare_runs.py $R/one_ref $R/one_cut one_stage
echo "=== 1. no --cpi on a directory with a checkpoint must refuse"
$RUN --one-stage --shuffle-each --frames2 16 --out $R/one_cut 2>&1 | tail -1

echo "=== 2. two stages: reference REX1 6 + REX2 8"
rm -rf $R/two_ref $R/two_cut
$RUN --frames1 6 --frames2 8 --cpt 0 --out $R/two_ref > $R/two_ref.log 2>&1
grep -E "Traceback|Error" $R/two_ref.log; tail -2 $R/two_ref.log | cut -c1-200
echo "=== 2. SIGTERM in REX1 at frame >= 3"
$RUN --frames1 6 --frames2 8 --cpt 0.05 --cpi --out $R/two_cut > $R/two_cut_1.log 2>&1 &
P=$!; wait_lines $R/two_cut/rex1/trace.jsonl 3 $P && kill -TERM $P; wait $P; echo "exit $?"; stopped $R/two_cut_1.log
echo "=== 2. resume: finish REX1, SIGTERM in REX2 at frame >= 3"
$RUN --frames1 6 --frames2 8 --cpt 0.05 --cpi --out $R/two_cut > $R/two_cut_2.log 2>&1 &
P=$!; wait_lines $R/two_cut/rex2/trace.jsonl 3 $P && kill -TERM $P; wait $P; echo "exit $?"; stopped $R/two_cut_2.log
echo "=== 2. resume to the end"
$RUN --frames1 6 --frames2 8 --cpt 0.05 --cpi --out $R/two_cut > $R/two_cut_3.log 2>&1; echo "exit $?"; stopped $R/two_cut_3.log
python $T/compare_runs.py $R/two_ref $R/two_cut rex1 rex2

echo "=== 3. RMF from traj.dof (one-stage reference): best 5 and ladder 0"
$RUN --one-stage --shuffle-each --frames2 16 --out $R/one_ref --to-rmf one_stage --best 5 --ladder 0 > $R/one_ref_rmf.log 2>&1
grep -E "wrote|Traceback|Error" $R/one_ref_rmf.log; head -3 $R/one_ref/one_stage/best_5.csv
echo "=== 3. live chain_0.rmf3 vs ladder_0.rmf3 from traj.dof"
python $T/compare_rmf.py $R/one_ref/one_stage/chain_0.rmf3 $R/one_ref/one_stage/ladder_0.rmf3
echo TESTS_DONE
