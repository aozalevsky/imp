#!/bin/bash
# The measurements behind perf/node_results.jsonl: what one node does with
# all its GPUs (the jax_mc templates) and with all its CPU threads (C++
# IMP), for the NPC basket (npc) and the mature NPC (pm).
#
#   ENV_RELEASE='source <env with a release IMP build>' \
#   ENV_FAST='source <env with a fast IMP build>' \
#   perf/node.sh OUT
#
# Run from prototypes/jax_mc on an otherwise idle node. The environments are
# those of perf/run.sh; JAX runs use ENV_RELEASE. In the environment:
#   GPUS     CUDA devices to use (default 0,1,2,3)
#   NCORE    physical cores (default: counted with lscpu); logical CPUs
#            0..NCORE-1 must sit on different cores
#   NTHREAD  hardware threads (default: nproc)
# One after the other, each with perf/node.py (all processes of a run are
# timed over one common window):
#   JAX   one process per GPU on all GPUs at once, then on one GPU alone;
#   C++   fast build: one thread on the idle node, one process per core,
#         one process per hardware thread; release build: one thread.
# Results are appended to OUT/results.jsonl, logs go to OUT/*/.
set -u
O=${1:?usage: perf/node.sh OUT}
: "${ENV_RELEASE:?set ENV_RELEASE}" "${ENV_FAST:?set ENV_FAST}"
GPUS=${GPUS:-0,1,2,3}
NCORE=${NCORE:-$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)}
NTHREAD=${NTHREAD:-$(nproc)}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$O"
R=$O/results.jsonl
P="python -m perf.node drive"
keep() { grep '^NODE ' | sed 's/^NODE //' >> "$R"; }

# C++ steps per call and score_moved (the faster exact setting; the mature
# NPC moves every body at every step, so it has nothing to skip), burn-in;
# JAX chains per GPU (the template's best count) and sweeps per call
declare -A CPP=([npc]="--steps 3540 --moved True"
                [pm]="--steps 200 --burn 20000")
declare -A JAX=([npc]="--chains 8192 --sweeps 20"
                [pm]="--chains 1024 --sweeps 500 --burn 20000")
# the window in seconds; longer where a C++ call takes seconds under load
declare -A LONG=([npc]=60 [pm]=90)

( eval "$ENV_RELEASE"
  for s in npc pm; do
      $P "$O/jax_${s}_all" --kind jax --system $s --n "$NGPU" --gpus "$GPUS" \
          --cpus 0-$((4 * NGPU - 1)) --seconds 60 -- ${JAX[$s]} | keep
      $P "$O/jax_${s}_1" --kind jax --system $s --n 1 --gpus "${GPUS%%,*}" \
          --cpus 0-3 --seconds 60 -- ${JAX[$s]} | keep
  done )
( eval "$ENV_FAST"
  for s in npc pm; do
      $P "$O/cpp_fast_${s}_1" --kind cpp --system $s --n 1 --cpus 8 \
          --seconds 60 -- ${CPP[$s]} | keep
      $P "$O/cpp_fast_${s}_cores" --kind cpp --system $s --n "$NCORE" \
          --cpus 0-$((NCORE - 1)) --seconds ${LONG[$s]} -- ${CPP[$s]} | keep
      $P "$O/cpp_fast_${s}_threads" --kind cpp --system $s --n "$NTHREAD" \
          --cpus 0-$((NTHREAD - 1)) --seconds ${LONG[$s]} -- ${CPP[$s]} | keep
  done )
( eval "$ENV_RELEASE"
  for s in npc pm; do
      $P "$O/cpp_release_${s}_1" --kind cpp --system $s --n 1 --cpus 8 \
          --seconds 60 -- ${CPP[$s]} | keep
  done )
echo "$(wc -l < "$R") results in $R"
