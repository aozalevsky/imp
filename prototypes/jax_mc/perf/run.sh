#!/bin/bash
# The measurements behind perf/speed.csv and perf/speed.png.
#
#   ENV_RELEASE='source <env with a release IMP build>' \
#   ENV_FAST='source <env with a fast IMP build>' \
#   perf/run.sh OUT [GPU] [CPU_CORE] [GPU_CORES]
#
# Run from prototypes/jax_mc. Both environments need JAX and the JAX_MC_*
# system directories (jax_mc/sources.py); JAX runs use ENV_RELEASE (IMP only
# builds the system there). Two streams run at the same time:
#   CPU: C++ release, C++ fast, the JAX core engine with one chain and the
#        system's template with one chain, one after the other, pinned to
#        core CPU_CORE (single-threaded XLA);
#   GPU: JAX on GPU (CUDA device index): the core engine with one chain,
#        then the system's template at growing chain counts (each in its
#        own process, so running out of memory ends only that point), host
#        threads on GPU_CORES; then the core engine at growing chain counts.
# SYSTEMS and STREAMS (cpu, gpu, or core for only the core-engine part of
# gpu) in the environment run a subset.
# Results are appended to OUT/results.jsonl, logs to OUT/*.log; then
#   python -m perf.one_vs_many plot OUT/results.jsonl OUT
set -u
O=${1:?usage: perf/run.sh OUT [GPU] [CPU_CORE] [GPU_CORES]}
GPU=${2:-0}; CORE=${3:-8}; GCORES=${4:-4-7}
: "${ENV_RELEASE:?set ENV_RELEASE}" "${ENV_FAST:?set ENV_FAST}"
mkdir -p "$O"
R=$O/results.jsonl
P="python -m perf.one_vs_many"
keep() { grep '^RESULT ' | sed 's/^RESULT //' >> "$R"; }

# per system: C++ steps per call and score_moved modes to time; burn-in (C++
# steps, JAX template sweeps); JAX core sweeps per call with one chain on
# the GPU and on the CPU; sweeps per call with many chains; GPU chain counts
# to try. The mature NPC moves every body at every step, so
# score_moved has nothing to skip there and runs far slower; not timed.
declare -A CPP_STEPS=([mzid]=80000 [npc]=3540 [pm]=200 [polii]=24500)
declare -A MODES=([mzid]=False,True [npc]=False,True [pm]=False
                  [polii]=False,True)
declare -A BURN=([mzid]=0 [npc]=0 [pm]=20000 [polii]=0)
declare -A SW1=([mzid]=20000 [npc]=100 [pm]=100 [polii]=2000)
declare -A SWC=([mzid]=20000 [npc]=100 [pm]=3 [polii]=2000)
# template, one chain on the CPU (not the NPC basket: its unrolled sweep
# didn't finish compiling on one core within 39 minutes)
declare -A SWT=([mzid]=20000 [pm]=500 [polii]=2000)
declare -A SWN=([mzid]=100 [npc]=20 [pm]=500 [polii]=200)
declare -A MANY=([mzid]="32768 65536 131072 262144"
                 [npc]="4096 8192 12288"
                 [pm]="512 1024 2048 4096"
                 [polii]="16384 32768 65536 131072")
# the core engine: chain counts, sweeps per call. The core runs without
# burn-in: only the mature NPC's template has one, and its core engine
# rescores everything at every step (all-pairs EV), so its speed doesn't
# depend on the state.
declare -A CORE_MANY=([mzid]="65536 262144 524288 1048576"
                      [npc]="8192 12288 16384 32768 65536"
                      [pm]="64 256 1024" [polii]="16384 32768 65536 131072")
declare -A CORE_SW=([mzid]=100 [npc]=20 [pm]=5 [polii]=200)
SYSTEMS=${SYSTEMS:-mzid npc pm polii}
STREAMS=${STREAMS:-cpu gpu}

cpu_stream() {
    for s in $SYSTEMS; do
        for b in release fast; do
            [ $b = release ] && E=$ENV_RELEASE || E=$ENV_FAST
            ( eval "$E"; taskset -c $CORE $P cpp $s --steps ${CPP_STEPS[$s]} \
                --burn ${BURN[$s]} --modes ${MODES[$s]} ) \
                2> "$O/cpp_${b}_$s.log" | keep
        done
        ( eval "$ENV_RELEASE"; JAX_PLATFORMS=cpu \
          XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
          taskset -c $CORE $P jax $s --core --chains 1 --sweeps ${SWC[$s]} \
          ) 2> "$O/jax_cpu_$s.log" | keep
        [ -n "${SWT[$s]:-}" ] && ( eval "$ENV_RELEASE"; JAX_PLATFORMS=cpu \
          XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
          taskset -c $CORE $P jax $s --chains 1 --sweeps ${SWT[$s]} \
            --burn ${BURN[$s]} ) 2> "$O/jax_cpu_template_$s.log" | keep
    done
}

gpu_stream() {
    for s in $SYSTEMS; do
        ( eval "$ENV_RELEASE"; JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$GPU \
          taskset -c $GCORES $P jax $s --core --chains 1 --sweeps ${SW1[$s]} \
        ) 2> "$O/jax_gpu_${s}_1.log" | keep
        for c in ${MANY[$s]}; do
            ( eval "$ENV_RELEASE"; JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$GPU \
              taskset -c $GCORES $P jax $s --chains $c --sweeps ${SWN[$s]} \
                --burn ${BURN[$s]} ) 2> "$O/jax_gpu_${s}_$c.log" | keep
        done
    done
    core_stream
}

core_stream() {
    for s in $SYSTEMS; do
        for c in ${CORE_MANY[$s]:-}; do
            ( eval "$ENV_RELEASE"; JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$GPU \
              taskset -c $GCORES $P jax $s --core --chains $c \
                --sweeps ${CORE_SW[$s]} ) 2> "$O/jax_gpu_core_${s}_$c.log" | keep
        done
    done
}

for st in $STREAMS; do ${st}_stream & done
wait
echo "$(wc -l < "$R") results in $R"
