"""MC speed of a whole node: N processes at once, timed over one window.

  python -m perf.node drive OUT --kind cpp --system npc --n 64 --cpus 0-63 \\
      --seconds 60 -- --steps 3540 --moved True
  python -m perf.node drive OUT --kind jax --system npc --n 4 --gpus 0,1,2,3 \\
      --cpus 0-15 --seconds 60 -- --chains 8192 --sweeps 20

Run from prototypes/jax_mc, in the environment the workers should use
(perf/node.sh has the runs behind perf/node_results.jsonl). The driver
starts N workers, each pinned to its share of --cpus (and, for jax, to one
of --gpus). Every worker sets its system up and burns in, reports ready and
waits; when all are ready the driver names a start time, and every worker
then runs MC for --seconds. The node's speed is the sum over the workers of
the MC steps they finish inside that window, divided by the window. The
driver prints it as one line 'NODE {json}'.

cpp workers: IMP.core.MonteCarlo as perf/one_vs_many.py cpp builds it, one
single-threaded process with one chain (--steps per optimize call, --moved
for score_moved, --burn untimed steps first with score_moved off).
jax workers: the system's template as perf/one_vs_many.py jax runs it
(--core for the core engine), float32, one process per GPU; --platform cpu
runs them on the CPU instead.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def barrier(d, tag):
    """report ready, then wait for the driver's start time"""
    open(os.path.join(d, f'ready_{tag}'), 'w').close()
    go = os.path.join(d, 'go')
    while not os.path.exists(go):
        time.sleep(0.05)
    t0 = float(open(go).read())
    time.sleep(max(0., t0 - time.time()))
    return t0


def timed(call, per_call, seconds):
    """run call() until `seconds` have passed; the steps done inside the
       window count the call that was cut by its end in proportion"""
    ends, t = [], time.perf_counter()
    while True:
        call()
        ends.append(time.perf_counter() - t)
        if ends[-1] >= seconds:
            break
    k = sum(e <= seconds for e in ends)       # calls finished in the window
    part = 0.
    if k < len(ends):
        a = ends[k - 1] if k else 0.
        part = (seconds - a) / (ends[k] - a)
    took = np.diff([0.] + ends)
    return dict(calls=len(ends), per_call=per_call, total_s=ends[-1],
                steps_in_window=(k + part) * per_call,
                steps_per_s=(k + part) * per_call / seconds,
                call_s_median=float(np.median(took)),
                call_s_max=float(np.max(took)))


def cpp(a):
    from jax_mc import sources
    import IMP
    import IMP.core
    IMP.set_log_level(IMP.SILENT)
    t_setup = time.time()
    src = sources.load(a.system)
    sf = IMP.core.RestraintsScoringFunction(src.restraints)
    moved = a.moved == 'True'

    def make(moved):
        mc = IMP.core.MonteCarlo(src.model)
        mc.set_scoring_function(sf)
        mc.set_kt(src.kT)
        mc.set_return_best(False)
        mc.set_score_moved(moved)
        if src.mode == 'serial':
            mc.add_mover(IMP.core.SerialMover(src.movers))
        else:
            mc.add_movers(src.movers)
        return mc

    if a.burn:
        make(False).optimize(a.burn)
    mc = make(moved)
    mc.optimize(max(a.steps // 10, 1))
    t_setup = time.time() - t_setup
    t0 = barrier(a.dir, a.tag)
    r = timed(lambda: mc.optimize(a.steps), a.steps, a.seconds)
    e_mc, e_full = mc.get_last_accepted_energy(), sf.evaluate(False)
    return dict(kind='cpp', system=a.system, imp=IMP.get_module_version(),
                imp_build=os.path.basename(os.path.dirname(os.path.dirname(
                    IMP.__file__))),            # lib/<build>/IMP
                check_level=int(IMP.get_check_level()), mode=src.mode,
                kT=src.kT, burn=a.burn, score_moved=moved, setup_s=t_setup,
                start=t0, rel_diff_vs_full=abs(e_mc - e_full)
                / max(abs(e_full), 1.), **r)


def jax_worker(a):
    from jax_mc import xla_defaults
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', False)
    from jax_mc import check, engine, sources, system
    from perf.one_vs_many import SAMPLERS
    t_setup = time.time()
    s = system.extract(sources.load(a.system))
    spec = SAMPLERS[a.system]
    core = a.core or spec is None
    smp = engine.make(s) if core else check._factory(spec)(s)
    n = smp.steps_per_sweep
    run = jax.jit(smp.run, static_argnums=(1, 2), donate_argnums=0)
    st = smp.init(a.chains, a.seed)
    st = jax.block_until_ready(run(st, a.sweeps))
    for _ in range(a.burn // a.sweeps):
        st = run(st, a.sweeps)
    st = jax.block_until_ready(st)
    acc0 = np.asarray(st['n_acc']).sum()
    t_setup = time.time() - t_setup
    box = [st]

    def call():
        box[0] = jax.block_until_ready(run(box[0], a.sweeps))
    t0 = barrier(a.dir, a.tag)
    r = timed(call, a.chains * n * a.sweeps, a.seconds)
    acc = (float(np.asarray(box[0]['n_acc']).sum() - acc0)
           / (r['calls'] * r['per_call']))
    dev = jax.devices()[0]
    return dict(kind='jax', system=a.system, sampler=type(smp).__name__,
                core=core, platform=dev.platform, device=dev.device_kind,
                chains=a.chains, sweeps=a.sweeps, steps_per_sweep=n,
                burn=a.burn, setup_s=t_setup, start=t0, acceptance=acc,
                jax=jax.__version__, **r)


def worker(a):
    out = cpp(a) if a.cmd == 'cpp' else jax_worker(a)
    out.update(tag=a.tag, seconds=a.seconds,
               cpus=sorted(os.sched_getaffinity(0)),
               gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    p = os.path.join(a.dir, f'res_{a.tag}.json')
    json.dump(out, open(p + '.tmp', 'w'))
    os.rename(p + '.tmp', p)


def cpulist(s):
    out = []
    for part in s.split(','):
        lo, _, hi = part.partition('-')
        out += range(int(lo), int(hi or lo) + 1)
    return out


def drive(a):
    os.makedirs(a.dir, exist_ok=True)
    for f in os.listdir(a.dir):
        if f == 'go' or f.startswith(('ready_', 'res_')):
            os.remove(os.path.join(a.dir, f))
    cpus = cpulist(a.cpus)
    per = len(cpus) // a.n
    gpus = a.gpus.split(',') if a.gpus else []
    procs = []
    for i in range(a.n):
        mine = ','.join(map(str, cpus[i * per:(i + 1) * per]))
        env = dict(os.environ)
        cmd = ['taskset', '-c', mine, sys.executable, '-m', 'perf.node',
               a.kind, a.system, '--dir', a.dir, '--tag', str(i),
               '--seconds', str(a.seconds)] + a.rest
        if a.kind == 'jax':
            env['JAX_PLATFORMS'] = a.platform
            if a.platform == 'cuda':
                env['CUDA_VISIBLE_DEVICES'] = gpus[i]
            cmd += ['--seed', str(i)]
        procs.append(subprocess.Popen(
            cmd, env=env, stdout=open(f'{a.dir}/w_{i}.out', 'w'),
            stderr=open(f'{a.dir}/w_{i}.err', 'w')))
    t = time.time()
    while True:
        ready = sum(os.path.exists(f'{a.dir}/ready_{i}') for i in range(a.n))
        dead = [i for i, p in enumerate(procs) if p.poll() not in (None, 0)]
        if dead:
            for p in procs:
                p.kill()
            sys.exit(f'workers {dead} died; see {a.dir}/w_*.err')
        if ready == a.n:
            break
        time.sleep(0.5)
    t_ready = time.time() - t
    with open(f'{a.dir}/go.tmp', 'w') as f:
        f.write(repr(time.time() + 3.))
    os.rename(f'{a.dir}/go.tmp', f'{a.dir}/go')
    for p in procs:
        p.wait()
    res = [json.load(open(f'{a.dir}/res_{i}.json')) for i in range(a.n)]
    r = np.array([x['steps_per_s'] for x in res])
    out = dict(kind=a.kind, system=a.system, n=a.n, seconds=a.seconds,
               cpus=a.cpus, gpus=a.gpus, args=' '.join(a.rest),
               all_ready_after_s=t_ready, steps_per_s=float(r.sum()),
               worker_min=float(r.min()), worker_median=float(np.median(r)),
               worker_max=float(r.max()),
               total_s_max=max(x['total_s'] for x in res),
               call_s_max=max(x['call_s_max'] for x in res),
               start_spread_s=max(x['start'] for x in res)
               - min(x['start'] for x in res))
    for k in ('imp', 'imp_build', 'check_level', 'score_moved', 'burn',
              'sampler', 'device', 'chains', 'sweeps', 'steps_per_sweep',
              'jax'):
        if k in res[0]:
            out[k] = res[0][k]
    if a.kind == 'cpp':
        out['rel_diff_vs_full_max'] = max(x['rel_diff_vs_full'] for x in res)
    else:
        out['acceptance'] = float(np.mean([x['acceptance'] for x in res]))
    print('NODE ' + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('drive', help='start the workers and sum their speeds;'
                       ' arguments after -- go to every worker')
    d.add_argument('dir')
    d.add_argument('--kind', choices=('cpp', 'jax'), required=True)
    d.add_argument('--system', required=True)
    d.add_argument('--n', type=int, required=True, help='workers')
    d.add_argument('--cpus', required=True,
                   help='CPU list, e.g. 0-63, split evenly over the workers')
    d.add_argument('--gpus', default='', help='one CUDA device per jax worker')
    d.add_argument('--platform', default='cuda', choices=('cuda', 'cpu'))
    d.add_argument('--seconds', type=float, default=60.)
    for kind in ('cpp', 'jax'):
        w = sub.add_parser(kind,
                           help=f'one {kind} worker (the driver runs it)')
        w.add_argument('system')
        w.add_argument('--dir', required=True)
        w.add_argument('--tag', required=True)
        w.add_argument('--seconds', type=float, required=True)
        w.add_argument('--burn', type=int, default=0)
        if kind == 'cpp':
            w.add_argument('--steps', type=int, required=True)
            w.add_argument('--moved', default='False')
        else:
            w.add_argument('--chains', type=int, required=True)
            w.add_argument('--sweeps', type=int, required=True)
            w.add_argument('--seed', type=int, default=0)
            w.add_argument('--core', action='store_true')
    a, rest = ap.parse_known_args()
    if a.cmd == 'drive':
        a.rest = [x for x in rest if x != '--']
        drive(a)
    elif rest:
        ap.error(f'unknown arguments {rest}')
    else:
        worker(a)


if __name__ == '__main__':
    main()
