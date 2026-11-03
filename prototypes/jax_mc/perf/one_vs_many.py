"""MC speed of the jax_mc systems: C++ IMP with one chain, against jax_mc
with one chain on one CPU core, one chain on one GPU, and as many chains as
pay off on one GPU.

  python -m perf.one_vs_many cpp SYSTEM [--steps N] [--reps 5] [--burn B]
      [--modes False,True]
  python -m perf.one_vs_many jax SYSTEM --chains C [--sweeps S] [--reps 5]
      [--burn B] [--core]
  python -m perf.one_vs_many plot results.jsonl OUTDIR

Each measurement prints one line 'RESULT {json}' (perf/run.sh appends them
to results.jsonl); plot writes OUTDIR/speed.csv and OUTDIR/speed.png.

cpp: IMP.core.MonteCarlo on the system's own restraints and movers
(jax_mc.sources builds them with the protocol's code), at the protocol's kT:
a SerialMover over the movers for PMI's serial systems (one mover per MC
step), every mover at every step for the mature NPC. Timed with
score_moved off and on (--modes); after each call the MC energy is compared
with a full rescore, and the plot uses the faster mode that stays exact.
--burn runs that many steps untimed first, with score_moved off. Which IMP
build runs (release or fast) is up to the environment.

jax: the system's template if it has one (SAMPLERS), else the core engine
(engine.make: Engine, or CollectiveEngine for the mature NPC); --core takes
the core engine anyway. float32, the protocol's kT. The state is donated
between calls, as in production.
--burn runs that many sweeps untimed first (the mature NPC's collective
steps rebuild their neighbour lists constantly from the starting structure).

MC steps/s = mover proposals plus Metropolis tests per second; the mature
NPC's step moves all 112 bodies, as IMP's does. Speeds are medians of
--reps timed calls.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SAMPLERS = dict(polii='templates.few_bodies_beads.sampler:factory',
                pm='templates.collective.sampler:factory',
                npc=None,                        # the reference engine
                mzid=None)
NAMES = dict(mzid='mzIdentML', npc='NPC basket', pm='mature NPC',
             polii='Pol II')
ORDER = ('mzid', 'npc', 'pm', 'polii')
CPU1 = 'JAX core, 1 CPU core'
GPU1 = 'JAX core, 1 GPU, 1 chain'
CORE = 'JAX core, 1 GPU, best chain count'
TCPU1 = 'JAX template, 1 CPU core'
TEMPLATE = 'JAX template, 1 GPU, best chain count'
BARS = ['C++ release', 'C++ fast', CPU1, TCPU1,      # one CPU core
        GPU1, CORE, TEMPLATE]                       # one GPU
BLOCKS = (4, 3)


def result(d):
    print('RESULT ' + json.dumps(d), flush=True)


def cpp(a):
    from jax_mc import sources
    import IMP
    import IMP.core
    IMP.set_log_level(IMP.SILENT)
    src = sources.load(a.system)
    sf = IMP.core.RestraintsScoringFunction(src.restraints)
    out = dict(kind='cpp', system=a.system, imp=IMP.get_module_version(),
               imp_build=os.path.basename(os.path.dirname(os.path.dirname(
                   IMP.__file__))),            # lib/<build>/IMP
               check_level=int(IMP.get_check_level()), mode=src.mode,
               kT=src.kT, steps=a.steps, reps=a.reps, burn=a.burn)
    modes = [m == 'True' for m in a.modes.split(',')]

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
    for moved in modes:
        mc = make(moved)
        mc.optimize(max(a.steps // 10, 1))
        rates, diff = [], 0.
        for _ in range(a.reps):
            t = time.perf_counter()
            mc.optimize(a.steps)
            rates.append(a.steps / (time.perf_counter() - t))
            e_mc, e_full = mc.get_last_accepted_energy(), sf.evaluate(False)
            diff = max(diff, abs(e_mc - e_full) / max(abs(e_full), 1.))
        out[f'score_moved_{moved}'] = dict(
            steps_per_s=float(np.median(rates)), steps_per_s_max=max(rates),
            max_rel_diff_vs_full=diff)
    result(out)


def jax_run(a):
    from jax_mc import xla_defaults
    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', False)
    from jax_mc import check, engine, sources, system
    s = system.extract(sources.load(a.system))
    spec = SAMPLERS[a.system]
    core = a.core or spec is None
    smp = engine.make(s) if core else check._factory(spec)(s)
    n = smp.steps_per_sweep
    run = jax.jit(smp.run, static_argnums=(1, 2), donate_argnums=0)
    t0 = time.perf_counter()
    st = smp.init(a.chains, 0)
    st = jax.block_until_ready(run(st, a.sweeps))
    first = time.perf_counter() - t0
    for _ in range(a.burn // a.sweeps):
        st = run(st, a.sweeps)
    st = jax.block_until_ready(st)
    acc0 = np.asarray(st['n_acc']).sum()
    rates = []
    for _ in range(a.reps):
        t = time.perf_counter()
        st = jax.block_until_ready(run(st, a.sweeps))
        rates.append(a.chains * n * a.sweeps / (time.perf_counter() - t))
    acc = (float(np.asarray(st['n_acc']).sum() - acc0)
           / (a.chains * n * a.sweeps * a.reps))
    dev = jax.devices()[0]
    result(dict(kind='jax', system=a.system, sampler=type(smp).__name__,
                core=core,
                platform=dev.platform, device=dev.device_kind,
                chains=a.chains, sweeps=a.sweeps, steps_per_sweep=n,
                reps=a.reps, burn=a.burn, first_call_s=first,
                acceptance=acc, steps_per_s=float(np.median(rates)),
                steps_per_s_max=max(rates), jax=jax.__version__,
                xla_flags=os.environ.get('XLA_FLAGS', '')))


def rows(path):
    """one row per system and bar, from results.jsonl (the last
       measurement of each configuration wins)"""
    cfg = {}
    for line in open(path):
        d = json.loads(line)
        if d['kind'] == 'cpp':
            fast = 'fast' in d['imp_build']
            ok = [m for m in ('score_moved_True', 'score_moved_False')
                  if m in d and d[m]['max_rel_diff_vs_full'] < 1e-6]
            m = max(ok, key=lambda m: d[m]['steps_per_s'])
            cfg[(d['system'], 'fast' if fast else 'release')] = dict(
                system=d['system'],
                bar='C++ fast' if fast else 'C++ release',
                steps_per_s=d[m]['steps_per_s'], chains=1,
                detail=f'score_moved {"on" if m.endswith("True") else "off"}; '
                       f'IMP {d["imp"]}; '
                       f'check level {d["check_level"]}')
        else:
            core = d.get('core', SAMPLERS[d['system']] is None)
            if d['platform'] == 'cpu':
                key, bar = ('cpu', CPU1) if core else ('tcpu', TCPU1)
            elif d['chains'] == 1:
                if not core:                # the GPU one-chain bar: core
                    continue
                key, bar = 'gpu1', GPU1
            else:
                key = f'gpu{d["chains"]}' + ('core' if core else '')
                bar = CORE if core else TEMPLATE
            cfg[(d['system'], key)] = dict(
                system=d['system'], bar=bar, steps_per_s=d['steps_per_s'],
                chains=d['chains'],
                detail=f'{d["sampler"]}; {d["device"]}; float32; '
                       f'acceptance {d["acceptance"]:.2g}')
    out = []
    for sysn in ORDER:
        mine = [v for k, v in cfg.items() if k[0] == sysn]
        keep = [v for v in mine if v['bar'] not in (CORE, TEMPLATE)]
        for bar in (CORE, TEMPLATE):
            many = [v for v in mine if v['bar'] == bar]
            if many:
                keep.append(max(many, key=lambda v: v['steps_per_s']))
        out += sorted(keep, key=lambda v: BARS.index(v['bar']))
    return out


def plot(a):
    import csv
    os.makedirs(a.outdir, exist_ok=True)
    R = rows(a.results)
    with open(os.path.join(a.outdir, 'speed.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['system', 'configuration', 'chains', 'mc_steps_per_s',
                    'x_vs_cpp_release', 'detail'])
        base = {r['system']: r['steps_per_s'] for r in R
                if r['bar'] == 'C++ release'}
        for r in R:
            b = base.get(r['system'])
            w.writerow([NAMES[r['system']], r['bar'], r['chains'],
                        f'{r["steps_per_s"]:.6g}',
                        f'{r["steps_per_s"] / b:.4g}' if b else '',
                        r['detail']])
    meta = [json.loads(x) for x in open(a.results)]
    imp = sorted({d['imp'] for d in meta if d['kind'] == 'cpp'})
    jx = sorted({d['jax'] for d in meta if d['kind'] == 'jax'})
    gpu = sorted({d['device'] for d in meta
                  if d['kind'] == 'jax' and d['platform'] != 'cpu'})
    off = [NAMES[r['system']] for r in R if r['bar'] == 'C++ release'
           and r['detail'].startswith('score_moved off')]
    note = (f'C++: IMP {", ".join(imp)}, one chain on one core, score_moved '
            f'on' + (f' (off for the {", ".join(off)})' if off else '') +
            f'. JAX {", ".join(jx)}, float32, on one CPU core or one '
            f'{", ".join(gpu)}. JAX core: the generic engine. JAX template: '
            'templates/few_bodies_beads (Pol II) and templates/collective '
            '(mature NPC); mzIdentML and the NPC basket have none yet. A '
            'mature NPC step moves all 112 bodies, in C++ and in JAX; its '
            'core engine scores all 520 M bead pairs per step (no neighbour '
            'lists), C++ and the template only close pairs.')
    sub = {}
    for sysn in ORDER:
        best = {r['bar']: r['chains'] for r in R
                if r['system'] == sysn and r['bar'] in (CORE, TEMPLATE)}
        if TEMPLATE in best:
            sub[sysn] = (f'best chain counts: core {best[CORE]:,}, '
                         f'template {best[TEMPLATE]:,}')
        elif CORE in best:
            sub[sysn] = f'best chain count: {best[CORE]:,}'
    from perf.figure import draw
    draw(R, NAMES, ORDER, BARS, os.path.join(a.outdir, 'speed.png'),
         title='Monte Carlo speed: C++ IMP vs jax_mc', note=note, sub=sub,
         blocks=BLOCKS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='what', required=True)
    c = sub.add_parser('cpp')
    c.add_argument('system')
    c.add_argument('--steps', type=int, default=10000)
    c.add_argument('--reps', type=int, default=5)
    c.add_argument('--burn', type=int, default=0)
    c.add_argument('--modes', default='False,True',
                   help='score_moved settings to time')
    j = sub.add_parser('jax')
    j.add_argument('system')
    j.add_argument('--chains', type=int, required=True)
    j.add_argument('--sweeps', type=int, default=100)
    j.add_argument('--reps', type=int, default=5)
    j.add_argument('--burn', type=int, default=0)
    j.add_argument('--core', action='store_true',
                   help='the core engine even if the system has a template')
    p = sub.add_parser('plot')
    p.add_argument('results')
    p.add_argument('outdir')
    a = ap.parse_args()
    if a.what != 'plot':
        from jax_mc import pin_hash_seed
        pin_hash_seed()
    dict(cpp=cpp, jax=jax_run, plot=plot)[a.what](a)


if __name__ == '__main__':
    sys.exit(main())
