"""Run Level-D exactness checks (`jax_mc.check.decisions`) for all recipes
R1-R7 and `templates.few_bodies_beads.sampler:factory` on RNA Polymerase II.

Usage (on a GPU node):
  python -m recipes.check_all [--chains-small 1024] [--chains-large 8192]
      [--ref-small 256] [--ref-large 128] [--sweeps 5]

Checks both chain-count regimes (`<= 2,048` and `> 4,096` chains) at `kT = 1`
and `kT = 10` (`~50%` acceptance) in `float64`, asserting 0 differing
Metropolis decisions against `jax_mc.engine.Engine`. The template also runs
at every chain count where its code changes path (--template-chains: block
sizes, X0 reuse and the R6 precompute switch at 2,048, 4,096, 8,192, 16,384
and 32,768 chains).

R5 has no check of its own: its functions are used by the template and
checked there (the core's EMIsd has its own closed-form covariances).
"""
import argparse
import time

from jax_mc import check, sources, system, xla_defaults

SPECS = [
    ('R1 (static EV grid)', 'recipes.r1_static_ev_grid:factory', 'rotate'),
    ('R2 (unrolled sweep)', 'recipes.r2_unrolled_sweep:factory', 'rotate'),
    ('R3 (blocked beads)', 'recipes.r3_blocked_beads:factory', 'rotate'),
    ('R4 (EM body cache)', 'recipes.r4_em_body_cache:factory', 'rotate'),
    ('R6 (sweep precompute)', 'recipes.r6_sweep_precompute:factory', 'rotate'),
    ('R7 (sliced RNG)', 'recipes.r7_sliced_rng:factory', 'rotate'),
    ('few_bodies_beads (rotate)',
     'templates.few_bodies_beads.sampler:factory', 'rotate'),
    ('few_bodies_beads (frozen)',
     'templates.few_bodies_beads.sampler:factory', 'frozen'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--chains-small', type=int, default=1024)
    ap.add_argument('--chains-large', type=int, default=8192)
    ap.add_argument('--ref-small', type=int, default=256)
    ap.add_argument('--ref-large', type=int, default=128)
    ap.add_argument('--sweeps', type=int, default=5)
    ap.add_argument('--kT', default='1,10')
    ap.add_argument('--template-chains', default='4096,16384,32768,65536')
    a = ap.parse_args()

    xla_defaults()
    import jax
    jax.config.update('jax_enable_x64', True)

    kTs = tuple(float(x) for x in a.kT.split(','))
    regimes = [
        (a.chains_small, a.ref_small),
        (a.chains_large, a.ref_large),
    ]

    t_start = time.time()
    src = sources.load('polii')
    systems = {
        'rotate': system.extract(src, cov_mode='rotate'),
        'frozen': system.extract(src, cov_mode='frozen'),
    }

    summary = []
    for label, spec, cov in SPECS:
        fn = check._factory(spec)
        s = systems[cov]
        for chains, ref_chains in regimes:
            print(f'\n=== {label} [{spec}] | cov={cov} | '
                  f'chains={chains} (ref={ref_chains}) ===', flush=True)
            res = check.decisions(
                s, sweeps=a.sweeps, chains=chains, kTs=kTs,
                ref_chains=ref_chains, sampler=fn)
            for r in res:
                assert r['ndiff'] == 0, (
                    f'FAIL {label} chains={chains} kT={r["kT"]}: '
                    f'{r["ndiff"]}/{r["n"]} differing decisions')
                summary.append((label, cov, chains, r['kT'], r['ndiff'],
                                r['n'], r.get('dE_track', 0.),
                                r.get('dX', 0.)))

    fn = check._factory('templates.few_bodies_beads.sampler:factory')
    for chains in [int(x) for x in a.template_chains.split(',') if x]:
        label = 'few_bodies_beads (frozen)'
        print(f'\n=== {label} | chains={chains} (ref=64) ===', flush=True)
        for r in check.decisions(systems['frozen'], sweeps=3, chains=chains,
                                 kTs=kTs, ref_chains=64, sampler=fn):
            assert r['ndiff'] == 0, (
                f'FAIL {label} chains={chains} kT={r["kT"]}: '
                f'{r["ndiff"]}/{r["n"]} differing decisions')
            summary.append((label, 'frozen', chains, r['kT'], r['ndiff'],
                            r['n'], r.get('dE_track', 0.), r.get('dX', 0.)))

    elapsed = time.time() - t_start
    print(f'\n=== ALL CHECKS PASSED in {elapsed:.1f}s ===')
    print(f'{"Target":28s} {"cov":>6s} {"chains":>6s} {"kT":>4s} '
          f'{"diff":>10s} {"max dE_track":>13s} {"max dX":>10s}')
    for label, cov, chains, kT, ndiff, n, dE, dX in summary:
        print(f'{label:28s} {cov:>6s} {chains:6d} {kT:4g} '
              f'{f"{ndiff}/{n}":>10s} {dE:13.2e} {dX:10.2e}')


if __name__ == '__main__':
    main()
