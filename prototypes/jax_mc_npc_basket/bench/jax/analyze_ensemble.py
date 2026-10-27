"""Compare C++ and JAX chain ensembles written by ensemble.py."""
import json
import sys

import numpy as np
from scipy import stats


def load(fn):
    return [json.loads(line) for line in open(fn)]


def main(cpp_fn, jax_fn):
    C, Jx = load(cpp_fn), load(jax_fn)
    sc = np.array([r['scores'] for r in C])     # (chains, frames+1)
    sj = np.array([r['scores'] for r in Jx])
    nf = min(sc.shape[1], sj.shape[1])
    print(f'C++ chains {len(sc)}, JAX chains {len(sj)}, frames {nf - 1}')
    print('| frame | C++ mean (sd) | JAX mean (sd) | Mann-Whitney p | '
          'KS p |')
    print('|---|---|---|---|---|')
    nsig = 0
    for f in range(nf):
        a, b = sc[:, f], sj[:, f]
        if f == 0:
            p_mw = p_ks = float('nan')
        else:
            p_mw = stats.mannwhitneyu(a, b).pvalue
            p_ks = stats.ks_2samp(a, b).pvalue
            nsig += p_mw < 0.05
        print(f'| {f} | {a.mean():.1f} ({a.std():.1f}) | {b.mean():.1f} '
              f'({b.std():.1f}) | {p_mw:.3f} | {p_ks:.3f} |')
    print(f'\nframes with Mann-Whitney p < 0.05: {nsig} of {nf - 1}')
    half = nf // 2
    ma, mb = sc[:, half:].mean(1), sj[:, half:].mean(1)
    print(f'per-chain mean score over frames {half}-{nf - 1}: '
          f'C++ {ma.mean():.1f} +- {ma.std():.1f}, JAX {mb.mean():.1f} +- '
          f'{mb.std():.1f}; MW p {stats.mannwhitneyu(ma, mb).pvalue:.3f}, '
          f'KS p {stats.ks_2samp(ma, mb).pvalue:.3f}')
    keys = sorted(set(C[0]['acc']) | set(Jx[0]['acc']))
    print('\nacceptance by mover type (mean over chains):')
    for k in keys:
        ac = np.mean([r['acc'].get(k, np.nan) for r in C])
        aj = np.mean([r['acc'].get(k, np.nan) for r in Jx])
        print(f'  {k:18s} C++ {ac:.3f}  JAX {aj:.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
