"""Score against frame over all runs of run_npc_jax.py: median and 10-90%
band, and the lowest score at each frame, with a reference line (by default
the mean score of the published yeast models; --published 0 for none).

  python plot_scores.py runs/poc_g0 runs/poc_g1 --out scores.png
  python plot_scores.py runs/y256_one --stage one_stage --out scores.png

With --rex K > 1 the trace holds every replica; this plots all of them.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--stage', default='rex2', help='rex1, rex2 or one_stage')
ap.add_argument('--published', type=float, default=2149.0,
                help='reference line: mean total score of the published '
                     'yeast models (0: none)')
a = ap.parse_args()

S = []
for r in a.runs:
    with open(os.path.join(r, a.stage, 'trace.jsonl')) as fh:
        S.append(np.array([json.loads(x)['score'] for x in fh]))
n = min(len(s) for s in S)
S = np.concatenate([s[:n] for s in S], axis=1)        # frames x runs
S = np.where(np.isfinite(S), S, np.nan)
f = np.arange(1, n + 1)
q10, q50, q90 = np.nanpercentile(S, [10, 50, 90], axis=1)
best = np.nanmin(S, 1)

# system and MC steps per frame from the run's summary
system, nsteps = 'NPC', None
try:
    with open(os.path.join(a.runs[0], 'summary.jsonl')) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get('name', f"rex{d['stage']}") == a.stage:
                system, nsteps = d['system'].capitalize(), d['nsteps']
except OSError:
    pass

BLUE, BLUE_DARK, BAND = '#2a78d6', '#104281', '#cde2fb'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2})
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')
ax.fill_between(f, q10, q90, color=BAND, lw=0, label='10-90% of runs')
ax.plot(f, q50, color=BLUE, lw=2, label='median run')
ax.plot(f, best, color=BLUE_DARK, lw=1.2, label='lowest score at each frame')
lo = np.nanmin(S) * 0.95
if a.published:
    lo = min(lo, a.published * 0.95)
    ax.axhline(a.published, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(n * 0.02, a.published * 0.99,
            f'published models (after their score filter), mean '
            f'{a.published:,.0f}', color=INK2, ha='left', va='top',
            fontsize=9)
hi = np.nanmax(q90[:max(1, n // 50)]) * 1.05
ax.set_yscale('log')
ax.set_ylim(lo, hi)
ticks = [t for t in (2000, 2200, 2500, 3000, 4000, 5000, 7000, 1e4, 3e4,
                     1e5, 1e6, 1e7, 1e8, 1e9) if lo <= t <= hi]
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda y, _: f'{y:,.0f}'))
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlim(1, n)
stage = {'one_stage': 'one-stage', 'rex1': 'REX1', 'rex2': 'REX2'}.get(
    a.stage, a.stage)
ax.set_xlabel(f'{stage} frame'
              + (f' ({nsteps:,} MC steps each)' if nsteps else ''))
ax.set_ylabel('total score')
ax.set_title(f'{system} basket, {S.shape[1]:,} GPU runs: score during '
             f'{stage} sampling', color=INK, loc='left', fontsize=11)
ax.grid(True, which='major', color=GRID, lw=0.8)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='upper right', labelcolor=INK2, fontsize=9)
for y, dy in ((q50[-1], 5), (best[-1], -5)):
    ax.annotate(f'{y:,.0f}', (n, y), xytext=(4, dy),
                textcoords='offset points', va='center', color=INK,
                fontsize=9, annotation_clip=False)
fig.tight_layout()
fig.savefig(a.out, facecolor=fig.get_facecolor())
print(f'{S.shape[1]} runs, {n} frames: median {q50[-1]:.1f}, '
      f'10% {q10[-1]:.1f}, best {best[-1]:.1f}')
