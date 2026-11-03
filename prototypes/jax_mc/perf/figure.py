"""speed.png: grouped columns, one group per system, log scale (the bars
span six orders of magnitude). Colors: the first five slots of a
colorblind-checked categorical order, used in that order; values are
labelled on every column, and speed.csv is the table view."""
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

BARS = ['C++ release', 'C++ fast', 'JAX, 1 CPU core', 'JAX, 1 GPU, 1 chain',
        'JAX, 1 GPU, best chain count']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def fmt(v):
    for div, unit in ((1e9, ' bn'), (1e6, ' M'), (1e3, 'k')):
        if v >= div:
            x = v / div
            return (f'{x:.3g}' if x < 100 else f'{x:.0f}') + unit
    return f'{v:.0f}'


def draw(rows, names, order, path, title=None, note=None):
    systems = [s for s in order if any(r['system'] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    w, gap = 0.15, 0.012                 # column width, air between columns
    for k, bar in enumerate(BARS):
        xs, ys, labels = [], [], []
        for i, s in enumerate(systems):
            r = [r for r in rows if r['system'] == s and r['bar'] == bar]
            if not r:
                continue
            r = r[0]
            xs.append(i + (k - 2) * w)
            ys.append(r['steps_per_s'])
            lab = fmt(r['steps_per_s'])
            if r['chains'] > 1:
                lab += f'\n{r["chains"]:,} ch.'
            labels.append(lab)
        ax.bar(xs, ys, width=w - gap, color=COLORS[k], label=bar, zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.text(x, y * 1.12, lab, ha='center', va='bottom', fontsize=7,
                    color=INK, zorder=4)
    ax.set_yscale('log')
    lo = min(r['steps_per_s'] for r in rows)
    hi = max(r['steps_per_s'] for r in rows)
    ax.set_ylim(10 ** math.floor(math.log10(lo)), hi * 12)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([names[s] for s in systems], fontsize=10, color=INK)
    ax.set_ylabel('MC steps/s (log scale)', color=INK2, fontsize=9)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.tick_params(axis='y', colors=INK2, labelsize=8)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis='y', which='minor', visible=False)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(INK2)
    ax.legend(loc='lower left', ncol=5, frameon=False, fontsize=8,
              labelcolor=INK, bbox_to_anchor=(0, 1.0), handlelength=1.2,
              columnspacing=1.2, borderaxespad=0.3)
    if title:
        fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=INK, y=0.985)
    if note:
        note = textwrap.fill(note, 175)
        fig.text(0.01, 0.012, note, ha='left', va='bottom', fontsize=7,
                 color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075 if note else 0, 1, 0.95))
    fig.subplots_adjust(top=0.875)       # legend right under the title
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
