"""speed.png: grouped columns, one group per system, log scale (the bars
span six orders of magnitude). Colors: the first slots of a
colorblind-checked categorical order, used in that order; values are
labelled on every column, and speed.csv is the table view."""
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#6250d6']
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def fmt(v):
    for div, unit in ((1e9, ' bn'), (1e6, ' M'), (1e3, 'k')):
        if v >= div:
            x = v / div
            return (f'{x:.3g}' if x < 100 else f'{x:.0f}') + unit
    return f'{v:.0f}' if v >= 10 else f'{v:.2g}'


def draw(rows, names, order, bars, path, title=None, note=None, sub=None,
         blocks=None):
    """rows: dicts with system, bar, steps_per_s, chains; bars in order
       (one color each, in that order); sub: system -> a line under its
       name (the chain counts, which don't fit on the columns); blocks:
       sizes of runs of bars set apart within a group (e.g. CPU, GPU)"""
    systems = [s for s in order if any(r['system'] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    blocks = blocks or (len(bars),)
    assert sum(blocks) == len(bars)
    sep = 0.5                            # extra space between blocks, in w
    w = 0.82 / (len(bars) + sep * (len(blocks) - 1))
    gap = 0.012                          # air between columns
    off, k0 = [], 0
    for j, b in enumerate(blocks):
        off += [(k0 + i + sep * j) * w for i in range(b)]
        k0 += b
    off = [o - (off[0] + off[-1]) / 2 for o in off]
    val = {(r['system'], r['bar']): r['steps_per_s'] for r in rows}
    for k, bar in enumerate(bars):
        xs = [i + off[k] for i, s in enumerate(systems) if (s, bar) in val]
        ys = [val[s, bar] for s in systems if (s, bar) in val]
        ax.bar(xs, ys, width=w - gap, color=COLORS[k], label=bar, zorder=3)
    # value labels. A long label next to another long one at about the
    # same height goes one line up; one wider than its column is aligned
    # away from a taller neighbour.
    line = 2.3                           # a label's height, as a factor
    for i, s in enumerate(systems):
        prev = None
        for k, bar in enumerate(bars):
            if (s, bar) not in val:
                prev = None
                continue
            lab, y = fmt(val[s, bar]), val[s, bar] * 1.12
            if prev and len(lab) + len(prev[1]) > 10 and \
                    prev[0] / line < y < prev[0] * line:
                y = prev[0] * line
            x, ha = i + off[k], 'center'
            near = [val.get((s, bars[j]), 0.) if 0 <= j < len(bars) else 0.
                    for j in (k - 1, k + 1)]
            if len(lab) >= 6 and near[1] > y:
                x, ha = x + (w - gap) / 2, 'right'
            elif len(lab) >= 6 and near[0] > y:
                x, ha = x - (w - gap) / 2, 'left'
            ax.text(x, y, lab, ha=ha, va='bottom', fontsize=7, color=INK,
                    zorder=4)
            prev = (y, lab)
    ax.set_yscale('log')
    lo = min(r['steps_per_s'] for r in rows)
    hi = max(r['steps_per_s'] for r in rows)
    ax.set_ylim(10 ** math.floor(math.log10(lo)), hi * 12)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([names[s] for s in systems], fontsize=10, color=INK)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for i, s in enumerate(systems):
        if sub and s in sub:
            ax.text(i, -0.075, sub[s], transform=tr, ha='center', va='top',
                    fontsize=7.5, color=INK2)
    ax.set_ylabel('MC steps/s (log scale)', color=INK2, fontsize=9)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.tick_params(axis='y', colors=INK2, labelsize=8)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis='y', which='minor', visible=False)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(INK2)
    leg = ax.legend(loc='lower left', ncol=len(bars), frameon=False,
                    fontsize=8, labelcolor=INK, bbox_to_anchor=(0, 1.0),
                    handlelength=1.2, columnspacing=1.0, handletextpad=0.4,
                    borderaxespad=0.3)
    leg.set_in_layout(False)             # one row; keep the axes wide
    if title:
        fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=INK, y=0.985)
    if note:
        note = textwrap.fill(note, 175)
        fig.text(0.01, 0.012, note, ha='left', va='bottom', fontsize=7,
                 color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075 if note else 0, 1, 0.95))
    fig.subplots_adjust(top=0.89)        # legend right under the title
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
