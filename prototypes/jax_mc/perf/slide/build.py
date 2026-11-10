"""A slide of the node-against-node speeds (perf/node_results.jsonl).

  python build.py [--results ../node_results.jsonl] [--out slide.pptx]

One 1920 x 1080 pt slide, in the style of the deck it was made for (Arial
blue title, Helvetica grey text, rounded grey panels with red headings).
Per system: the model's picture with a scale bar and its mass (models.py
makes both, into models.json), the GPU node's speed over the CPU node's, and
four speeds on a log axis. Every element is a native text box, shape or
picture, so the slide stays editable; positions are in points. The texts are
in texts.json.

Needs python-pptx. For a PDF and a PNG of it:
  soffice --headless --convert-to pdf slide.pptx
  pdftoppm -png -singlefile -scale-to-x 3840 -scale-to-y 2160 slide.pdf slide
"""
import argparse
import json
import os
from math import floor, log10

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

BLUE = RGBColor(0x00, 0x76, 0xBA)        # the deck's titles
RED = RGBColor(0xB5, 0x17, 0x00)         # the deck's panel headings
GREY = RGBColor(0x5E, 0x5E, 0x5E)        # the deck's body text and outlines
CPU_ONE, CPU_NODE = RGBColor(0xC4, 0xC4, 0xC4), RGBColor(0x7F, 0x7F, 0x7F)
GPU_ONE, GPU_NODE = RGBColor(0x8C, 0xC1, 0xE2), BLUE
GRID = RGBColor(0xD6, 0xD6, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = 'Arial', 'Helvetica'


def P(v):
    return Emu(int(round(v * 12700)))


def text(slide, x, y, w, h, runs, size, color=GREY, font=BODY, align='l',
         anchor='t', name=None, bold=False, italic=False, spacing=None):
    """runs: a string, or a list of paragraphs, each a string or a list of
       (text, {bold, italic, color, size, font}) runs"""
    tb = slide.shapes.add_textbox(P(x), P(y), P(w), P(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = dict(t=MSO_ANCHOR.TOP, m=MSO_ANCHOR.MIDDLE,
                              b=MSO_ANCHOR.BOTTOM)[anchor]
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = dict(l=PP_ALIGN.LEFT, c=PP_ALIGN.CENTER,
                           r=PP_ALIGN.RIGHT)[align]
        if spacing:
            p.line_spacing = spacing
        for r in ([(para, {})] if isinstance(para, str) else para):
            t, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run()
            run.text = t
            f = run.font
            f.name = o.get('font', font)
            f.size = Pt(o.get('size', size))
            f.bold = o.get('bold', bold)
            f.italic = o.get('italic', italic)
            f.color.rgb = o.get('color', color)
            # the same face for every script, so Keynote doesn't substitute
            rpr = run._r.get_or_add_rPr()
            for tag in ('a:ea', 'a:cs'):
                e = etree.SubElement(rpr, qn(tag))
                e.set('typeface', o.get('font', font))
    return tb


def shadow(shape, blur=5., dist=2., alpha=0.5):
    sp = shape._element.spPr
    eff = etree.SubElement(sp, qn('a:effectLst'))
    sh = etree.SubElement(eff, qn('a:outerShdw'))
    sh.set('blurRad', str(int(blur * 12700)))
    sh.set('dist', str(int(dist * 12700)))
    sh.set('dir', '5400000')                 # straight down, as in the deck
    sh.set('algn', 'ctr')
    sh.set('rotWithShape', '0')
    c = etree.SubElement(sh, qn('a:srgbClr'))
    c.set('val', '000000')
    a = etree.SubElement(c, qn('a:alpha'))
    a.set('val', str(int(alpha * 100000)))


def panel(slide, x, y, w, h, radius=60., name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, P(x), P(y), P(w),
                               P(h))
    s.adjustments[0] = radius / min(w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = WHITE
    s.line.color.rgb = GREY
    s.line.width = Pt(1)
    shadow(s)
    if name:
        s.name = name
    return s


def bar(slide, x, y, w, h, color, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, P(x), P(y), P(w), P(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def line(slide, x1, y1, x2, y2, color=GRID, width=0.75, name=None):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, P(x1), P(y1),
                                   P(x2), P(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if name:
        c.name = name
    return c


def sig(v, n=2):
    if v == 0:
        return 0.
    d = n - 1 - floor(log10(abs(v)))
    return round(v, d)


def words(v):
    """108,582 -> '109 thousand'; 4.3e3 -> '4.3 thousand'; 180.6 -> '181'"""
    for scale, word in ((1e9, 'billion'), (1e6, 'million'), (1e3, 'thousand')):
        if v >= scale:
            x = v / scale
            return f'{x:.0f} {word}' if x >= 9.95 else f'{x:.1f} {word}'
    return f'{v:.0f}'


def times(v):
    return f'{sig(v):,.0f}×'


def duration(minutes):
    """time on the GPU node for a month on the CPU node"""
    if minutes < 55:
        m = sig(minutes)
        return f'{m:.0f} minutes' if m >= 10 else f'{m:.1f} minutes'
    if minutes < 66:
        return '1 hour'
    h = sig(minutes / 60)
    return f'{h:.0f} hours' if h >= 10 else f'{h:g} hours'


LO, HI = 1, 9                 # the speed axis: 10 to 1e9 MC steps/s, log
TICKS = ((3, '1 thousand'), (6, '1 million'), (9, '1 billion'))


BOX = (370., 330.)             # room for a model's picture, points


def common_scale(systems):
    """points per A at which every picture fits its box: one scale for all,
       so the scale bars are the same length"""
    return min(min(BOX[0] * d['px_per_A'] / d['image_size'][0],
                   BOX[1] * d['px_per_A'] / d['image_size'][1])
               for d in systems)


def system_panel(slide, x, y, w, h, d, tag, scale):
    panel(slide, x, y, w, h, name=f'{tag} panel')
    text(slide, x + 20, y + 18, w - 40, 44, d['title'], 28, RED, HEAD, 'c',
         'm', name=f'{tag} heading')
    # the model, a scale bar with the model's mass, and what is sampled
    iw, ih = d['image_size']
    box_w, box_h = BOX
    k = scale / d['px_per_A']                   # points per image pixel
    pw, ph = iw * k, ih * k
    slide.shapes.add_picture(d['image'], P(x + 36 + (box_w - pw) / 2),
                             P(y + 66 + (box_h - ph) / 2), P(pw),
                             P(ph)).name = f'{tag} model'
    bar_len = d['scale_A'] * scale
    label = f"{d['scale_A']:,.0f} Å  ·  {d['mass']}"
    lw = 9.4 * len(label)                       # about, at 20 pt
    sx = x + 36 + (box_w - (bar_len + 10 + lw)) / 2
    sy = y + 66 + box_h + 18
    line(slide, sx, sy, sx + bar_len, sy, GREY, 2., name=f'{tag} scale bar')
    for ex in (sx, sx + bar_len):
        line(slide, ex, sy - 6, ex, sy + 6, GREY, 2.,
             name=f'{tag} scale bar end')
    text(slide, sx + bar_len + 10, sy - 14, lw + 40, 28, label, 20, GREY,
         BODY, 'l', 'm', name=f'{tag} scale and mass')
    text(slide, x + 16, sy + 20, box_w + 40, 28, d['size'], 20, GREY, BODY,
         'c', name=f'{tag} model size')
    # the headline: node against node
    speedup = d['gpu_node'] / d['cpu_node']
    hx, hw = x + 425, w - 445
    text(slide, hx, y + 106, hw, 136, times(speedup), 118, BLUE, HEAD, 'c',
         'm', name=f'{tag} speed-up')
    text(slide, hx, y + 246, hw, 38, 'more sampling per node', 28, GREY,
         BODY, 'c', name=f'{tag} speed-up caption')
    text(slide, hx, y + 306, hw, 70,
         [[('1 month', {}), (' on the CPU node', {})],
          [('≈ ', {}), (duration(30 * 24 * 60 / speedup),
                        {'color': RED}), (' on the GPU node', {})]],
         24, GREY, BODY, 'c', name=f'{tag} time')
    # the speeds on a log axis
    bx, bw, bh, pitch = x + 286, 370., 30., 42.
    top = sy + 62

    def pos(v):
        return bw * (log10(v) - LO) / (HI - LO)
    rows = (('1 CPU thread', d['cpu1'], CPU_ONE),
            (d['cpu_label'], d['cpu_node'], CPU_NODE),
            ('1 GPU', d['gpu1'], GPU_ONE),
            (d['gpu_label'], d['gpu_node'], GPU_NODE))
    bottom = top + pitch * len(rows) - (pitch - bh)
    # the axis sits clear of the last bar, its ticks point away from it
    major = [e for e, _ in TICKS]
    ay = bottom + 14
    for e in range(LO, HI + 1):
        gx = bx + pos(10 ** e)
        line(slide, gx, ay, gx, ay + (9 if e in major else 5), GREY, 0.75,
             name=f'{tag} tick 1e{e}')
    line(slide, bx, ay, bx + bw, ay, GREY, 0.75, name=f'{tag} axis')
    for e, label in TICKS:
        text(slide, bx + pos(10 ** e) - 70, ay + 12, 140, 24, label, 18,
             GREY, BODY, 'c', name=f'{tag} tick label 1e{e}')
    for i, (label, v, color) in enumerate(rows):
        ry = top + i * pitch
        node = i % 2 == 1
        text(slide, x + 16, ry, bx - x - 28, bh, label, 22, GREY, BODY, 'r',
             'm', name=f'{tag} {label} label')
        bar(slide, bx, ry, pos(v), bh, color, name=f'{tag} {label} bar')
        text(slide, bx + pos(v) + 10, ry, 200, bh, words(v), 24, GREY, BODY,
             'l', 'm', bold=node, name=f'{tag} {label} value')
    text(slide, bx - 40, ay + 40, bw + 80, 26,
         'Monte Carlo steps per second (log scale)', 20, GREY, BODY, 'c',
         name=f'{tag} axis title')
    # the source of the model
    text(slide, x + 20, y + h - 46, w - 40, 28, d['citation'], 20, GREY,
         HEAD, 'c', name=f'{tag} citation')


def plain(slide):
    """drop the theme's shape style (it adds a shadow and an outline)"""
    for s in slide.shapes:
        for st in s._element.findall(qn('p:style')):
            s._element.remove(st)


def build(numbers, out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = P(1920), P(1080)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    text(slide, 110, 26, 1700, 96, numbers['title'], 46, BLUE, HEAD, 'c',
         'm', name='Title')
    text(slide, 110, 130, 1700, 40, numbers['subtitle'], 26, GREY, BODY,
         'c', 'm', name='Subtitle')
    pw, ph, py = 870., 780., 196.
    scale = common_scale([numbers['basket'], numbers['npc']])
    for k, sysname in enumerate(('basket', 'npc')):
        system_panel(slide, 70 + k * (pw + 40), py, pw, ph, numbers[sysname],
                     sysname, scale)
    text(slide, 80, py + ph + 24, 1760, 34, numbers['footnote'], 22, GREY,
         BODY, 'c', name='Footnote')
    plain(slide)
    prs.core_properties.title = numbers['title']
    prs.core_properties.author = ''
    prs.save(out)


def numbers(results, here):
    """what build() draws: the texts, the models and the measured speeds"""
    from PIL import Image
    texts = json.load(open(os.path.join(here, 'texts.json')))
    models = json.load(open(os.path.join(here, 'models.json')))
    recs = [json.loads(line) for line in open(results)]

    def speed(system, **kw):
        """the last record that matches"""
        r = [d for d in recs if d['system'] == system
             and all(d.get(k) == v for k, v in kw.items())]
        assert r, (system, kw)
        return r[-1]['steps_per_s']
    out = {k: texts[k] for k in ('title', 'subtitle', 'footnote')}
    print(f"{'':8s}{'1 thread':>10s}{'CPU node':>12s}{'1 GPU':>14s}"
          f"{'GPU node':>14s}{'GPU/CPU':>10s}   MC steps/s")
    for key, d in texts['systems'].items():
        m, s = models[key], d['record']
        # C++: the fast build (checks compiled out)
        cpp = dict(kind='cpp', check_level=0)
        image = os.path.join(here, m['image'])
        cit = d['citation']
        out[key] = dict(
            title=d['title'], size=d['size'], image=image,
            image_size=Image.open(image).size, px_per_A=m['px_per_A'],
            scale_A=texts['scale_A'], mass=f"model: {m['mass_MDa']:.0f} MDa",
            citation=[[(cit[0], {}), (cit[1], {'italic': True}),
                       (cit[2], {})]],
            cpu_label=texts['cpu_label'], gpu_label=texts['gpu_label'],
            cpu1=speed(s, n=1, **cpp),
            cpu_node=speed(s, n=texts['cpu_processes'], **cpp),
            gpu1=speed(s, kind='jax', n=1),
            gpu_node=speed(s, kind='jax', n=texts['gpu_processes']))
        o = out[key]
        print(f"{key:8s}{o['cpu1']:10.1f}{o['cpu_node']:12.1f}"
              f"{o['gpu1']:14.0f}{o['gpu_node']:14.0f}"
              f"{o['gpu_node'] / o['cpu_node']:10.1f}")
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results',
                    default=os.path.join(here, '..', 'node_results.jsonl'))
    ap.add_argument('--out', default=os.path.join(here, 'slide.pptx'))
    a = ap.parse_args()
    build(numbers(a.results, here), a.out)


if __name__ == '__main__':
    main()
