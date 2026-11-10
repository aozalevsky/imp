"""The slide's two model pictures, with their scale and size (models.json).

  python models.py NPC_assembly.cif yeast_npcbasket_c8.cif

NPC_assembly.cif is in pdb_dev_deposit/ of the PM_NPC_Assembly repository
(its model 6 is the mature snapshot), yeast_npcbasket_c8.cif in
Yeast_NPC_Basket/ of the NPC_Basket repository (model 1, all 8 spokes).
Writes model_npc.png and model_basket.png (beads as shaded discs, colored by
protein, cropped to their content) and models.json with, per model:
- px_per_A: image pixels per A, horizontally (both views are tilted about
  the horizontal axis, so vertical distances are foreshortened);
- mass_MDa: the mass of the model's chains at full length, from their
  sequences. A chain written as poly-Ala (the basket's generic Mlp1/Mlp2)
  counts 110 Da per residue. The files' formula_weight is not used: it
  comes out about 16% higher than the sequences give;
- diameter_A: across the symmetry axis, bead radii included (99th
  percentile); chains and beads: counts.

Needs numpy, matplotlib and Pillow.
"""
import collections
import json
import os
import re
import shlex
import sys

import numpy as np

# average residue masses (Da), water removed
RESIDUE = dict(
    ALA=71.08, ARG=156.19, ASN=114.10, ASP=115.09, CYS=103.14, GLN=128.13,
    GLU=129.12, GLY=57.05, HIS=137.14, ILE=113.16, LEU=113.16, LYS=128.17,
    MET=131.19, PHE=147.18, PRO=97.12, SER=87.08, THR=101.10, TRP=186.21,
    TYR=163.18, VAL=99.13)
# picture: file, the mmCIF model, tilt of the view (degrees from the top
# view), and a color per protein (first pattern that matches its name)
PICTURES = dict(
    npc=dict(image='model_npc.png', model='6', tilt=30., colors=(
        (r'_yc_', '#5B8FD0'), (r'Nup155', '#6AA84F'),
        (r'Nup93|Nup205', '#E08A2E'), (r'Nup188', '#8E5BB5'),
        (r'p54|p58|p62', '#A82032'))),
    basket=dict(image='model_basket.png', model='1', tilt=62., colors=(
        (r'^mlp', '#E3A72F'), (r'^nup60', '#B5312A'), (r'^nup2$', '#6AA84F'),
        (r'^nup1$', '#8E5BB5'), (r'.', '#9DB7D9'))))


def loops(path, want):
    """{category: (columns, rows)} for the mmCIF loops named in `want`
       (rows on one line each, as python-ihm writes them)"""
    out = {}
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            if not line.startswith('loop_'):
                continue
            cols = []
            line = next(lines)
            while line.startswith('_'):
                cols.append(line.strip())
                line = next(lines)
            cat = cols[0].split('.')[0]
            if cat not in want:
                continue
            rows = []
            while line.strip() and not line.startswith('#'):
                quoted = "'" in line or '"' in line
                rows.append(shlex.split(line) if quoted else line.split())
                line = next(lines, '#')
            out[cat] = ([c.split('.')[1] for c in cols], rows)
    return out


def read(path, model):
    """the beads of one model: positions, radii, protein names; and the
       residues of every chain the model has"""
    cat = loops(path, {'_entity', '_struct_asym', '_entity_poly_seq',
                       '_ihm_sphere_obj_site'})
    c, rows = cat['_entity']
    name = {r[c.index('id')]: r[c.index('pdbx_description')] for r in rows}
    c, rows = cat['_struct_asym']
    entity = {r[c.index('id')]: r[c.index('entity_id')] for r in rows}
    c, rows = cat['_entity_poly_seq']
    seq = collections.defaultdict(list)
    for r in rows:
        seq[r[c.index('entity_id')]].append(r[c.index('mon_id')])
    c, rows = cat['_ihm_sphere_obj_site']
    ix = {k: c.index(k) for k in ('asym_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                                  'object_radius', 'model_id')}
    rows = [r for r in rows if r[ix['model_id']] == model]
    chains = sorted({r[ix['asym_id']] for r in rows})
    xyz = ('Cartn_x', 'Cartn_y', 'Cartn_z')
    return dict(
        xyz=np.array([[float(r[ix[k]]) for k in xyz] for r in rows]),
        rad=np.array([float(r[ix['object_radius']]) for r in rows]),
        name=[name[entity[r[ix['asym_id']]]] for r in rows],
        chains=[seq[entity[a]] for a in chains])


def mass(chains):
    """Da, of the chains at full length"""
    def one(s):
        if len(s) > 100 and set(s) == {'ALA'}:       # a generic chain
            return 110. * len(s) + 18.02
        return sum(RESIDUE.get(m, 110.) for m in s) + 18.02
    return sum(one(s) for s in chains)


def diameter(xyz, rad):
    c = xyz.mean(0)
    r = np.hypot(xyz[:, 0] - c[0], xyz[:, 1] - c[1]) + rad
    return 2 * np.percentile(r, 99)


def render(xyz, rad, colors, out, tilt, px=1400, fog=0.45, pad=0.03):
    """discs sorted by depth, the far ones faded; image pixels per A"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = np.radians(-tilt)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                    [0, np.sin(a), np.cos(a)]])
    x, y, d = ((xyz - xyz.mean(0)) @ rot.T).T          # d: toward the viewer
    w = max(np.ptp(x), np.ptp(y)) * (1 + 2 * pad)
    fig = plt.figure(figsize=(px / 100, px / 100 * (
        np.ptp(y) * (1 + 2 * pad) / w)), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    h = w * fig.get_figheight() / fig.get_figwidth()
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    o = np.argsort(d)
    t = (d - d.min()) / np.ptp(d)                      # 0 far .. 1 near
    c = np.array(colors)
    c = c + (1 - c) * (fog * (1 - t))[:, None]
    s = (2 * rad * px / w * 72 / 100) ** 2             # points squared
    ax.scatter(x[o], y[o], s=s[o], c=c[o],
               edgecolors=np.clip(c[o] * 0.72, 0, 1), linewidths=0.25)
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return px / w


def main():
    from matplotlib.colors import to_rgb
    from PIL import Image
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for (key, p), path in zip(PICTURES.items(), sys.argv[1:3]):
        m = read(path, p['model'])
        colors = [to_rgb(next(c for pat, c in p['colors'] + (('', '#BBBBBB'),)
                              if re.search(pat, n))) for n in m['name']]
        image = os.path.join(here, p['image'])
        scale = render(m['xyz'], m['rad'], colors, image, p['tilt'])
        im = Image.open(image)
        im.crop(im.getbbox()).save(image)
        out[key] = dict(image=p['image'], px_per_A=scale,
                        mass_MDa=mass(m['chains']) / 1e6,
                        diameter_A=diameter(m['xyz'], m['rad']),
                        chains=len(m['chains']), beads=len(m['rad']))
        print(key, out[key])
    with open(os.path.join(here, 'models.json'), 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')


if __name__ == '__main__':
    main()
