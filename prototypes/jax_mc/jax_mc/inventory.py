"""What an IMP system is made of: restraint tree (types, weights, score
objects, containers), movers and score states. Used to decide what the
extractors must support.

  python -m jax_mc.inventory polii|npc|pm|mzid
"""
import collections
import sys

from . import imp_hooks, sources


def restraint_tree(rs, out, depth=0, w=1.0, counts=None):
    import IMP
    for r in rs:
        d = imp_hooks.derived(r)
        cls = type(d).__name__
        ww = w * r.get_weight()
        extra = ''
        if hasattr(d, 'get_score_object'):
            try:
                so = d.get_score_object()
                try:
                    so = so.get_derived_object()
                except Exception:
                    pass
                extra += f' score={type(so).__name__}'
            except Exception:
                pass
        if hasattr(d, 'get_container'):
            try:
                extra += f' container={d.get_container().get_type_name()}'
            except Exception:
                pass
        if isinstance(d, IMP.RestraintSet):
            out.append(f'{"  " * depth}{cls} "{r.get_name()}" w={ww:g} '
                       f'({d.get_number_of_restraints()})')
            restraint_tree([d.get_restraint(i)
                            for i in range(d.get_number_of_restraints())],
                           out, depth + 1, ww, counts)
        elif cls == 'LogWrapper':
            sub = [x.get_derived_object() for x in d.restraints]
            kinds = collections.Counter(type(x).__name__ for x in sub)
            out.append(f'{"  " * depth}LogWrapper "{r.get_name()}" w={ww:g} '
                       f'{dict(kinds)}')
            counts[('LogWrapper', tuple(sorted(kinds)))] += 1
        else:
            counts[(cls, extra.strip())] += 1
            if counts[(cls, extra.strip())] <= 2:
                out.append(f'{"  " * depth}{cls} "{r.get_name()}" w={ww:g}'
                           f'{extra}')


def movers_of(movers):
    import IMP.core
    flat = []
    for mv in movers:
        d = mv.get_derived_object() if hasattr(mv, 'get_derived_object') \
            else mv
        if isinstance(d, IMP.core.SerialMover):
            flat += movers_of(list(d.get_movers()))
        else:
            flat.append(d)
    return flat


def main(name):
    src = sources.load(name)
    m = src.model
    out, counts = [], collections.Counter()
    restraint_tree(src.restraints, out, counts=counts)
    print(f'== {name}: mode {src.mode}, kT {src.kT}')
    print('\n'.join(out))
    print('-- leaf restraint kinds:')
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {v:6d}  {k}')
    mv = movers_of(src.movers)
    print('-- movers:', dict(collections.Counter(type(x).__name__
                                                 for x in mv)), 'total',
          len(mv))
    ss = collections.Counter()
    for s in m.get_ordered_score_states():
        try:
            s = s.get_derived_object()
        except Exception:
            pass
        ss[type(s).__name__] += 1
    print('-- score states:', dict(ss))
    import IMP
    sf = IMP.core.RestraintsScoringFunction(src.restraints)
    print('-- total score (C++):', sf.evaluate(False))


if __name__ == '__main__':
    main(sys.argv[1])
