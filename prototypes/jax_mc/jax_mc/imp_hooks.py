"""Small hooks into IMP's Python layer.

Some restraints don't expose their parameters to Python (for example the
particle lists and cutoffs of the GaussianEMRestraints), and bayesianem's
wrapper can't cast a Restraint back to its class. install() wraps the
constructors so that every such restraint built afterwards is recorded
with its arguments, keyed by its C++ pointer. Call it before building the
system.
"""
import math

RECORDED = {}       # int(this) -> (python proxy, class name, args, kwargs)

_CLASSES = [('IMP.isd', 'GaussianEMRestraint'),
            ('IMP.bayesianem', 'GaussianEMRestraint')]


def install():
    import importlib
    for mod, cls_name in _CLASSES:
        try:
            cls = getattr(importlib.import_module(mod), cls_name)
        except (ImportError, AttributeError):
            continue
        if getattr(cls, '_jax_mc_recorded', False):
            continue
        orig = cls.__init__
        tag = f'{mod}.{cls_name}'

        def __init__(self, *args, _orig=orig, _tag=tag, **kw):
            _orig(self, *args, **kw)
            RECORDED[int(self.this)] = (self, _tag, args, kw)
        cls.__init__ = __init__
        cls._jax_mc_recorded = True


def derived(r):
    """The restraint as its most derived Python class"""
    rec = RECORDED.get(int(r.this))
    if rec is not None:
        return rec[0]
    try:
        return r.get_derived_object()
    except (NameError, AttributeError):
        return r


def recorded(r):
    """(class tag, constructor args, kwargs) of a recorded restraint"""
    rec = RECORDED.get(int(r.this))
    if rec is None:
        raise KeyError(f'{r.get_name()} was built before imp_hooks.install()')
    return rec[1:]


def info_floats(r):
    ri = r.get_static_info()
    return {ri.get_float_key(i): ri.get_float_value(i)
            for i in range(ri.get_number_of_float())}


def input_particles(r):
    """Indexes of the particles among r's inputs, in input order"""
    import IMP
    out = []
    for o in r.get_inputs():
        try:
            out.append(int(IMP.Particle.get_from(o).get_index()))
        except (ValueError, TypeError):
            pass
    return out


def pis(seq):
    """ints from IMP particles, decorators or indexes"""
    out = []
    for p in seq:
        if hasattr(p, 'get_particle_index'):
            out.append(int(p.get_particle_index()))
        elif hasattr(p, 'get_index'):
            out.append(int(p.get_index()))
        else:
            out.append(int(p))
    return out


FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi
