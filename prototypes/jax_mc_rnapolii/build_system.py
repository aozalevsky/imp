"""Build the imp_tutorial RNA Pol II stalk system in-process, exactly as
rnapolii/modeling/modeling.py does, and stop right before sampling.

The tutorial script is executed unchanged except that `mc1.execute_macro()`
is removed (the same trick as the tutorial's own test_pickle), in a scratch
working directory with `../data` pointing at the tutorial's data. `--test`
is passed to it; that only shortens the tutorial's own number of frames,
which is not used here.

Set IMP_TUTORIAL to a clone of https://github.com/salilab/imp_tutorial
(tested at commit 8058229 of main).

    from build_system import build
    S = build(seed=1)
    S.m, S.root_hier, S.dof, S.mc1, S.outputobjects, S.rs (restraint set)
"""
import os
import random
import sys
import tempfile
import types

# constructor arguments of restraints whose parameters have no Python
# getters, keyed by int(restraint.this)
REGISTRY = {}


def _tutorial_paths():
    repo = os.environ.get('IMP_TUTORIAL')
    if not repo:
        raise RuntimeError('set IMP_TUTORIAL to a clone of '
                           'https://github.com/salilab/imp_tutorial')
    return (os.path.join(repo, 'rnapolii', 'modeling', 'modeling.py'),
            os.path.join(repo, 'rnapolii', 'data'))


def _record(cls, tag):
    if getattr(cls, '_tut_recorded', False):
        return
    orig = cls.__init__

    def __init__(self, *args, **kw):
        orig(self, *args, **kw)
        REGISTRY[int(self.this)] = (tag, args, kw)
    cls.__init__ = __init__
    cls._tut_recorded = True


def build(seed=1, workdir=None, test=True, quiet=True):
    import numpy as np
    import IMP
    import IMP.isd
    import IMP.pmi.tools

    script, data = _tutorial_paths()
    random.seed(seed)
    np.random.seed(seed)
    IMP.random_number_generator.seed(seed)
    _record(IMP.isd.GaussianEMRestraint, 'isd_gem')
    workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix='rnapolii_'))
    os.makedirs(os.path.join(workdir, 'modeling'), exist_ok=True)
    link = os.path.join(workdir, 'data')
    if not os.path.exists(link):
        os.symlink(data, link)
    src = open(script).read()
    assert src.count('mc1.execute_macro()') == 1
    src = src.replace('mc1.execute_macro()', '')
    cwd, argv = os.getcwd(), sys.argv
    os.chdir(os.path.join(workdir, 'modeling'))
    sys.argv = ['modeling.py'] + (['--test'] if test else [])
    g = {'__name__': '__tutorial__'}
    stdout = sys.stdout
    try:
        if quiet:
            sys.stdout = open(os.devnull, 'w')
        exec(compile(src, script, 'exec'), g)
    finally:
        if quiet:
            sys.stdout.close()
        sys.stdout = stdout
        os.chdir(cwd)
        sys.argv = argv
    S = types.SimpleNamespace(**{k: g[k] for k in (
        'm', 'root_hier', 'dof', 'mc1', 'outputobjects', 'num_mc_steps',
        'num_frames', 'gemt', 'xl1', 'xl2', 'ev')})
    S.workdir = workdir
    S.rs = IMP.pmi.tools.get_restraint_set(S.m)
    S.movers = S.dof.get_movers()
    return S
