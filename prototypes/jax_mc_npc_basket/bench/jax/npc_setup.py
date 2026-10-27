"""Build an NPC_Basket system in-process and stop right before REX2 runs.

The modeling script runs unchanged (via runpy).
REX1 runs in C++ for `frames1` frames. Then REX2's execute_macro is
intercepted, and the script's globals (model, hierarchy, dof, restraint
wrappers) plus the macro are returned. That gives a realistic REX2 scoring
function and configuration for JAX experiments.

    from npc_setup import build
    S = build('yeast', frames1=2, out='/tmp/x')
    S.model, S.g['dof'], S.g['gem'], S.rex2 ...
"""
import os
import random
import runpy
import shutil
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)

SYSTEMS = {
    'yeast': dict(wd=os.path.join(ROOT, 'Yeast_NPC_Basket'),
                  script=os.path.join(BENCH, 'patched', 'Yeast_src',
                                      'modeling_basket.py'),
                  data='data',
                  gmm=os.path.join(BENCH, 'gmm_cache_yeast')),
    'mouse': dict(wd=os.path.join(BENCH, 'mouse_wd'),
                  script=os.path.join(BENCH, 'patched', 'Mouse_src',
                                      'modeling_mbasket.py'),
                  data=os.path.join(ROOT, 'Mouse_NPC_Basket', 'data'),
                  gmm=os.path.join(BENCH, 'gmm_cache_mouse')),
}


class _Stop(Exception):
    pass


def build(system, frames1=2, out=None, seed=1, quiet=True):
    import numpy
    import IMP
    import IMP.pmi.macros
    import IMP.pmi.tools

    cfg = SYSTEMS[system]
    out = os.path.abspath(out or os.path.join(
        HERE, 'scratch', f'{system}_{os.getpid()}'))
    IMP.random_number_generator.seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for fn in os.listdir(cfg['gmm']):
        shutil.copy(os.path.join(cfg['gmm'], fn), out)

    S = types.SimpleNamespace(system=system, out=out)
    orig_init = IMP.pmi.macros.ReplicaExchange.__init__
    orig_exec = IMP.pmi.macros.ReplicaExchange.execute_macro
    nstage = [0]

    def rex_init(self, model, **kw):
        nstage[0] += 1
        self._stage = nstage[0]
        kw['number_of_frames'] = frames1 if self._stage == 1 else 1
        S.rex_kw = dict(kw)
        orig_init(self, model, **kw)

    def rex_exec(self):
        if self._stage == 1:
            return orig_exec(self)
        S.rex2 = self
        S.g = sys._getframe(1).f_globals
        raise _Stop()

    IMP.pmi.macros.ReplicaExchange.__init__ = rex_init
    IMP.pmi.macros.ReplicaExchange.execute_macro = rex_exec
    cwd = os.getcwd()
    argv = sys.argv
    devnull = open(os.devnull, 'w') if quiet else None
    old_stdout = sys.stdout
    try:
        os.chdir(cfg['wd'])
        sys.argv = [cfg['script'], '-d', cfg['data'], '-o', out]
        sys.path.insert(0, os.path.dirname(cfg['script']))
        if quiet:
            sys.stdout = devnull
        runpy.run_path(cfg['script'], run_name='__main__')
        raise RuntimeError('script finished without reaching REX2')
    except _Stop:
        pass
    finally:
        sys.stdout = old_stdout
        sys.argv = argv
        os.chdir(cwd)
        IMP.pmi.macros.ReplicaExchange.__init__ = orig_init
        IMP.pmi.macros.ReplicaExchange.execute_macro = orig_exec
        if devnull:
            devnull.close()
    S.model = S.rex2.model
    S.rs = IMP.pmi.tools.get_restraint_set(S.model)
    S.movers = S.g['dof'].get_movers()
    return S
