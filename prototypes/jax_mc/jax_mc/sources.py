"""Build the IMP systems we test on, and hand back what jax_mc needs: the
model, the restraints, the movers and how the movers are applied.

Each loader calls an existing builder (from the prototype directories),
so the system is exactly the one the C++ protocol samples. Directories come
from environment variables:

  JAX_MC_POLII   prototypes/jax_mc_rnapolii (plus IMP_TUTORIAL)
  JAX_MC_NPC     NPC_Basket/bench/jax
  JAX_MC_PM      PM_NPC_Assembly bench (pm_setup.py)
  JAX_MC_MZID    mzIdentML tutorial bench (common.py)
"""
import os
import sys
from dataclasses import dataclass, field

from . import imp_hooks


@dataclass
class Source:
    name: str
    model: object
    restraints: list            # top-level IMP restraints (sets allowed)
    movers: list                # IMP movers, SerialMover order
    # mode 'serial': one mover per MC step; 'all': every mover every step
    mode: str
    kT: float = 1.0
    extra: dict = field(default_factory=dict)
    ladder: list = None         # the protocol's REX temperatures, if any
    hier: object = None         # root hierarchy (for RMF output)


def _dir(var):
    d = os.environ.get(var)
    if not d or not os.path.isdir(d):
        raise RuntimeError(f'set {var} to the system\'s directory')
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def polii(**kw):
    """RNA Pol II stalk tutorial (imp_tutorial rnapolii/modeling.py)"""
    _dir('JAX_MC_POLII')
    imp_hooks.install()
    import build_system
    S = build_system.build(**kw)
    return Source('polii', S.m, list(S.rs.get_restraints()), list(S.movers),
                  'serial', extra=dict(S=S), hier=S.root_hier)


def npc_basket(system='yeast', frames1=2, **kw):
    """NPC basket, REX2 stage (NPC_Basket); frames1 = REX1 frames run in
       C++ first, as the protocol does"""
    _dir('JAX_MC_NPC')
    imp_hooks.install()
    import imp_jax_npc
    imp_jax_npc.install()
    from npc_setup import build
    S = build(system, frames1=frames1, **kw)
    return Source('npc_' + system, S.model, [S.rs], list(S.movers),
                  'serial', extra=dict(S=S), hier=S.rex2.root_hier)


def mature_npc(time='mature', temp_index=0, **kw):
    """Mature NPC refinement (PM_NPC_Assembly): an IMP.core.MonteCarlo with
       one RigidBodyMover per subcomplex, all applied every step"""
    _dir('JAX_MC_PM')
    imp_hooks.install()
    import IMP.core
    from pm_setup import build
    S = build(time, **kw)
    movers = [IMP.core.RigidBodyMover(S.m, rb, S.mc_max_trans, S.mc_max_rot)
              for rb in S.rigbods]
    return Source('mature_npc', S.m, list(S.rs), movers, 'all',
                  kT=float(S.temps[temp_index]), extra=dict(S=S),
                  ladder=[float(t) for t in S.temps], hier=S.hc)


def mzid(seed=1):
    """PMI mzIdentML tutorial (the notebook's run_modeling())"""
    _dir('JAX_MC_MZID')
    imp_hooks.install()
    import common
    m, smp, root = common.build_system(seed=seed, return_root=True)
    mc = smp.mc
    movers = [mc.get_mover(i) for i in range(mc.get_number_of_movers())]
    return Source('mzid', m, list(mc.get_scoring_function()
                                  .create_restraints()),
                  movers, 'serial', kT=float(mc.get_kt()),
                  extra=dict(smp=smp), hier=root)


LOADERS = dict(polii=polii, npc=npc_basket, pm=mature_npc, mzid=mzid)


def load(name, **kw):
    return LOADERS[name](**kw)
