"""Prototype fixes for IMP's JAX sampling path (IMP nightly 9bd20857cd).

Each fix is a monkeypatch of IMP's Python layer that can be switched on
independently, so that its effect can be measured on its own:

    import imp_jax_patches
    imp_jax_patches.apply(['sync', 'rb_dedupe', 'xl_vec', ...])
    # or: imp_jax_patches.apply(imp_jax_patches.ALL)

Apply the patches *before* the JAX optimizer is created (i.e. before
ReplicaExchange / MonteCarlo.set_use_jax).

Patches:
  sync        One jax.device_get per frame instead of per-element reads;
              add per-frame *deltas* of MC/mover statistics (not cumulative
              totals); also copy mover-modified Float attributes (e.g. ISD
              nuisances sigma/psi) back into the IMP Model.
  total_score Report Total_Score in PMI stat files from the MC state's score
              instead of jit-compiling a second copy of the scoring function.
  rb_dedupe   Each _RigidBodyPositionConstraint updates only its own body
              (upstream, every one of them updates *all* bodies).
  rb_batched  Update the members of all (non-nested) rigid bodies with a
              single gather/scatter (implies rb_dedupe).
  xl_vec      Vectorized JAX implementation of an ISD LogWrapper over
              CrossLinkMSRestraints (no per-crosslink Python loop/lax.cond).
  pair_vec    Score all PairRestraints in a RestraintSet that share one
              PairScore with a single vectorized call.
  unroll=N    Unroll the MC fori_loop N times.
  fused       Fused branch-free MC step on compact arrays (imp_jax_fused.py);
              use together with xl_vec, pair_vec and total_score.
  fused_unroll=N  Unroll the fused step's fori_loop N times (4 on GPU).
  pdb_cache   (PMI, not JAX-specific) Cache the static part of PMI's
              best-scoring PDB output and refresh only the coordinates.
  fast_update (IMP kernel, not JAX-specific) Model.update() with a cached
              score-state update order (3 ms -> ~0.02 ms here).
  pdb_fast    (PMI) With pdb_cache: build each PDB line once with IMP's
              get_pdb_string and only splice in new coordinates.
  stat_fast   (PMI) CrossLinkingMassSpectrometryRestraint.get_output
              computes cross-link distances with numpy instead of two XYZ
              decorators per cross-link per frame.
  rex=N       Batched replica exchange with N replicas on the fused step
              (imp_jax_rex.py); PMI output follows the lowest-temperature
              replica. rex_tmax=X overrides the macro's maximum
              temperature (put it after rex=N).
  rex_ladders=M  With rex=N: run M independent N-replica ladders on one
              device instead of PMI's sampling loop, writing one sampcon-style
              run_<i>/ per ladder. rex_thin=k saves an RMF frame every k
              frames (default 10); rex_block=F runs F frames per device call
              (default 50).
"""
import functools
import math

import numpy as np

ALL = ['sync', 'total_score', 'rb_batched', 'xl_vec', 'pair_vec']
MAX = ['fused', 'total_score', 'xl_vec', 'pair_vec', 'pdb_cache',
       'fast_update', 'pdb_fast', 'stat_fast']
_applied = set()


def expand(spec):
    """Patch list from a comma-separated spec; 'all' and 'max' expand to the
       predefined sets, 'none' or '' means no patches."""
    out = []
    for tok in (spec or '').split(','):
        tok = tok.strip()
        if tok in ('', 'none'):
            continue
        for name in {'all': ALL, 'max': MAX}.get(tok, [tok]):
            if name not in out:
                out.append(name)
    return out


def apply(names):
    for name in names:
        if name in _applied:
            continue
        if name.startswith('unroll='):
            _patch_unroll(int(name.split('=')[1]))
        elif name == 'fused':
            import imp_jax_fused
            imp_jax_fused.apply()
        elif name.split('=')[0] in ('rex_ladders', 'rex_thin', 'rex_block'):
            import imp_jax_rex
            k, v = name.split('=')
            imp_jax_rex._settings[k[4:]] = int(v)
        elif name.startswith('rex_tmax='):
            import imp_jax_rex
            imp_jax_rex.configure(imp_jax_rex._settings['n'] or 1,
                                  tmax=float(name.split('=')[1]))
        elif name.startswith('rex='):
            import imp_jax_rex
            imp_jax_rex.configure(int(name.split('=')[1]))
            imp_jax_rex.apply()
        elif name.startswith('fused_unroll='):
            import imp_jax_fused
            imp_jax_fused.UNROLL = int(name.split('=')[1])
        else:
            {'sync': _patch_sync,
             'total_score': _patch_total_score,
             'rb_dedupe': _patch_rb_dedupe,
             'rb_batched': _patch_rb_batched,
             'xl_vec': _patch_xl_vec,
             'pair_vec': _patch_pair_vec,
             'pdb_cache': _patch_pdb_cache,
             'fast_update': _patch_fast_update,
             'pdb_fast': _patch_pdb_fast,
             'stat_fast': _patch_stat_fast}[name]()
        _applied.add(name)


# ---------------------------------------------------------------------------
# sync: one host transfer per frame, delta statistics, nuisance sync
# ---------------------------------------------------------------------------
def _patch_sync():
    import jax
    import IMP
    import IMP.core._jax_util as cj
    from IMP.core import _jax_rigid

    def _float_keys(opt):
        # Model attributes that movers modify, other than xyz/rigid bodies
        return [k for k in opt._jax_info._keys if isinstance(k, IMP.FloatKey)]

    def _host_view(opt, jm):
        """The subset of the JAX state that must be copied back to IMP"""
        d = {'xyz': jm['xyz']}
        if 'rigid_bodies' in jm:
            rbs = jm['rigid_bodies']
            d['quaternion'] = rbs.quaternion
            d['lquaternion'] = rbs.lquaternion
            d['intcoord'] = rbs.intcoord
        for k in _float_keys(opt):
            d[k.get_string()] = jm[k.get_string()]
        return d

    def _write_imp(opt, m, host):
        m.get_spheres_numpy()[0][:] = host['xyz']
        if 'quaternion' in host:
            rbi = _jax_rigid._get_rigid_body_indexes(m)
            m.get_numpy(_jax_rigid._RB_QUAT_KEY)[rbi] = host['quaternion']
            nrbi = list(_jax_rigid._get_nested_rigid_body_indexes(m, rbi))
            if nrbi:
                m.get_numpy(_jax_rigid._RB_LQUAT_KEY)[nrbi] = \
                    host['lquaternion']
            is_rigid = m.get_numpy(_jax_rigid._RB_IS_RIGID_KEY)
            non_rigid = np.flatnonzero(is_rigid == 0)
            if non_rigid.size:
                m.get_internal_coordinates_numpy()[non_rigid] = \
                    host['intcoord'][non_rigid]
        if not hasattr(opt, '_float_key_indexes'):
            opt._float_key_indexes = {
                k.get_string(): np.asarray(
                    [int(p) for p in m.get_particle_indexes()
                     if m.get_has_attribute(k, IMP.ParticleIndex(int(p)))],
                    dtype=np.int64)
                for k in _float_keys(opt)}
        for k in _float_keys(opt):
            idx = opt._float_key_indexes[k.get_string()]
            m.get_numpy(k)[idx] = host[k.get_string()][idx]

    def optimize(self, mc_state):
        m = self.opt.get_model()
        use_best = self.opt.get_return_best()
        for _ in self._loop():
            mc_state = self.apply_func(mc_state)
            if self._imp_opt_states:
                # IMP OptimizerStates run between inner loops need the model
                _write_imp(self, m, jax.device_get(
                    _host_view(self, mc_state.jm)))
        jm = mc_state.best_jm if use_best else mc_state.jm
        stats = (mc_state.accepted_steps, mc_state.rejected_steps,
                 mc_state.downward_steps_taken, mc_state.upward_steps_taken,
                 mc_state.score, mc_state.best_score,
                 [getattr(ms, 'proposed_mover_steps', None)
                  for ms in mc_state.mover_state],
                 [getattr(ms, 'accepted_mover_steps', None)
                  for ms in mc_state.mover_state])
        # --- a single device->host transfer for everything we need ---
        host, stats = jax.device_get((_host_view(self, jm), stats))
        _write_imp(self, m, host)
        acc, rej, down, up, score, best, prop_i, acc_i = stats
        prev = getattr(self, '_prev_stats', None) or (
            0, 0, 0, 0, [None if p is None else np.zeros_like(p)
                         for p in prop_i],
            [None if a is None else np.zeros_like(a) for a in acc_i])
        p_acc, p_rej, p_down, p_up, p_prop_i, p_acc_i = prev
        self._prev_stats = (acc, rej, down, up, prop_i, acc_i)
        # IMP's C++ MonteCarlo reports per-optimize() counts
        self.opt.set_number_of_downward_steps(int(down - p_down))
        self.opt.set_number_of_upward_steps(int(up - p_up))
        self.opt.set_number_of_rejected_steps(int(rej - p_rej))
        self.opt.set_best_accepted_energy(float(best))
        self.opt.set_last_accepted_energy(float(score))
        nprop = int(acc + rej - p_acc - p_rej)
        for i, mover in enumerate(self._movers):
            mover.add_to_statistics(nprop, int(rej - p_rej))
            if prop_i[i] is not None:
                # SerialMover: per-submover deltas
                dp = prop_i[i] - p_prop_i[i]
                da = acc_i[i] - p_acc_i[i]
                for j, sub in enumerate(mover.get_movers()):
                    sub.add_to_statistics(int(dp[j]), int(dp[j] - da[j]))
        self._last_host_score = float(best if use_best else score)
        return self._last_host_score, mc_state

    cj._MCJAXOptimizer.optimize = optimize


# ---------------------------------------------------------------------------
# total_score: don't compile a second scoring function for stat output
# ---------------------------------------------------------------------------
def _patch_total_score():
    from collections import namedtuple
    import IMP.pmi.io
    import IMP.pmi.macros
    import IMP.pmi.tools

    _JAXDataScore = namedtuple('_JAXDataScore', ('model', 'space', 'score'))

    def _get_jax_data(self, sampler_mc):
        if self.use_jax:
            opt = sampler_mc._jax_optimizer
            score = getattr(opt, '_last_host_score', None)
            if score is None:
                score = sampler_mc._jax_state.score
            return _JAXDataScore(model=sampler_mc.get_jax_model(),
                                 space=self._jax_space, score=score)

    orig_call = IMP.pmi.io._TotalScorer.__call__

    def __call__(self, jax_data=None):
        if jax_data is not None and hasattr(jax_data, 'score'):
            return float(jax_data.score)
        return orig_call(self, jax_data)

    IMP.pmi.macros.ReplicaExchange._get_jax_data = _get_jax_data
    IMP.pmi.io._TotalScorer.__call__ = __call__


# ---------------------------------------------------------------------------
# rigid bodies
# ---------------------------------------------------------------------------
def _patch_rb_dedupe():
    import IMP.core
    from IMP.core import _jax_rigid

    def _get_jax(self):
        m = self.get_model()
        body = _jax_rigid._get_rigid_body_index(m, self.get_index())

        def apply_func(jm):
            return jm['rigid_bodies'].bodies[body].update_members(jm)
        return self._wrap_jax(apply_func, keys=['rigid_bodies'])

    IMP.core._RigidBodyPositionConstraint._get_jax = _get_jax


def _patch_rb_batched():
    import jax
    import jax.numpy as jnp
    import IMP.core
    from IMP.core import _jax_rigid
    from IMP.algebra._jax_util import _quaternion_to_rotation_matrix
    to_matrix = jax.vmap(_quaternion_to_rotation_matrix)
    _patch_rb_dedupe()      # fallback for models with nested bodies
    dedupe_get_jax = IMP.core._RigidBodyPositionConstraint._get_jax

    def _get_jax(self):
        m = self.get_model()
        rb_pis = list(_jax_rigid._get_rigid_body_indexes(m))
        if any(True for _ in _jax_rigid._get_nested_rigid_body_indexes(
                m, rb_pis)):
            return dedupe_get_jax(self)
        if int(self.get_index()) != int(rb_pis[0]):
            # the first body's constraint updates every body at once
            return self._wrap_jax(lambda jm: jm, keys=['rigid_bodies'])
        members, owner = [], []
        for i, pi in enumerate(rb_pis):
            mi = np.asarray(IMP.core.RigidBody(
                m, pi).get_member_particle_indexes(), dtype=np.int32)
            members.append(mi)
            owner.append(np.full(len(mi), i, dtype=np.int32))
        members = np.concatenate(members)
        owner = np.concatenate(owner)
        rb_pis = np.asarray(rb_pis, dtype=np.int32)

        def apply_func(jm):
            rbs = jm['rigid_bodies']
            rot = to_matrix(rbs.quaternion)             # (nb, 3, 3)
            trans = jm['xyz'][rb_pis]                   # (nb, 3)
            local = rbs.intcoord[members]               # (M, 3)
            # same convention as Transformation3DWithMatrix.get_transformed
            glob = jnp.einsum('mi,mij->mj', local, rot[owner]) + trans[owner]
            jm['xyz'] = jm['xyz'].at[members].set(glob)
            return jm
        return self._wrap_jax(apply_func, keys=['rigid_bodies'])

    IMP.core._RigidBodyPositionConstraint._get_jax = _get_jax


# ---------------------------------------------------------------------------
# xl_vec: vectorized CrossLinkMSRestraint set inside an ISD LogWrapper
# ---------------------------------------------------------------------------
def _xl_arrays(restraints):
    """Flatten all contributions of CrossLinkMSRestraints into arrays"""
    pis, sig, psi, rid = [], [], [], []
    for i, r in enumerate(restraints):
        for k in range(r.get_number_of_contributions()):
            pis.append([int(x) for x in r.get_contribution_particle_indexes(k)])
            sig.append([int(x) for x in r.get_contribution_sigma_indexes(k)])
            psi.append(int(r.get_contribution_psi_index(k)))
            rid.append(i)
    return (np.asarray(pis, np.int32), np.asarray(sig, np.int32),
            np.asarray(psi, np.int32), np.asarray(rid, np.int32))


def _make_xl_prob_func(restraints, space):
    """JAX function returning the per-restraint probability (or -log prob,
       following each restraint's get_log_prob()) times its weight, for a
       list of CrossLinkMSRestraints. Same math as
       IMP.isd.CrossLinkMSRestraint._get_jax, vectorized over all
       contributions."""
    import jax
    import jax.numpy as jnp
    r0 = restraints[0]
    length = r0.get_length()
    slope = r0.get_slope() if r0.get_has_slope() else None
    for r in restraints:
        if (r.get_is_length_variable() or r.get_length() != length
                or r.get_has_slope() != (slope is not None)
                or (slope is not None and r.get_slope() != slope)
                or r.get_log_prob() != r0.get_log_prob()):
            return None
    log_prob = r0.get_log_prob()
    pis, sig, psi, rid = _xl_arrays(restraints)
    nres = len(restraints)
    single = len(rid) == nres and bool(np.all(rid == np.arange(nres)))
    weights = np.asarray([r.get_weight() for r in restraints], np.float32)
    same = pis[:, 0] == pis[:, 1]
    half = length / 2.
    four3pi = 4.0 / 3.0 * math.pi
    xlvol = four3pi * half ** 3

    def sphere_cap(r1, r2, d):
        short = jnp.minimum(four3pi * r1 ** 3, four3pi * r2 ** 3)
        dsafe = jnp.where(d > 0, d, 1.0)
        mid = ((math.pi / 12 / dsafe * (r1 + r2 - d) * (r1 + r2 - d))
               * (d * d + 2 * d * r1 - 3 * r1 * r1 + 2 * d * r2
                  + 6 * r1 * r2 - 3 * r2 * r2))
        return jnp.where(d <= jnp.abs(r1 - r2), short,
                         jnp.where(d >= r1 + r2, 0., mid))

    def prob_func(jm):
        xyz, rad, nuis = jm['xyz'], jm['r'], jm['nuisance']
        dist = jnp.where(same, 36.0 / 35.0 * rad[pis[:, 0]],
                         space.distance(xyz[pis[:, 0]] - xyz[pis[:, 1]]))
        dist = jnp.maximum(dist, 0.0001)
        psik = nuis[psi]
        si, sj = nuis[sig[:, 0]], nuis[sig[:, 1]]
        voli, volj = four3pi * si ** 3, four3pi * sj ** 3
        close = dist < si + sj
        fi = jnp.where(close, jnp.minimum(voli, xlvol),
                       sphere_cap(si, half, jnp.abs(dist - sj - half)))
        fj = jnp.where(close, jnp.minimum(volj, xlvol),
                       sphere_cap(sj, half, jnp.abs(dist - si - half)))
        pofr = fi * fj / voli / volj
        term = psik * (1.0 - pofr) + pofr * (1 - psik)
        if slope is not None:
            term = term * jnp.exp(-slope * dist)
        if single:
            # one contribution per restraint (the usual PMI case): no segment op
            onemprob = 1.0 - term
        else:
            onemprob = jax.ops.segment_prod(1.0 - term, rid,
                                            num_segments=nres,
                                            indices_are_sorted=True)
        prob = 1.0 - onemprob
        val = -jnp.log(prob) if log_prob else prob
        return val * weights

    return prob_func


def _patch_xl_vec():
    import jax.numpy as jnp
    import IMP.isd
    orig = IMP.isd.LogWrapper._get_jax

    def _get_jax(self, space):
        rs = [r.get_derived_object() for r in self.restraints]
        if not rs or not all(isinstance(r, IMP.isd.CrossLinkMSRestraint)
                             for r in rs):
            return orig(self, space)
        f = _make_xl_prob_func(rs, space)
        if f is None:
            return orig(self, space)

        def jax_sf(jm):
            return -jnp.sum(jnp.log(f(jm)))
        return self._wrap_jax(jax_sf, keys=[IMP.isd.Scale.get_scale_key()])

    IMP.isd.LogWrapper._get_jax = _get_jax


# ---------------------------------------------------------------------------
# pair_vec: one vectorized call per shared PairScore in a RestraintSet
# ---------------------------------------------------------------------------
def _patch_pair_vec():
    import jax.numpy as jnp
    import IMP

    orig = IMP.RestraintSet._get_jax

    def _get_jax(self, space):
        rs = [r.get_derived_object() for r in self.restraints]
        groups, others = {}, []
        for r in rs:
            if type(r) is IMP.core.PairRestraint:
                ps = r.get_score_object()
                key = (int(ps.this), r.get_weight())
                groups.setdefault(key, (ps.get_derived_object(), []))[1] \
                    .append(r.get_index())
            else:
                others.append(r)
        if not any(len(v[1]) > 1 for v in groups.values()):
            return orig(self, space)
        funcs, keys = [], set()
        m = self.get_model()
        for (_, weight), (ps, idx) in groups.items():
            ji = ps._get_jax(m, jnp.asarray(np.asarray(idx, np.int32)),
                             space)
            keys |= ji._keys
            funcs.append(functools.partial(
                lambda jm, f, w: w * jnp.sum(f(jm)), f=ji.score_func,
                w=weight))
        for r in others:
            ji = r._get_jax(space)
            keys |= ji._keys
            funcs.append(ji.score_func)

        def jax_sf(jm):
            return sum(f(jm) for f in funcs)
        return self._wrap_jax(jax_sf, keys=keys)

    IMP.RestraintSet._get_jax = _get_jax


# ---------------------------------------------------------------------------
# unroll the MC inner loop
# ---------------------------------------------------------------------------
def _patch_unroll(n):
    import jax
    import IMP.core._jax_util as cj
    orig_init = cj._MCJAXOptimizer.__init__

    def __init__(self, mc, max_steps, space):
        orig_init(self, mc, max_steps, space)
        ji = self._jax_info
        self.apply_func = jax.jit(
            lambda s: jax.lax.fori_loop(0, self.inner_steps,
                                        lambda i, s: ji.apply_func(s), s,
                                        unroll=n))

    cj._MCJAXOptimizer.__init__ = __init__


# ---------------------------------------------------------------------------
# pdb_cache: PMI best-scoring PDB output recomputes a static topology per write
# ---------------------------------------------------------------------------
def _patch_pdb_cache():
    import IMP
    import IMP.atom
    import IMP.pmi.output
    orig = IMP.pmi.output.Output.get_particle_infos_for_pdb_writing

    def _record(self, name, infos):
        """Match every output row to its particle by exact coordinates.
           Returns particle indexes (in output order) or None if ambiguous."""
        prot = self.dictionary_pdbs[name]
        if not hasattr(prot, 'get_model'):
            return None
        sel = IMP.atom.Selection(prot, resolution=0).get_selected_particles()
        if not sel:
            return None
        m = prot.get_model()
        xyz = m.get_spheres_numpy()[0]
        byxyz = {}
        for p in sel:
            pi = int(p.get_index())
            byxyz.setdefault(tuple(xyz[pi]), []).append(pi)
        pidx = []
        for t in infos:
            cand = byxyz.get(tuple(float(x) for x in t[0]))
            if cand is None or len(cand) != 1:
                return None
            pidx.append(cand[0])
        return m, np.asarray(pidx, dtype=np.int64), [t[1:] for t in infos]

    def get_particle_infos_for_pdb_writing(self, name):
        cache = self.__dict__.setdefault('_pdb_info_cache', {})
        prot = self.dictionary_pdbs[name]
        key = (id(prot), tuple(sorted(self.dictchain[name].items())))
        ent = cache.get(key, False)
        if ent is False:
            infos, gc = orig(self, name)
            cache[key] = _record(self, name, infos)
            return infos, gc
        if ent is None:
            return orig(self, name)
        m, pidx, rest = ent[:3]
        coords = m.get_spheres_numpy()[0][pidx]
        gc = tuple(coords.mean(axis=0)) if len(coords) else (0, 0, 0)
        return [(tuple(c),) + r for c, r in zip(coords.tolist(), rest)], gc

    IMP.pmi.output.Output.get_particle_infos_for_pdb_writing = \
        get_particle_infos_for_pdb_writing


# ---------------------------------------------------------------------------
# fast_update: Model.update() recomputes dependencies on every call
# ---------------------------------------------------------------------------
def _patch_fast_update():
    import IMP
    orig = IMP.Model.update
    cache = {}

    def update(self):
        sss = self.get_score_states()
        key = id(self)
        ids = tuple(int(s.this) for s in sss)
        ent = cache.get(key)
        if ent is None or ent[0] != ids:
            ent = cache[key] = (ids, list(IMP.get_update_order(sss)))
        for s in ent[1]:
            s.before_evaluate()

    update.__doc__ = orig.__doc__
    IMP.Model.update = update


# ---------------------------------------------------------------------------
# pdb_fast: per-row PDB line templates, only coordinates change
# ---------------------------------------------------------------------------
def _patch_pdb_fast():
    import IMP
    import IMP.atom
    import IMP.pmi.output
    _patch_pdb_cache()
    orig_write_pdb = IMP.pmi.output.Output.write_pdb

    def _templates(ent):
        if len(ent) == 4:
            return ent[3]
        m, pidx, rest = ent[:3]
        tpl = []
        for n, r in enumerate(rest):
            atom_type, residue_type, chain_id, resind, _, radius = r
            if atom_type is None:
                atom_type = IMP.atom.AT_CA
            line = IMP.atom.get_pdb_string((0., 0., 0.), n + 1, atom_type,
                                           residue_type, chain_id[:1],
                                           resind, ' ', 1.00, radius)
            if line[30:54] != '   0.000   0.000   0.000':
                return None
            tpl.append((line[:30], line[54:]))
        return tpl

    def write_pdb(self, name, appendmode=True,
                  translate_to_geometric_center=False,
                  write_all_residues_per_bead=False):
        if self._pdb_mmcif[name] or write_all_residues_per_bead:
            return orig_write_pdb(self, name, appendmode,
                                  translate_to_geometric_center,
                                  write_all_residues_per_bead)
        infos, gc = self.get_particle_infos_for_pdb_writing(name)
        cache = self.__dict__.get('_pdb_info_cache', {})
        prot = self.dictionary_pdbs[name]
        key = (id(prot), tuple(sorted(self.dictchain[name].items())))
        ent = cache.get(key)
        tpl = _templates(ent) if ent else None
        if tpl is None or len(tpl) != len(infos):
            return orig_write_pdb(self, name, appendmode,
                                  translate_to_geometric_center,
                                  write_all_residues_per_bead)
        if len(ent) == 3:
            cache[key] = ent + (tpl,)
        if not translate_to_geometric_center:
            gc = (0, 0, 0)
        g0, g1, g2 = gc
        out = [f'{h}{x - g0:8.3f}{y - g1:8.3f}{z - g2:8.3f}{t}'
               for (h, t), ((x, y, z), *_) in zip(tpl, infos)]
        with open(name, 'a' if appendmode else 'w') as fh:
            fh.write(''.join(out))
            fh.write('ENDMDL\n')

    IMP.pmi.output.Output.write_pdb = write_pdb


# ---------------------------------------------------------------------------
# stat_fast: cross-link distances for stat files via numpy
# ---------------------------------------------------------------------------
def _patch_stat_fast():
    from math import log
    import IMP.pmi.restraints.crosslinking as cl
    XL = cl.CrossLinkingMassSpectrometryRestraint

    def get_output(self):
        output = super(XL, self).get_output()(None)
        c = self.__dict__.get('_stat_fast')
        if c is None:
            i0 = np.asarray([int(xl['Particle1'].get_particle_index())
                             for xl in self.xl_list], dtype=np.int64)
            i1 = np.asarray([int(xl['Particle2'].get_particle_index())
                             for xl in self.xl_list], dtype=np.int64)
            keys = [("CrossLinkingMassSpectrometryRestraint_Score_"
                     + xl["ShortLabel"],
                     "CrossLinkingMassSpectrometryRestraint_Distance_"
                     + xl["ShortLabel"], xl["Restraint"])
                    for xl in self.xl_list]
            c = self._stat_fast = (i0, i1, keys)
        i0, i1, keys = c
        xyz = self.model.get_spheres_numpy()[0]
        d = xyz[i0] - xyz[i1]
        # same operation order as IMP::algebra::get_distance
        dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
                       + d[:, 2] * d[:, 2])
        for (skey, dkey, ln), dk in zip(keys, dist.tolist()):
            output[skey] = str(-log(ln.unprotected_evaluate(None)))
            output[dkey] = str(dk)
        for psiname in self.psi_dictionary:
            output["CrossLinkingMassSpectrometryRestraint_Psi_" +
                   str(psiname) + self._label_suffix] = str(
                        self.psi_dictionary[psiname][0].get_scale())
        for sigmaname in self.sigma_dictionary:
            output["CrossLinkingMassSpectrometryRestraint_Sigma_" +
                   str(sigmaname) + self._label_suffix] = str(
                    self.sigma_dictionary[sigmaname][0].get_scale())
        return output

    XL.get_output = get_output
