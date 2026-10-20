"""Per-term comparison of scoring.py (numpy and JAX, float64) with C++ IMP on
configurations visited by a C++ MC run of the tutorial's own sampler.

* 'frozen' covariances vs C++ as it runs (stale core::Gaussian cache);
* 'rotate' covariances vs C++ after update_global_covariance() on the
  model Gaussians.

    JAX_PLATFORMS=cpu python validate_terms.py [--configs 8]
"""
import argparse

import numpy as np
import IMP
import IMP.algebra
import IMP.container
import IMP.core
import IMP.pmi.samplers

import scoring as T
from build_system import build


def dofs(s):
    m = s.m
    q, t = [], []
    for b in s.bodies:
        tr = b.get_reference_frame().get_transformation_to()
        q.append(tr.get_rotation().get_quaternion())
        t.append(tr.get_translation())
    bead = [IMP.core.XYZ(m, IMP.ParticleIndex(int(p))).get_coordinates()
            for p in s.bead_pi]
    return np.array(q), np.array(t), np.array(bead)


def set_dofs(s, q, t, bead):
    m = s.m
    for b, qq, tt in zip(s.bodies, q, t):
        b.set_reference_frame(IMP.algebra.ReferenceFrame3D(
            IMP.algebra.Transformation3D(IMP.algebra.Rotation3D(*qq),
                                         IMP.algebra.Vector3D(*tt))))
    for p, x in zip(s.bead_pi, bead):
        nr = IMP.core.NonRigidMember(m, IMP.ParticleIndex(int(p)))
        tr = nr.get_rigid_body().get_reference_frame()
        nr.set_internal_coordinates(tr.get_local_coordinates(
            IMP.algebra.Vector3D(*x)))
    m.update()


def em_list_check(s):
    """C++ EM close-pair lists vs sphere distance thresholds"""
    gem = s.S.gemt.gaussianEM_restraint
    m = s.m
    out = {}
    for o in gem.get_inputs():
        tn = o.get_type_name()
        if tn.startswith('CloseBipartitePairContainer'):
            c = IMP.container.CloseBipartitePairContainer.get_from(o)
            key, ra, rb = 'md', s.em_rad, s.d_rad
            ia = {int(p): i for i, p in enumerate(s.em_model_pi)}
            ib = {int(p): i for i, p in enumerate(s.em_dens_pi)}
            xa = np.array([IMP.core.XYZ(m, IMP.ParticleIndex(int(p)))
                           .get_coordinates() for p in s.em_model_pi])
            xb = s.d_xyz
        elif tn.startswith('ClosePairContainer'):
            c = IMP.container.ClosePairContainer.get_from(o)
            key, ra, rb = 'mm', s.em_rad, s.em_rad
            ia = ib = {int(p): i for i, p in enumerate(s.em_model_pi)}
            xa = xb = np.array([IMP.core.XYZ(m, IMP.ParticleIndex(int(p)))
                                .get_coordinates() for p in s.em_model_pi])
        else:
            continue
        pairs = [(ia[int(a)], ib[int(b)]) for a, b in c.get_indexes()]
        sph = [np.linalg.norm(xa[i] - xb[j]) - ra[i] - rb[j]
               for i, j in pairs]
        out[key] = (len(pairs), max(sph) if sph else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', type=int, default=8)
    ap.add_argument('--steps', type=int, default=490)
    a = ap.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    S = build(seed=1)
    sysf = T.TutSystem(S, cov_mode='frozen')
    sysr = T.TutSystem(S, cov_mode='rotate')
    smp = IMP.pmi.samplers.MonteCarlo(S.m, S.movers, 1.0)
    mc = smp.get_mc()
    configs = []
    for c in range(a.configs):
        if c:
            mc.optimize(a.steps * (1 if c < a.configs // 2 else 20))
        configs.append(dofs(sysf))

    import copy

    def jterms_for(s):
        def f(q, t, b, mdm, mmm):
            s2 = copy.copy(s)
            s2.em_md_mask, s2.em_mm_mask = mdm, mmm
            return T.energy_terms(s2, q, t, b, jnp)
        return jax.jit(f)
    jt = {'frozen': jterms_for(sysf), 'rotate': jterms_for(sysr)}
    worst = {}
    for mode, s in (('frozen', sysf), ('rotate', sysr)):
        print(f'\n== covariances: {mode}')
        for c, (q, t, b) in enumerate(configs):
            set_dofs(s, q, t, b)
            if mode == 'rotate':
                for p in s.em_model_pi:
                    IMP.core.Gaussian(s.m, IMP.ParticleIndex(int(p))) \
                        .update_global_covariance()
            ref = T.cpp_terms(S)
            q2, t2, b2 = dofs(s)
            s.em_md_mask = s.em_mm_mask = None
            em_cut = T.energy_terms(s, q2, t2, b2, np)['em']
            s.em_masks_from_cpp()
            npv = T.energy_terms(s, q2, t2, b2, np, f32_caps=True)
            jv = jt[mode](jnp.asarray(q2), jnp.asarray(t2), jnp.asarray(b2),
                          jnp.asarray(s.em_md_mask),
                          jnp.asarray(s.em_mm_mask))
            lst = em_list_check(s)
            line = []
            for k in ref:
                e_np = abs(npv[k] - ref[k])
                e_j = abs(float(jv[k]) - ref[k])
                worst[(mode, k)] = max(worst.get((mode, k), 0.), e_np, e_j)
                line.append(f'{k} {ref[k]:.6f} (np {e_np:.1e}, jax {e_j:.1e})')
            print(f'config {c}: ' + '; '.join(line))
            print(f'   C++ EM lists: md {lst["md"]}, mm {lst["mm"]} '
                  f'(n, max sphere dist); EM with pairs within the cutoff '
                  f'only: {em_cut:.6f} ({em_cut - ref["em"]:+.4f})')
    print('\nworst |JAX or numpy - C++| per term (numpy emulates the float '
          'arithmetic of sphere_cap, JAX does not; both use the C++ EM '
          'pair lists):')
    for k, v in worst.items():
        print('  %-24s %.2e' % (k, v))


if __name__ == '__main__':
    main()
