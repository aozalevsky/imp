"""Minimal check: does GaussianEMRestraint see the rotation of a Gaussian
that is a member of a rigid body?

One elongated model Gaussian (variances 100, 1, 1) is a member of a rigid
body; the data GMM is the same Gaussian. Rotating the body by 90 degrees
about z should change the score (the model Gaussian then lies across the
data Gaussian). Prints the C++ score before/after the rotation, the score
after clearing the Model's caches, and what the cached global covariance
holds. Runs on any IMP with IMP.bayesianem (2.18, 2.25, nightly)."""
import math

import IMP
import IMP.algebra
import IMP.atom
import IMP.core
import IMP.isd
import IMP.bayesianem

print('IMP', IMP.get_module_version())
m = IMP.Model()


def gaussian(name, var, center=(0, 0, 0)):
    p = IMP.Particle(m, name)
    g = IMP.algebra.Gaussian3D(
        IMP.algebra.ReferenceFrame3D(IMP.algebra.Transformation3D(
            IMP.algebra.get_identity_rotation_3d(),
            IMP.algebra.Vector3D(*center))),
        IMP.algebra.Vector3D(*var))
    IMP.core.Gaussian.setup_particle(p, g)
    IMP.atom.Mass.setup_particle(p, 1000.)
    IMP.core.XYZR.setup_particle(p, 10.)
    return p


model_g = gaussian('model', (100., 1., 1.))
data_g = gaussian('data', (100., 1., 1.))
# an extra rigid member offset from the body center, so the body is not
# just the Gaussian itself
anchor = IMP.Particle(m, 'anchor')
IMP.core.XYZR.setup_particle(anchor, IMP.algebra.Sphere3D(
    IMP.algebra.Vector3D(5, 0, 0), 1.))
IMP.atom.Mass.setup_particle(anchor, 1000.)
body = IMP.atom.create_rigid_body([model_g, anchor])

sigma = IMP.isd.Scale.setup_particle(IMP.Particle(m, 'sigma'), 1.0)
import sys
if 'isd' in sys.argv:
    print('restraint: IMP.isd.GaussianEMRestraint')
    r = IMP.isd.GaussianEMRestraint(
        m, [model_g.get_index()], [data_g.get_index()],
        sigma.get_particle_index(), 10., 10., 0., False, False)
else:
    print('restraint: IMP.bayesianem.GaussianEMRestraint')
    r = IMP.bayesianem.GaussianEMRestraint(
        m, [model_g.get_index()], [data_g.get_index()],
        sigma.get_particle_index(), 10., 10., 0., True, False)
sf = IMP.core.RestraintsScoringFunction([r])


def cov():
    c = IMP.core.Gaussian(model_g).get_global_covariance()
    return [round(float(c[i][i]), 2) for i in range(3)]


print('before rotation: score %.6f  cached cov diag %s'
      % (sf.evaluate(False), cov()))
rot = IMP.algebra.get_rotation_about_axis(IMP.algebra.Vector3D(0, 0, 1),
                                          math.pi / 2)
rb = IMP.core.RigidBody(body)
tr = rb.get_reference_frame().get_transformation_to()
rb.set_reference_frame(IMP.algebra.ReferenceFrame3D(
    IMP.algebra.Transformation3D(rot * tr.get_rotation(),
                                 tr.get_translation())))
print('after 90 deg rotation: score %.6f  cached cov diag %s'
      % (sf.evaluate(False), cov()))
m.clear_caches() if hasattr(m, 'clear_caches') else None
print('after Model.clear_caches(): score %.6f  cached cov diag %s'
      % (sf.evaluate(False), cov()))
IMP.core.Gaussian(model_g).update_global_covariance()
print('after Gaussian.update_global_covariance(): score %.6f  cov diag %s'
      % (sf.evaluate(False), cov()))
