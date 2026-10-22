"""Fast DOF -> IMP model for writers: the beads' internal coordinates set
through Model.get_internal_coordinates_numpy() in one numpy assignment
(their owner bodies never move, so their frames are fixed)."""
import numpy as np


class FastDofs:
    def __init__(self, s):
        import IMP
        import IMP.core
        self.s = s
        m = s.m
        self.pi = np.array([int(p) for p in s.bead_pi])
        Rt, t0 = [], []
        for p in s.bead_pi:
            nr = IMP.core.NonRigidMember(m, IMP.ParticleIndex(int(p)))
            tr = nr.get_rigid_body().get_reference_frame() \
                .get_transformation_to()
            R = np.array([tr.get_rotation().get_rotated(
                IMP.algebra.Vector3D(*e)) for e in np.eye(3)]).T
            Rt.append(R.T)
            t0.append(np.array(tr.get_translation()))
        self.Rt, self.t0 = np.array(Rt), np.array(t0)

    def set(self, quat, trans, bead):
        import IMP
        import IMP.algebra
        s, m = self.s, self.s.m
        for b, qq, tt in zip(s.bodies, quat, trans):
            b.set_reference_frame(IMP.algebra.ReferenceFrame3D(
                IMP.algebra.Transformation3D(
                    IMP.algebra.Rotation3D(*[float(x) for x in qq]),
                    IMP.algebra.Vector3D(*[float(x) for x in tt]))))
        ic = m.get_internal_coordinates_numpy()
        ic[self.pi] = np.einsum('nij,nj->ni', self.Rt,
                                np.asarray(bead, np.float64) - self.t0)
        m.update()
