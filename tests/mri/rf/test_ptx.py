import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt
import scipy.ndimage.filters as filt

from sigpy.mri import rf, linop, sim

if __name__ == '__main__':
    unittest.main()


class TestPtx(unittest.TestCase):

    @staticmethod
    def problem_2d(dim):
        img_shape = [dim, dim]
        sens_shape = [8, dim, dim]

        # target - slightly blurred circle
        x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
                        -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
        circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
        target = np.zeros(img_shape)
        target[circle] = 1
        target = filt.gaussian_filter(target, 1)
        target = target.astype(np.complex)

        sens = sim.birdcage_maps(sens_shape)

        return target, sens

    @staticmethod
    def problem_3d(dim, Nz):
        Nc = 8
        img_shape = [dim, dim, Nz]
        sens_shape = [Nc, dim, dim, Nz]

        # target - slightly blurred circle
        x, y, z = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
                           -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2,
                           -img_shape[2] / 2: img_shape[2] - img_shape[2] / 2]
        circle = x * x + y * y + z * z <= int(img_shape[0] / 5) ** 2
        target = np.zeros(img_shape)
        target[circle] = 1
        target = filt.gaussian_filter(target, 1)
        target = target.astype(np.complex)
        sens = sp.mri.sim.birdcage_maps(sens_shape)

        return target, sens

    def test_stspa_radial(self):

        target, sens = self.problem_2d(8)

        # makes dim*dim*2 trajectory
        traj = sp.mri.radial((sens.shape[1], sens.shape[1], 2),
                             target.shape, golden=True, dtype=np.float)
        # reshape to be Nt*2 trajectory
        traj = np.reshape(traj, [traj.shape[0]*traj.shape[1], 2])

        A = linop.Sense(sens, coord=traj,
                        weights=None, ishape=target.shape).H

        pulses = rf.stspa(target, sens, traj, dt=4e-6, alpha=1,
                          b0=None, st=None,
                          explicit=False, max_iter=100, tol=1E-4)

        npt.assert_array_almost_equal(A*pulses, target, 1E-3)

    def test_stspa_spiral(self):

        target, sens = self.problem_2d(8)

        fov = 0.55
        gts = 6.4e-6
        gslew = 190
        gamp = 40
        R = 1
        dx = 0.025  # in m
        # construct a trajectory
        g, k, t, s = rf.spiral_arch(fov / R, dx, gts, gslew, gamp)

        A = linop.Sense(sens, coord=k, ishape=target.shape).H

        pulses = rf.stspa(target, sens, k, dt=4e-6, alpha=1,
                          b0=None, st=None,
                          explicit=False, max_iter=100, tol=1E-4)

        npt.assert_array_almost_equal(A*pulses, target, 1E-3)

    def test_stspa_2d_explicit(self):
        target, sens = self.problem_2d(8)
        dim = target.shape[0]
        g, k1, t, s = rf.spiral_arch(0.24, dim, 4e-6, 200, 0.035)
        k1 = k1 / dim

        A = rf.PtxSpatialExplicit(sens, k1, dt=4e-6, img_shape=target.shape,
                                  b0=None)
        pulses = sp.mri.rf.stspa(target, sens, st=None, coord=k1, dt=4e-6,
                                 max_iter=100, alpha=10, tol=1E-4,
                                 phase_update_interval=200, explicit=True)

        npt.assert_array_almost_equal(A*pulses, target, 1E-3)

    def test_stspa_3d_explicit(self):
        nz = 4
        target, sens = self.problem_3d(3, nz)
        dim = target.shape[0]

        g, k1, t, s = rf.spiral_arch(0.24, dim, 4e-6, 200, 0.035)
        k1 = k1 / dim

        k1 = rf.stack_of(k1, nz, 0.1)
        A = rf.linop.PtxSpatialExplicit(sens, k1, dt=4e-6,
                                        img_shape=target.shape, b0=None)

        pulses = sp.mri.rf.stspa(target, sens, st=None,
                                 coord=k1,
                                 dt=4e-6, max_iter=30, alpha=10, tol=1E-3,
                                 phase_update_interval=200, explicit=True)

        npt.assert_array_almost_equal(A*pulses, target, 1E-3)

    def test_stspa_3d_nonexplicit(self):
        nz = 3
        target, sens = self.problem_3d(3, nz)
        dim = target.shape[0]

        g, k1, t, s = rf.spiral_arch(0.24, dim, 4e-6, 200, 0.035)
        k1 = k1 / dim

        k1 = rf.stack_of(k1, nz, 0.1)
        A = sp.mri.linop.Sense(sens, k1, weights=None, tseg=None,
                               ishape=target.shape).H

        pulses = sp.mri.rf.stspa(target, sens, st=None,
                                 coord=k1,
                                 dt=4e-6, max_iter=30, alpha=10, tol=1E-3,
                                 phase_update_interval=200, explicit=False)

        npt.assert_array_almost_equal(A*pulses, target, 1E-3)

    def test_spokes(self):

        # spokes problem definition:
        dim = 20  # size of the b1 matrix loaded
        n_spokes = 5
        fov = 20  # cm
        dx_max = 2  # cm
        gts = 4E-6
        sl_thick = 5  # slice thickness, mm
        tbw = 4
        dgdtmax = 18000  # g/cm/s
        gmax = 2  # g/cm

        _, sens = self.problem_2d(dim)
        roi = np.zeros((dim, dim))
        radius = dim//2
        cx, cy = dim//2, dim//2
        y, x = np.ogrid[-radius:radius, -radius:radius]
        index = x**2 + y**2 <= radius**2
        roi[cy-radius:cy+radius, cx-radius:cx+radius][index] = 1
        sens = sens * roi

        [pulses, g] = rf.stspk(roi, sens, n_spokes, fov, dx_max, gts, sl_thick,
                               tbw, dgdtmax, gmax, alpha=1)

        # should give the number of pulses corresponding to number of TX ch
        npt.assert_equal(np.shape(pulses)[0], np.shape(sens)[0])
        # should hit the max gradient constraint
        npt.assert_almost_equal(gmax, np.max(g), decimal=3)
