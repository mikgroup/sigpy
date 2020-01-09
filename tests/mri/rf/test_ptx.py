import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt
import scipy.ndimage.filters as filt

from sigpy.mri import rf, linop, sim

if __name__ == '__main__':
    unittest.main()


class TestPtx(unittest.TestCase):

    img_shape = [32, 32]
    sens_shape = [8, 32, 32]
    dt = 4e-6

    # target - slightly blurred circle
    x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
                    -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
    circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
    target = np.zeros(img_shape)
    target[circle] = 1
    target = filt.gaussian_filter(target, 1)
    target = target.astype(np.complex)

    sens = sim.birdcage_maps(sens_shape)

    def test_stspa_radial(self):

        # makes dim*dim*2 trajectory
        traj = sp.mri.radial((self.sens.shape[1], self.sens.shape[1], 2),
                             self.img_shape, golden=True, dtype=np.float)
        # reshape to be Nt*2 trajectory
        traj = np.reshape(traj, [traj.shape[0]*traj.shape[1], 2])

        A = linop.Sense(self.sens, coord=traj,
                        weights=None, ishape=self.target.shape).H

        pulses = rf.stspa(self.target, self.sens, traj, self.dt, alpha=1,
                          B0=None, pinst=float('inf'), pavg=float('inf'),
                          explicit=False, max_iter=100, tol=1E-4)

        npt.assert_array_almost_equal(A*pulses, self.target, 1E-3)

    def test_stspa_spiral(self):

        dim = self.img_shape[0]
        traj = sp.mri.spiral(fov=dim / 2, N=self.img_shape[0],
                             f_sampling=1, R=1, ninterleaves=1, alpha=1,
                             gm=0.03, sm=200)

        A = linop.Sense(self.sens, coord=traj, ishape=self.target.shape).H

        pulses = rf.stspa(self.target, self.sens, traj, self.dt, alpha=1,
                          B0=None, pinst=float('inf'), pavg=float('inf'),
                          explicit=False, max_iter=100, tol=1E-4)

        npt.assert_array_almost_equal(A*pulses, self.target, 1E-3)

    def test_stspa_phase_update(self):

        dim = self.img_shape[0]
        traj = sp.mri.spiral(fov=1, N=dim, f_sampling=1, R=0.75,
                             ninterleaves=1, alpha=1, gm=0.03, sm=200)

        A = linop.Sense(self.sens, coord=traj, ishape=self.target.shape).H

        pulses_nop = rf.stspa(self.target, self.sens, traj, self.dt, alpha=1,
                              B0=None, pinst=float('inf'), pavg=float('inf'),
                              phase_update_interval=200,
                              explicit=False, max_iter=100, tol=1E-9)
        pulses_wip = rf.stspa(self.target, self.sens, traj, self.dt, alpha=1,
                              B0=None, pinst=float('inf'), pavg=float('inf'),
                              phase_update_interval=2,
                              explicit=False, max_iter=100, tol=1E-9)

        err_nop = np.sum(np.sum(A*pulses_nop-self.target))
        err_wip = np.sum(np.sum(A*pulses_wip-self.target))

        # we expect less error if phase updates are used
        npt.assert_array_less(np.abs(err_wip), np.abs(err_nop))
