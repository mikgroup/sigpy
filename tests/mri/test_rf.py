import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt
import scipy.ndimage.filters as filt

from sigpy.mri import rf, linop, sim

if __name__ == '__main__':
    unittest.main()


class TestRf(unittest.TestCase):

    img_shape = [32, 32]
    sens_shape = [8, 32, 32]

    # target - slightly blurred circle
    x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
                    -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
    circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
    target = np.zeros(img_shape)
    target[circle] = 1
    target = filt.gaussian_filter(target, 1)
    target = target.astype(np.complex)

    sens = sim.birdcage_maps(sens_shape)

    def test_stspa_cartesian(self):

        # subsampled cartesian R=2 mask
        mask = np.zeros(self.sens_shape)
        mask[:, ::2, :] = 1.0

        A = linop.Sense(self.sens, coord=None,
                        weights=mask, ishape=self.target.shape).H

        pulses = rf.stspa(self.target, self.sens, coord=None, mask=mask,
                          max_iter=1000, tol=1E-3)

        npt.assert_array_almost_equal(A*pulses, self.target, 1E-3)

    def test_stspa_radial(self):

        # makes dim*dim*2 trajectory
        traj = sp.mri.radial((self.sens.shape[1], self.sens.shape[1], 2),
                             self.img_shape, golden=True, dtype=np.float)
        # reshape to be Nt*2 trajectory
        traj = np.reshape(traj, [traj.shape[0]*traj.shape[1], 2])

        A = linop.Sense(self.sens, coord=traj,
                        weights=None, ishape=self.target.shape).H

        pulses = rf.stspa(self.target, self.sens, mask=None, coord=traj,
                          max_iter=1000, tol=1E-3)

        npt.assert_array_almost_equal(A*pulses, self.target, 1E-3)

    def test_stspa_spiral(self):

        dim = self.img_shape[0]
        traj = sp.mri.spiral(fov=dim / 2, img_shape=self.img_shape, f_sampling=1, R=1, ninterleaves=1, alpha=1, gm=0.03,
                             sm=200) * (dim / 2)

        A = linop.Sense(self.sens, coord=traj, ishape=self.target.shape).H

        pulses = rf.stspa(self.target, self.sens, mask=None, pavg=np.Inf, pinst=np.Inf,
                          coord=traj, max_iter=1000, tol=1E-3)

        npt.assert_array_almost_equal(A*pulses, self.target, 1E-3)

    def test_slr(self):

        N = 128
        tb = 16
        rf = sp.mri.rf.dzrf(N, tb, ptype='st', ftype='pm', d1=0.01, d2=0.01)

        m = abs(np.fft.fftshift(np.fft.fft(rf)))

        npt.assert_almost_equal(np.array([m[int(N/2-10)], m[int(N/2)], m[int(N/2+10)]]), np.array([0, 1, 0]), decimal=2)
