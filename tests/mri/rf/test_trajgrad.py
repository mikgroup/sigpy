import unittest
import numpy as np
import numpy.testing as npt
import sigpy.mri.rf as rf
import math

if __name__ == '__main__':
    unittest.main()


class TestTrajGrad(unittest.TestCase):

    def test_min_gradient(self):
        t = np.linspace(0, 1, 1000)
        kx = np.sin(2.0 * math.pi * t)
        ky = np.cos(2.0 * math.pi * t)
        kz = t
        k = np.stack((kx, ky, kz), axis=-1)

        (g, k, s, t) = rf.min_time_gradient(k, 0.0, 0.0,
                                            gmax=4, smax=15, dt=4e-3,
                                            show=False, gamma=4.257)

        npt.assert_almost_equal(np.max(t), 0.916, decimal=4)
