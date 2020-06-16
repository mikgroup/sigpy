import unittest

import numpy as np
import numpy.testing as npt

import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestB1sel(unittest.TestCase):

    def test_b1sel_generic(self):

        dt = 2e-6  # sampling period
        d1 = 0.01  # passband ripple
        d2 = 0.01  # stopband ripple
        tb = 4  # time-bandwidth product
        ptype = 'ex'  # 'st', 'ex', 'inv' or 'sat'
        pbw = 0.5  # gauss, passband width
        pbc = 5  # gauss, passband center
        flip = np.pi / 4  # radians, flip angle

        [rf_am, rf_fm] = rf.b1sel.dz_b1_rf(dt, tb, ptype, flip, pbw, pbc, d1,
                                           d2)
        b1 = np.arange(0, 2 * pbc, 2 * pbc / np.size(
            rf_am) * 4)  # b1 grid we simulate the pulse over
        b1 = np.reshape(b1, (np.size(b1), 1))
        [a, b] = rf.sim.abrm_nd(2 * np.pi * dt * rf_fm, b1,
                                2 * np.pi * 4258 * dt *
                                np.reshape(rf_am, (np.size(rf_am), 1)))
        mxy = -2 * np.real(a * b) + 1j * np.imag(np.conj(a) ** 2 - b ** 2)

        pts = np.array([mxy[10], mxy[int(len(b1) / 2)],
                        mxy[len(b1) - 10]])
        npt.assert_almost_equal(abs(pts), np.array([0, 0.7, 0]), decimal=1)

    def test_b1sel_gslider(self):
        g = 5
        flip = np.pi / 2
        ptype = 'ex'  # 'ex' or 'st'
        tb = 12
        d1 = 0.01
        d2 = 0.01
        pbc = 1  # gauss, passband center
        pbw = 0.25  # passband width
        dt = 2e-6  # seconds, sampling rate
        [om1, dom] = rf.b1sel.dz_b1_gslider_rf(dt, g, tb, ptype, flip, pbw,
                                               pbc, d1, d2)

        n = np.shape(om1)[0]
        b1 = np.arange(0, 2 * pbc,
                       2 * pbc / n * 4)  # b1 grid we simulate the pulse over
        b1 = np.reshape(b1, (np.size(b1), 1))
        [a, b] = rf.sim.abrm_nd(2 * np.pi * dt * dom[:, 0], b1,
                                2 * np.pi * 4258 * dt * np.reshape(om1[:, 0],
                                                                   (n, 1)))
        mxy = -2 * np.real(a * b) + 1j * np.imag(np.conj(a) ** 2 - b ** 2)

        pts = np.array([mxy[10], mxy[int(len(b1) / 2)],
                        mxy[len(b1) - 10]])
        npt.assert_almost_equal(abs(pts), np.array([0, 1, 0]), decimal=2)
