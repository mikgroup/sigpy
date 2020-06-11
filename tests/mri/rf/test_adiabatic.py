import unittest
import numpy as np
import numpy.testing as npt

from sigpy.mri import rf

if __name__ == '__main__':
    unittest.main()


class TestAdiabatic(unittest.TestCase):

    def test_bir4(self):
        # test an excitation bir4 pulse
        n = 1176
        dt = 4e-6
        dw0 = 100 * np.pi / dt / n
        beta = 10
        kappa = np.arctan(20)
        flip = np.pi / 4
        [am_bir, om_bir] = rf.adiabatic.bir4(n, beta, kappa, flip, dw0)

        # check relatively homogeneous over range of B1 values
        b1 = np.arange(0.2, 0.8, 0.1)
        b1 = np.reshape(b1, (np.size(b1), 1))
        a = np.zeros(np.shape(b1), dtype='complex')
        b = np.zeros(np.shape(b1), dtype='complex')

        for ii in range(0, np.size(b1)):
            [a[ii], b[ii]] = rf.sim.abrm_nd(
                2 * np.pi * dt * 4258 * b1[ii] * am_bir, np.ones(1),
                dt * np.reshape(om_bir, (np.size(om_bir), 1)))

        mxy = 2 * np.multiply(np.conj(a), b)

        test = np.ones(mxy.shape) * 0.7  # magnetization value we expect

        npt.assert_array_almost_equal(np.abs(mxy), test, 2)

    def test_hyp_ex(self):
        # test an inversion adiabatic hyp pulse
        n = 512
        beta = 800
        mu = 4.9
        dur = 0.012
        [am_sech, om_sech] = rf.adiabatic.hypsec(n, beta, mu, dur)

        # check relatively homogeneous over range of B1 values
        b1 = np.arange(0.2, 0.8, 0.1)
        b1 = np.reshape(b1, (np.size(b1), 1))

        a = np.zeros(np.shape(b1), dtype='complex')
        b = np.zeros(np.shape(b1), dtype='complex')
        for ii in range(0, np.size(b1)):
            [a[ii], b[ii]] = rf.sim.abrm_nd(
                2 * np.pi * (dur / n) * 4258 * b1[ii] * am_sech, np.ones(1),
                dur / n * np.reshape(om_sech, (np.size(om_sech), 1)))
        mz = 1 - 2 * np.abs(b) ** 2

        test = np.ones(mz.shape) * -1  # magnetization value we expect

        npt.assert_array_almost_equal(mz, test, 2)

    def test_goia_wurst(self):
        # test a goia-wurst adiabatic pulse
        n = 512
        dur = 3.5e-3
        f = 0.9
        n_b1 = 16
        m_grad = 4
        [_, om_goia, g_goia] = rf.adiabatic.goia_wurst(n, dur, f, n_b1, m_grad)

        # test midpoint of goia pulse. Expect 1-f g, 0.1 fm
        npt.assert_almost_equal(g_goia[int(len(g_goia)/2)], 1-f, 2)
        npt.assert_almost_equal(g_goia[int(len(om_goia)/2)], 0.1, 2)
