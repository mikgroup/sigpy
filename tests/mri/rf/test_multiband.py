import unittest

import numpy as np
import numpy.testing as npt

import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestMultiband(unittest.TestCase):

    def test_multiband(self):

        # slr pulse
        tb = 8
        N = 512
        d1 = 0.01
        d2 = 0.01
        p_type = 'ex'
        f_type = 'ls'
        pulse = rf.slr.dzrf(N, tb, p_type, f_type, d1, d2, True)

        # multiband it
        n_bands = 3
        phs_type = 'phs_mod'  # phsMod, ampMod, or quadMod
        band_sep = 5 * tb  # separate by 5 slice widths
        mb_pulse = rf.multiband.mb_rf(pulse, n_bands, band_sep, phs_type)

        # simulate it
        [a, b] = rf.sim.abrm(mb_pulse,
                             np.arange(-20 * tb, 20 * tb, 40 * tb / 2000),
                             True)
        mxy = 2 * np.multiply(np.conj(a), b)

        pts = np.array([mxy[750], mxy[850], mxy[1000], mxy[1150], mxy[1250]])
        npt.assert_almost_equal(abs(pts), np.array([1, 0, 1, 0, 1]), decimal=2)

    def test_pins(self):

        # pins pulse specs
        tb = 8
        d1 = 0.01
        d2 = 0.01
        p_type = 'ex'
        f_type = 'ls'

        sl_sep = 3  # cm
        sl_thick = 0.3  # cm
        g_max = 4  # gauss/cm
        g_slew = 18000  # gauss/cm/s
        dt = 4e-6  # seconds, dwell time
        b1_max = 0.18  # gauss
        [rf_pins, g_pins] = rf.multiband.dz_pins(tb, sl_sep, sl_thick, g_max,
                                                 g_slew, dt, b1_max, p_type,
                                                 f_type, d1, d2)

        # simulate it
        x = np.reshape(np.arange(-1000, 1000), (2000, 1)) / 1000 * 12  # cm
        [a, b] = rf.sim.abrm_nd(2 * np.pi * dt * 4258 * rf_pins, x,
                                np.reshape(g_pins, (np.size(g_pins), 1)) *
                                4258 * dt * 2 * np.pi)
        mxy = 2 * np.conj(a) * b

        pts = np.array([mxy[100], mxy[1000], mxy[1900]])
        npt.assert_almost_equal(abs(pts), np.array([0, 1, 0]), decimal=2)
