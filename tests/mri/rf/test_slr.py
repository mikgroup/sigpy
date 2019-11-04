import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestSlr(unittest.TestCase):

    def test_st(self):
        #  check to make sure profile roughly matches anticipated within d1, d2
        N = 128
        tb = 16
        filts = ['ls', 'ms', 'pm', 'min', 'max']
        for idx, filt in enumerate(filts):
            pulse = sp.mri.rf.dzrf(N, tb, ptype='st', ftype=filt,
                                   d1=0.01, d2=0.01)

            m = abs(np.fft.fftshift(np.fft.fft(pulse)))

            pts = np.array([m[int(N / 2 - 10)], m[int(N / 2)],
                            m[int(N / 2 + 10)]])
            npt.assert_almost_equal(pts, np.array([0, 1, 0]), decimal=2)

    def test_inv(self):
        #  also provides testing of sim. Check inv profile.
        tb = 8
        N = 128
        d1 = 0.01
        d2 = 0.01
        ptype = 'ex'
        filts = ['min', 'max']  # filts produce inconsistent inversions

        for idx, filt in enumerate(filts):
            pulse = rf.slr.dzrf(N, tb, ptype, filt, d1, d2)

            [a, b] = rf.sim.abrm(pulse, np.arange(-2 * tb, 2 * tb, 0.01))
            Mz = 1 - 2 * np.abs(b) ** 2

            pts = np.array([Mz[int(len(Mz) / 2 - len(Mz)/3)],
                            Mz[int(len(Mz) / 2)],
                            Mz[int(len(Mz) / 2 + len(Mz)/3)]])

            npt.assert_almost_equal(pts, np.array([1, -0.2, 1]), decimal=1)
