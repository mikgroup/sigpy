import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp

if __name__ == '__main__':
    unittest.main()


class TestSlr(unittest.TestCase):

    def test_slr(self):
        N = 128
        tb = 16
        rf = sp.mri.rf.dzrf(N, tb, ptype='st', ftype='pm', d1=0.01, d2=0.01)

        m = abs(np.fft.fftshift(np.fft.fft(rf)))
        pts = np.array([m[int(N / 2 - 10)], m[int(N / 2)], m[int(N / 2 + 10)]])
        npt.assert_almost_equal(pts, np.array([0, 1, 0]), decimal=2)
