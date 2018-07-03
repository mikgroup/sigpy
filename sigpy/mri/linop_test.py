import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import linop
from sigpy.linop_test import check_linop_adjoint

if __name__ == '__main__':
    unittest.main()


class TestLinop(unittest.TestCase):

    def test_shepp_logan_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.util.randn(img_shape)
        mps = sp.util.randn(mps_shape)

        mask = np.zeros(img_shape)
        mask[::2, ::2] = 1.0

        A = linop.Sense(mps)

        check_linop_adjoint(A)

        npt.assert_allclose(sp.fft.fft(img * mps, axes=[-1, -2]),
                            A * img)

    def test_shepp_logan_noncart_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.util.randn(img_shape)
        mps = sp.util.randn(mps_shape)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)

        A = linop.Sense(mps, coord=coord)
        check_linop_adjoint(A)
        npt.assert_allclose(sp.fft.fft(img * mps, axes=[-1, -2]).ravel(),
                            (A * img).ravel(), atol=2, rtol=2)
