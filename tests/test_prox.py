import unittest

import numpy as np
import numpy.testing as npt

from sigpy import linop, prox, util

if __name__ == "__main__":
    unittest.main()


class TestProx(unittest.TestCase):
    def test_L1Reg(self):
        shape = [6]
        lamda = 1.0
        P = prox.L1Reg(shape, lamda)
        phase = np.exp(1j * np.random.random(shape))
        x = np.array([-3.0, -2.0, -1.0, 0, 1.0, 2.0]) * phase
        y = P(1.0, x)
        z = np.array([-2.0, -1.0, -0.0, 0, 0.0, 1.0]) * phase
        npt.assert_allclose(y, z)

    def test_L1Proj(self):
        shape = [6]
        epsilon = 1.0
        P = prox.L1Proj(shape, epsilon)
        x = util.randn(shape)
        y = P(1.0, x)
        z = 1.0 if np.linalg.norm(x, 1) > 1.0 else np.linalg.norm(x, 1)
        npt.assert_allclose(np.linalg.norm(y, 1), z)

        x = util.randn(shape) * 0.0001
        y = P(1.0, x)
        z = 1.0 if np.linalg.norm(x, 1) > 1.0 else np.linalg.norm(x, 1)
        npt.assert_allclose(np.linalg.norm(y, 1), z)

    def test_UnitaryTransform(self):
        shape = [6]
        lamda = 1.0
        A = linop.FFT(shape)
        P = prox.UnitaryTransform(prox.L2Reg(shape, lamda), A)
        x = util.randn(shape)
        y = P(0.1, x)
        npt.assert_allclose(y, x / (1 + lamda * 0.1), atol=1e-6, rtol=1e-6)

    def test_L2Reg(self):
        shape = [6]
        lamda = 1.0
        P = prox.L2Reg(shape, lamda)
        x = util.randn(shape)
        y = P(0.1, x)
        npt.assert_allclose(y, x / (1 + lamda * 0.1))

    def test_L2Proj(self):
        shape = [6]
        epsilon = 1.0
        P = prox.L2Proj(shape, epsilon)
        x = util.randn(shape) * 10
        y = P(1.0, x)
        npt.assert_allclose(y, x / np.linalg.norm(x.ravel()))

    def test_LInfProj(self):
        shape = [5]
        epsilon = 0.6
        P = prox.LInfProj(shape, epsilon)
        x = np.array([-1, -0.5, 0, 0.5, 1])
        y = P(1.0, x)
        npt.assert_allclose(y, [-0.6, -0.5, 0, 0.5, 0.6])

    def test_PsdProj(self):
        shape = [3, 3]
        P = prox.PsdProj(shape)
        x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -2]])
        y = P(None, x)
        npt.assert_allclose(y, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))

    def test_BoxConstraint(self):
        shape = [5]
        P = prox.BoxConstraint(shape, -1, 1)
        x = np.array([-2, -1, 0, 1, 2])
        y = P(None, x)
        npt.assert_allclose(y, [-1, -1, 0, 1, 1])

        P = prox.BoxConstraint(shape, [-1, 0, -1, -1, -1], [1, 1, 1, 0, 1])
        x = np.array([-2, -1, 0, 1, 2])
        y = P(None, x)
        npt.assert_allclose(y, [-1, 0, 0, 0, 1])

    def test_Conj(self):
        shape = [3, 3]
        x = util.randn(shape, dtype=float)

        F = linop.FiniteDifference(shape, axes=(-2, -1))
        proxg = prox.L1Reg(F.oshape, 1.)
        proxgc = prox.Conj(proxg)

        O = F(x)
        y1 = proxgc(1., O)

        x1 = O[0, :, :]
        x2 = O[1, :, :]
        d1 = np.maximum(1., abs(x1))
        x1n = np.divide(x1, d1)
        d2 = np.maximum(1., abs(x2))
        x2n = np.divide(x2, d2)
        y2 = np.stack((x1n, x2n))

        npt.assert_allclose(y1, y2)

    def test_LLRL1Reg(self):
        shape = [15, 48, 32]
        x = util.randn(shape, dtype=complex)
        L = prox.LLRL1Reg(shape, 1)
        y = L(0., x)
        npt.assert_allclose(y, x)

    def test_SLRMCReg(self):
        shape = [15, 48, 32]
        x = util.randn(shape, dtype=complex)
        L = prox.SLRMCReg(shape, 1, blk_shape=(8, 8))
        y = L(1., x)
        npt.assert_allclose(y, x)
