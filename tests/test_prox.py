import unittest
import numpy as np
import numpy.testing as npt
from sigpy import prox, util, linop

if __name__ == '__main__':
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
        npt.assert_allclose(y, x / (1 + lamda * 0.1))

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
        x = util.randn(shape)
        y = P(1.0, x)
        npt.assert_allclose(y, x / np.linalg.norm(x.ravel()))

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
