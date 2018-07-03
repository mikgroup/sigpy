import unittest
import numpy as np
import numpy.testing as npt
from sigpy import thresh, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestThresh(unittest.TestCase):

    def test_soft_thresh(self):
        x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
        y = np.array([-1, -0.5, 0, 0, 0, 0, 0, 0.5, 1])

        npt.assert_allclose(thresh.soft_thresh(1, x), y)

    def test_hard_thresh(self):
        x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
        y = np.array([-2, -1.5, 0, 0, 0, 0, 0, 1.5, 2])

        npt.assert_allclose(thresh.hard_thresh(1, x), y)

    def test_l0_proj(self):
        x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
        y = np.array([-2, -1.5, 0, 0, 0, 0, 0, 1.5, 2])

        npt.assert_allclose(thresh.l0_proj(4, x), y)

    def test_elitist_thresh(self):
        x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])

        lamda = 1
        # Subgradient method
        u = np.zeros(len(x))
        for i in range(1000):
            alpha = 1 / (i + 1)
            u = u - alpha * (u - x + lamda * np.linalg.norm(u, 1) * np.sign(u))

        npt.assert_allclose(thresh.elitist_thresh(lamda, x), u,
                            atol=1e-2, rtol=1e-2)

        lamda = 1e-5
        # Subgradient method
        u = np.zeros(len(x))
        for i in range(1000):
            alpha = 1 / (i + 1)
            u = u - alpha * (u - x + lamda * np.linalg.norm(u, 1) * np.sign(u))

        npt.assert_allclose(thresh.elitist_thresh(lamda, x), u,
                            atol=1e-2, rtol=1e-2)

        lamda = 100
        # Subgradient method
        u = np.zeros(len(x))
        for i in range(1000):
            alpha = 1 / (i + 1)
            u = u - alpha * (u - x + lamda * np.linalg.norm(u, 1) * np.sign(u))

        npt.assert_allclose(thresh.elitist_thresh(lamda, x), u,
                            atol=1e-2, rtol=1e-2)

    if config.cupy_enabled:

        def test_soft_thresh_cuda(self):
            x = cp.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
            y = cp.array([-1, -0.5, 0, 0, 0, 0, 0, 0.5, 1])
            lamda = cp.array([1.0])

            cp.testing.assert_allclose(thresh.soft_thresh(lamda, x), y)

        def test_hard_thresh_cuda(self):
            x = cp.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
            y = cp.array([-2, -1.5, 0, 0, 0, 0, 0, 1.5, 2])
            lamda = cp.array([1.0])

            cp.testing.assert_allclose(thresh.hard_thresh(lamda, x), y)

        def test_elitist_thresh_cuda(self):
            lamda = 1.0
            x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
            y = thresh.elitist_thresh(lamda, x)

            y_cuda = thresh.elitist_thresh(lamda, cp.array(x))

            cp.testing.assert_allclose(y, y_cuda,
                                       atol=1e-7, rtol=1e-7)

            lamda = 100.0
            y = thresh.elitist_thresh(lamda, x)

            y_cuda = thresh.elitist_thresh(lamda, cp.array(x))

            cp.testing.assert_allclose(y, y_cuda,
                                       atol=1e-7, rtol=1e-7)
