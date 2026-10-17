import unittest

import numpy as np
import numpy.testing as npt

from sigpy import config, thresh

if config.cupy_enabled:
    import cupy as cp

if __name__ == "__main__":
    unittest.main()


class TestThresh(unittest.TestCase):
    def test_l2_proj(self):
        x = np.ones(5)
        y = np.full(5, 1 / 5**0.5)
        npt.assert_allclose(thresh.l2_proj(1, x), y)

        x = np.ones(5)
        y = np.ones(5)
        npt.assert_allclose(thresh.l2_proj(5**0.5, x), y)

        x = np.ones(5)
        y = np.ones(5)
        npt.assert_allclose(thresh.l2_proj(10, x), y)

    def test_linf_proj(self):
        x = np.ones(5)
        y = np.ones(5)
        npt.assert_allclose(thresh.linf_proj(1.1, x), y)

        x = np.ones(5)
        y = np.ones(5) * 0.1
        npt.assert_allclose(thresh.linf_proj(0.1, x), y)

        x = [-2, -1, 0, 1, 2]
        y = [-1, -1, 0, 1, 1]
        npt.assert_allclose(thresh.linf_proj(1, x), y)

    def test_psd_proj(self):
        x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -2]])
        y = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        npt.assert_allclose(thresh.psd_proj(x), y)

    def test_psd_proj_repeated_eigenvalues(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        for xp in xps:
            for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                for sign in [-1, 1]:
                    with self.subTest(xp=xp, dtype=dtype, sign=sign):
                        # Two eigenvalues are 1; the third is 1 + 3 * sign.
                        x = xp.eye(3, dtype=dtype) + sign * xp.ones(
                            (3, 3), dtype=dtype
                        )
                        expected = (
                            x.copy()
                            if sign > 0
                            else (
                                xp.eye(3, dtype=dtype)
                                - xp.ones((3, 3), dtype=dtype) / 3
                            )
                        )
                        if np.issubdtype(dtype, np.complexfloating):
                            phase = xp.array([1, 1j, -1], dtype=dtype)
                            x = phase[:, None] * x * phase.conj()[None, :]
                            expected = (
                                phase[:, None]
                                * expected
                                * phase.conj()[None, :]
                            )

                        xp.testing.assert_allclose(
                            thresh.psd_proj(x), expected, atol=1e-6, rtol=1e-6
                        )

    def test_soft_thresh(self):
        x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
        y = np.array([-1, -0.5, 0, 0, 0, 0, 0, 0.5, 1])

        npt.assert_allclose(thresh.soft_thresh(1, x), y)

    def test_hard_thresh(self):
        x = np.array([-2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2])
        y = np.array([-2, -1.5, 0, 0, 0, 0, 0, 1.5, 2])

        npt.assert_allclose(thresh.hard_thresh(1, x), y)

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
