import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
from sigpy.mri import linop, precond

if __name__ == "__main__":
    unittest.main()


class TestPrecond(unittest.TestCase):
    def test_kspace_precond_cart(self):
        nc = 4
        n = 10
        shape = (nc, n)
        mps = sp.randn(shape, dtype=complex)
        mps /= np.linalg.norm(mps, axis=0, keepdims=True)
        weights = sp.randn([n]) >= 0

        A = sp.linop.Multiply(shape, weights**0.5) * linop.Sense(mps)

        AAH = np.zeros((nc, n, nc, n), complex)
        for d in range(nc):
            for j in range(n):
                x = np.zeros((nc, n), complex)
                x[d, j] = 1.0
                AAHx = A(A.H(x))

                for c in range(nc):
                    for i in range(n):
                        AAH[c, i, d, j] = AAHx[c, i]

        p_expected = np.ones((nc, n), complex)
        for c in range(nc):
            for i in range(n):
                if weights[i]:
                    p_expected_inv_ic = 0
                    for d in range(nc):
                        for j in range(n):
                            p_expected_inv_ic += abs(
                                AAH[c, i, d, j]
                            ) ** 2 / abs(AAH[c, i, c, i])

                    p_expected[c, i] = 1 / p_expected_inv_ic

        p = precond.kspace_precond(mps, weights=weights)
        npt.assert_allclose(
            p[:, weights == 1],
            p_expected[:, weights == 1],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_kspace_precond_noncart(self):
        n = 10
        nc = 3
        shape = [nc, n]
        mps = sp.randn(shape, dtype=complex)
        mps /= np.linalg.norm(mps, axis=0, keepdims=True)
        coord = sp.randn([n, 1], dtype=float)

        A = linop.Sense(mps, coord=coord)

        AAH = np.zeros((nc, n, nc, n), complex)
        for d in range(nc):
            for j in range(n):
                x = np.zeros(shape, complex)
                x[d, j] = 1.0
                AAHx = A(A.H(x))
                for c in range(nc):
                    for i in range(n):
                        AAH[c, i, d, j] = AAHx[c, i]

        p_expected = np.zeros([nc, n], complex)
        for c in range(nc):
            for i in range(n):
                p_expected_inv_ic = 0
                for d in range(nc):
                    for j in range(n):
                        p_expected_inv_ic += abs(AAH[c, i, d, j]) ** 2 / abs(
                            AAH[c, i, c, i]
                        )

                p_expected[c, i] = 1 / p_expected_inv_ic

        p = precond.kspace_precond(mps, coord=coord)
        npt.assert_allclose(p, p_expected, atol=1e-2, rtol=1e-2)

    def test_kspace_precond_simple_cart(self):
        # Check identity
        mps_shape = [1, 1]
        mps = np.ones(mps_shape, dtype=complex)
        p = precond.kspace_precond(mps)
        npt.assert_allclose(p, np.ones(mps_shape), atol=1e-6, rtol=1e-6)

        # Check scaling
        mps_shape = [1, 3]
        mps = np.ones(mps_shape, dtype=complex)
        p = precond.kspace_precond(mps)
        npt.assert_allclose(p, np.ones(mps_shape), atol=1e-6, rtol=1e-6)

        # Check 2d
        mps_shape = [1, 3, 3]
        mps = np.ones(mps_shape, dtype=complex)
        p = precond.kspace_precond(mps)
        npt.assert_allclose(p, np.ones(mps_shape), atol=1e-6, rtol=1e-6)

        # Check weights
        mps_shape = [1, 3]
        mps = np.ones(mps_shape, dtype=complex)
        weights = np.array([1, 0, 1], dtype=complex)
        p = precond.kspace_precond(mps, weights=weights)
        npt.assert_allclose(p, [[1, 1, 1]], atol=1e-6, rtol=1e-6)

    def test_kspace_precond_simple_noncart(self):
        # Check identity
        mps_shape = [1, 1]

        mps = np.ones(mps_shape, dtype=complex)
        coord = np.array([[0.0]])
        p = precond.kspace_precond(mps, coord=coord)
        npt.assert_allclose(p, [[1.0]], atol=1, rtol=1e-1)

        mps_shape = [1, 3]

        mps = np.ones(mps_shape, dtype=complex)
        coord = np.array([[0.0], [-1], [1]])
        p = precond.kspace_precond(mps, coord=coord)
        npt.assert_allclose(p, [[1.0, 1.0, 1.0]], atol=1, rtol=1e-1)

    def test_circulant_precond_cart(self):
        nc = 4
        n = 10
        shape = (nc, n)
        mps = sp.randn(shape, dtype=complex)
        mps /= np.linalg.norm(mps, axis=0, keepdims=True)
        weights = sp.randn([n]) >= 0

        A = sp.linop.Multiply(shape, weights**0.5) * linop.Sense(mps)
        F = sp.linop.FFT([n])

        p_expected = np.zeros(n, complex)
        for i in range(n):
            if weights[i]:
                x = np.zeros(n, complex)
                x[i] = 1.0
                p_expected[i] = 1 / F(A.H(A(F.H(x))))[i]

        p = precond.circulant_precond(mps, weights=weights)
        npt.assert_allclose(
            p[weights == 1], p_expected[weights == 1], atol=1e-6, rtol=1e-6
        )

    def test_circulant_precond_noncart(self):
        nc = 4
        n = 10
        shape = [nc, n]
        mps = np.ones(shape, dtype=complex)
        mps /= np.linalg.norm(mps, axis=0, keepdims=True)
        coord = sp.randn([n, 1], dtype=float)

        A = linop.Sense(mps, coord=coord)
        F = sp.linop.FFT([n])

        p_expected = np.zeros(n, complex)
        for i in range(n):
            x = np.zeros(n, complex)
            x[i] = 1.0
            p_expected[i] = 1 / F(A.H(A(F.H(x))))[i]

        p = precond.circulant_precond(mps, coord=coord)
        npt.assert_allclose(p, p_expected, atol=1e-1, rtol=1e-1)
