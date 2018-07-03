import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import linop, precond

if __name__ == '__main__':
    unittest.main()


class TestPrecond(unittest.TestCase):

    def test_sense_kspace_precond(self):
        nc = 4
        nx = 14
        shape = (nc, nx)
        mps = sp.util.randn(shape)
        mps /= np.linalg.norm(mps, axis=0, keepdims=True)
        weights = sp.util.randn([nx]) >= 0

        A = sp.linop.Multiply(shape, weights**0.5) * linop.Sense(mps)

        AAH = np.zeros((nc, nx, nc, nx), np.complex)
        for d in range(nc):
            for j in range(nx):
                x = np.zeros((nc, nx), np.complex)
                x[d, j] = 1.0
                AAHx = A(A.H(x))

                for c in range(nc):
                    for i in range(nx):
                        AAH[c, i, d, j] = AAHx[c, i]

        npt.assert_allclose(AAH, AAH.transpose(
            [2, 3, 0, 1]).conjugate(), atol=1e-7)

        density = np.zeros((nc, nx), np.complex)
        for c in range(nc):
            for i in range(nx):
                for d in range(nc):
                    for j in range(nx):
                        density[c, i] += abs(AAH[c, i, d, j])**2 / \
                            (abs(AAH[c, i, c, i]) + 1e-11)

        pre = precond.sense_kspace_precond(mps, weights=weights)

        npt.assert_allclose(1.0 / pre[pre != 1], density[density != 0])

    def test_sense_kspace_precond_noncart(self):
        n = 10
        shape = [1, n]
        mps = sp.util.ones(shape)
        coord = sp.util.randn([n, 1], dtype=np.float)

        A = linop.Sense(mps, coord=coord)

        AAH = np.zeros((n, n), np.complex)
        for j in range(n):
            x = np.zeros(shape, np.complex)
            x[0, j] = 1.0
            AAHx = A(A.H(x))
            for i in range(n):
                AAH[i, j] = AAHx[0, i]

        density = np.zeros([n], np.complex)
        for i in range(n):
            for j in range(n):
                density[i] += abs(AAH[i, j])**2 / (abs(AAH[i, i]) + 1e-11)

        pre = precond.sense_kspace_precond(mps, coord=coord)[0]
        npt.assert_allclose(1.0 / pre, density, atol=1e-2, rtol=1e-2)

    def test_sense_kspace_precond_simple(self):
        # Check identity
        mps_shape = [1, 1]

        mps = np.ones(mps_shape, dtype=np.complex)

        pre = precond.sense_kspace_precond(mps)

        npt.assert_allclose(pre, np.ones(mps_shape))

        # Check scaling
        mps_shape = [1, 3]

        mps = np.ones(mps_shape, dtype=np.complex)

        pre = precond.sense_kspace_precond(mps)

        npt.assert_allclose(pre, np.ones(mps_shape))

        # Check 2d
        mps_shape = [1, 3, 3]

        mps = np.ones(mps_shape, dtype=np.complex)

        pre = precond.sense_kspace_precond(mps)

        npt.assert_allclose(pre, np.ones(mps_shape))

        # Check weights
        mps_shape = [1, 3]

        mps = np.ones(mps_shape, dtype=np.complex)
        weights = np.array([1, 0, 1], dtype=np.complex)

        pre = precond.sense_kspace_precond(mps, weights=weights)

        npt.assert_allclose(pre, [[1, 1, 1]])

    def test_sense_kspace_precond_simple_noncart(self):
        # Check identity
        mps_shape = [1, 1]

        mps = np.ones(mps_shape, dtype=np.complex)
        coord = np.array([[0.0]])

        pre = precond.sense_kspace_precond(mps, coord=coord)

        npt.assert_allclose(pre, [[1.0]], atol=1, rtol=1e-1)

        mps_shape = [1, 3]

        mps = np.ones(mps_shape, dtype=np.complex)
        coord = np.array([[0.0], [-1], [1]])

        pre = precond.sense_kspace_precond(mps, coord=coord)

        npt.assert_allclose(pre, [[1.0, 1.0, 1.0]], atol=1, rtol=1e-1)
