import unittest
import numpy as np
import numpy.testing as npt
from sigpy.learn import app

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def test_ConvSparseDecom(self):
        lamda = 1e-9
        l = np.array([[1 , 1],
                      [1, -1]], dtype=np.float) / 2**0.5
        y = np.array([[1, 1]], dtype=np.float) / 2**0.5

        r_j = app.ConvSparseDecom(y, l, lamda=lamda).run()

        npt.assert_allclose(r_j, [[[1], [0]]])

    def test_ConvSparseCoefficients(self):

        lamda = 1e-10
        l = np.array([[1 , 1],
                      [1, -1]], dtype=np.float) / 2**0.5
        y = np.array([[1, 1]], dtype=np.float) / 2**0.5

        r_j = app.ConvSparseCoefficients(y, l, lamda=lamda)
        
        npt.assert_allclose(r_j[:], [[[1], [0]]])        
        npt.assert_allclose(r_j[0, :], [[1], [0]])
        npt.assert_allclose(r_j[:, 0], [[1]])
        npt.assert_allclose(r_j[:, :, 0], [[1, 0]])
        

    def test_ConvSparseCoding(self):
        num_atoms = 1
        filt_width = 2
        batch_size = 1
        y = np.array([[1, 1]], dtype=np.float) / 2**0.5
        lamda = 1e-3
        alpha = np.infty

        l, r = app.ConvSparseCoding(y, num_atoms, filt_width, batch_size,
                                    alpha=alpha, lamda=lamda, max_epoch=10).run()

        npt.assert_allclose(np.abs(l), [[1 / 2**0.5, 1 / 2**0.5]])

    def test_LinearRegression(self):
        n = 2
        k = 5
        m = 4
        batch_size = n

        r = np.random.randn(n, k)
        y = np.random.randn(n, m)
        
        alpha = 1 / np.linalg.svd(r, compute_uv=False)[0]**2
        mat = app.LinearRegression(r, y, batch_size, alpha, max_epoch=100).run()
        mat_lstsq = np.linalg.lstsq(r, y, rcond=None)[0]
        
        npt.assert_allclose(mat, mat_lstsq, atol=1e-3, rtol=1e-3)
