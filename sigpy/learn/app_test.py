import unittest
import numpy as np
import numpy.testing as npt
from sigpy.learn import app

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def test_ConvSparseDecom(self):
        lamda = 1e-9
        L = np.array([[1 , 1],
                      [1, -1]], dtype=np.float) / 2**0.5
        y = np.array([[1, 1]], dtype=np.float) / 2**0.5

        R = app.ConvSparseDecom(y, L, lamda=lamda).run()

        npt.assert_allclose(R, [[[1], [0]]])

    def test_ConvSparseCoefficients(self):
        lamda = 1e-10
        L = np.array([[1 , 1],
                      [1, -1]], dtype=np.float) / 2**0.5
        y = np.array([[1, 1]], dtype=np.float) / 2**0.5

        R_j = app.ConvSparseCoefficients(y, L, lamda=lamda)
        
        npt.assert_allclose(R_j[:], [[[1], [0]]])        
        npt.assert_allclose(R_j[0, :], [[1], [0]])
        npt.assert_allclose(R_j[:, 0], [[1]])
        npt.assert_allclose(R_j[:, :, 0], [[1, 0]])
        

    def test_ConvSparseCoding(self):
        num_atoms = 1
        filt_width = 2
        batch_size = 1
        y = np.array([[1, 1]], dtype=np.float) / 2**0.5
        lamda = 1e-10
        alpha = 1

        L, _ = app.ConvSparseCoding(y, num_atoms, filt_width, batch_size,
                                    alpha=alpha, lamda=lamda, max_iter=100).run()

        npt.assert_allclose(np.abs(L), [[1 / 2**0.5, 1 / 2**0.5]], atol=0.1, rtol=0.1)

    def test_LinearRegression(self):
        n = 2
        k = 5
        m = 4
        batch_size = n

        X = np.random.randn(n, k)
        y = np.random.randn(n, m)
        
        alpha = 1 / np.linalg.svd(X, compute_uv=False)[0]**2
        mat = app.LinearRegression(X, y, batch_size, alpha, max_iter=300).run()
        mat_lstsq = np.linalg.lstsq(X, y, rcond=-1)[0]
        
        npt.assert_allclose(mat, mat_lstsq, atol=1e-2, rtol=1e-2)
