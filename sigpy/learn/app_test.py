import unittest
import numpy as np
import numpy.testing as npt
from sigpy.learn import app

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def test_ConvSparseDecom(self):

        lamda = 0.1
        filt = np.array([[1 , 1],
                        [1, -1]], dtype=np.float) / 2**0.5
        data = np.array([[1, 1]], dtype=np.float) / 2**0.5

        coef = app.ConvSparseDecom(data, filt, lamda=lamda).run()

        npt.assert_allclose(coef, [[[0.9], [0]]])

    def test_ConvSparseCoefficients(self):

        lamda = 0.1
        filt = np.array([[1 , 1],
                        [1, -1]], dtype=np.float) / 2**0.5
        data = np.array([[1, 1]], dtype=np.float) / 2**0.5

        coef = app.ConvSparseCoefficients(data, filt, lamda=lamda)
        
        npt.assert_allclose(coef[:], [[[0.9], [0]]])        
        npt.assert_allclose(coef[0, :], [[0.9], [0]])
        npt.assert_allclose(coef[:, 0], [[0.9]])
        npt.assert_allclose(coef[:, :, 0], [[0.9, 0]])
        

    def test_ConvSparseCoding(self):

        num_atoms = 1
        filt_width = 2
        batch_size = 1
        data = np.array([[1, 1]], dtype=np.float) / 2**0.5

        filt = app.ConvSparseCoding(data, num_atoms, filt_width, batch_size, max_iter=10).run()

        npt.assert_allclose(np.abs(filt), [[1 / 2**0.5, 1 / 2**0.5]])

    def test_LinearRegression(self):

        n = 2
        k = 5
        m = 4
        batch_size = n

        coef = np.random.randn(n, k)
        data = np.random.randn(n, m)
        
        alpha = 1 / np.linalg.svd(coef, compute_uv=False)[0]**2

        mat = app.LinearRegression(coef, data, batch_size, alpha).run()
        
        mat_lstsq = np.linalg.lstsq(coef, data)[0]

        npt.assert_allclose(mat, mat_lstsq, atol=1e-3, rtol=1e-3)
