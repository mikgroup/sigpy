import unittest
import numpy as np
import numpy.testing as npt
from sigpy.learn import app

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def test_ConvSparseDecom(self):

        lamda = 0.1

        dic = np.array([[1 , 1],
                        [1, -1]], dtype=np.float) / 2**0.5
        dat = np.array([[1, 1]], dtype=np.float) / 2**0.5

        fea = app.ConvSparseDecom(dat, dic, lamda=lamda).run()

        npt.assert_allclose(fea, [[[0.9], [0]]])


    def test_ConvSparseCoding(self):

        num_atoms = 1
        dic_width = 2
        batch_size = 1
        dat = np.array([[1, 1]], dtype=np.float) / 2**0.5

        dic = app.ConvSparseCoding(dat, num_atoms, dic_width, batch_size, max_iter=10).run()

        npt.assert_allclose(np.abs(dic), [[1 / 2**0.5, 1 / 2**0.5]])

    def test_LinearRegression(self):

        n = 2
        k = 5
        m = 4
        batch_size = n

        fea = np.random.randn(n, k)
        dat = np.random.randn(n, m)
        
        alpha = 1 / np.linalg.svd(fea, compute_uv=False)[0]**2

        mat = app.LinearRegression(fea, dat, batch_size, alpha).run()
        
        mat_lstsq = np.linalg.lstsq(fea, dat)[0]

        npt.assert_allclose(mat, mat_lstsq)
