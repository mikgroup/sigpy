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

        dic = app.ConvSparseCoding(dat, num_atoms, dic_width, batch_size).run()

        npt.assert_allclose(np.abs(dic), [[1 / 2**0.5, 1 / 2**0.5]])
