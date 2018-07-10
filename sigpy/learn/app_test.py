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
