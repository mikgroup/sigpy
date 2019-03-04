import pickle
import unittest
import numpy as np
import numpy.testing as npt
from sigpy import util, backend

if __name__ == '__main__':
    unittest.main()


class TestUtil(unittest.TestCase):

    def test_device(self):
        device = backend.Device(-1)
        pickle.dumps(device)

    def test_dirac(self):
        output = util.dirac([5])
        truth = [0, 0, 1, 0, 0]
        npt.assert_allclose(output, truth)

        output = util.dirac([4])
        truth = [0, 0, 1, 0]
        npt.assert_allclose(output, truth)

    def test_triang(self):
        npt.assert_allclose(util.triang([3]), [0.5, 1, 0.5])
        npt.assert_allclose(util.triang([4]), [0.25, 0.75, 0.75, 0.25])

    def test_hanning(self):
        npt.assert_allclose(util.hanning([4]), [0, 0.5, 1, 0.5])
        npt.assert_allclose(util.hanning([5]), [0, 0.5, 1, 0.5, 0])

    def test_resize(self):
        # Zero-pad
        x = np.array([1, 2, 3])
        oshape = [5]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [0, 1, 2, 3, 0])

        x = np.array([1, 2, 3])
        oshape = [4]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [0, 1, 2, 3])

        x = np.array([1, 2])
        oshape = [5]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [0, 1, 2, 0, 0])

        x = np.array([1, 2])
        oshape = [4]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [0, 1, 2, 0])

        # Zero-pad non centered
        x = np.array([1, 2, 3])
        oshape = [5]
        y = util.resize(x, oshape, oshift=[0])
        npt.assert_allclose(y, [1, 2, 3, 0, 0])

        # Crop
        x = np.array([0, 1, 2, 3, 0])
        oshape = [3]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [1, 2, 3])

        x = np.array([0, 1, 2, 3])
        oshape = [3]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [1, 2, 3])

        x = np.array([0, 1, 2, 0, 0])
        oshape = [2]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [1, 2])

        x = np.array([0, 1, 2, 0])
        oshape = [2]
        y = util.resize(x, oshape)
        npt.assert_allclose(y, [1, 2])

        # Crop non centered
        x = np.array([1, 2, 3, 0, 0])
        oshape = [3]
        y = util.resize(x, oshape, ishift=[0])
        npt.assert_allclose(y, [1, 2, 3])

    def test_downsample(self):
        x = np.array([1, 2, 3, 4, 5])
        y = util.downsample(x, [2])
        npt.assert_allclose(y, [1, 3, 5])

    def test_upsample(self):
        x = np.array([1, 2, 3])
        y = util.upsample(x, [5], [2])
        npt.assert_allclose(y, [1, 0, 2, 0, 3])

    def test_circshift(self):
        input = np.array([0, 1, 2, 3])
        axes = [0]
        shift = [1]
        npt.assert_allclose(util.circshift(input, shift, axes),
                            [3, 0, 1, 2])

        input = np.array([[0, 1, 2],
                          [3, 4, 5]])
        axes = [-1]
        shift = [2]
        npt.assert_allclose(util.circshift(input, shift, axes),
                            [[1, 2, 0],
                             [4, 5, 3]])

        input = np.array([[0, 1, 2],
                          [3, 4, 5]])
        axes = [-2]
        shift = [1]
        npt.assert_allclose(util.circshift(input, shift, axes),
                            [[3, 4, 5],
                             [0, 1, 2]])

    def test_monte_carlo_sure(self):
        x = np.ones([100000], dtype=np.float)
        sigma = 0.1
        noise = 0.1 * util.randn([100000], dtype=np.float)
        y = x + noise

        def f(y):
            return y

        npt.assert_allclose(
            sigma**2, util.monte_carlo_sure(f, y, sigma), atol=1e-3)

    def test_ShuffledNumbers(self):
        n = 5
        idx = util.ShuffledNumbers(n)
        x = sorted([idx.next() for _ in range(2 * n)])
        npt.assert_allclose(x, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
