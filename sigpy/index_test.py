import unittest
import numpy as np
import numpy.testing as npt
from sigpy import index

if __name__ == '__main__':
    unittest.main()


class TestIndex(unittest.TestCase):

    def test_ShuffledIndex(self):

        n = 5
        idx = index.ShuffledIndex(n)

        x = sorted([idx.next() for _ in range(2 * n)])

        npt.assert_allclose(x, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_PingPongIndex(self):

        n = 5
        idx = index.PingPongIndex(n)

        x = [idx.next() for _ in range(2 * n)]

        npt.assert_allclose(x, [0, 1, 2, 3, 4, 4, 3, 2, 1, 0])
