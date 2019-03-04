import unittest
import numpy as np
from sigpy import interp, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestInterp(unittest.TestCase):

    def test_interpolate(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        batch = 2
        for xp in xps:
            for ndim in [1, 2, 3]:
                with self.subTest(ndim=ndim, xp=xp):
                    shape = [3] + [1] * (ndim - 1)
                    width = 2.0
                    kernel = xp.array([1.0, 0.5])
                    coord = xp.array([[0.1] + [0] * (ndim - 1),
                                      [1.1] + [0] * (ndim - 1),
                                      [2.1] + [0] * (ndim - 1)])

                    input = xp.array([[0, 1.0, 0]] * batch).reshape(
                        [batch] + shape)
                    output = interp.interpolate(input, width, kernel, coord)
                    output_expected = xp.array([[0.1, 0.9, 0]] * batch)
                    xp.testing.assert_allclose(output, output_expected,
                                               atol=1e-7)

    def test_gridding(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        batch = 2
        for xp in xps:
            for ndim in [1, 2, 3]:
                with self.subTest(ndim=ndim, xp=xp):
                    shape = [3] + [1] * (ndim - 1)
                    width = 2.0
                    kernel = xp.array([1.0, 0.5])
                    coord = xp.array([[0.1] + [0] * (ndim - 1),
                                      [1.1] + [0] * (ndim - 1),
                                      [2.1] + [0] * (ndim - 1)])

                    input = xp.array([[0, 1.0, 0]] * batch)
                    output = interp.gridding(
                        input, [batch] + shape, width, kernel, coord)
                    output_expected = xp.array(
                        [[0, 0.9, 0.1]] * batch).reshape([batch] + shape)
                    xp.testing.assert_allclose(output, output_expected,
                                               atol=1e-7)
