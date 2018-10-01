import unittest
import numpy as np
from sigpy import util, interp, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestInterp(unittest.TestCase):

    def test_interpolate(self):

        batch = 2

        for ndim in [1, 2, 3]:
            shape = [3] + [1] * (ndim - 1)
            width = 2.0
            kernel = np.array([1.0, 0.5])
            coord = np.array([[0.1] + [0] * (ndim - 1),
                              [1.1] + [0] * (ndim - 1),
                              [2.1] + [0] * (ndim - 1)])

            input = np.array([[0, 1.0, 0]] * batch).reshape([batch] + shape)
            output = interp.interpolate(input, width, kernel, coord)
            output_expected = np.array([[0.1, 0.9, 0]] * batch)
            np.testing.assert_allclose(output, output_expected)

    def test_gridding(self):

        batch = 2

        for ndim in [1, 2, 3]:
            shape = [3] + [1] * (ndim - 1)
            width = 2.0
            kernel = np.array([1.0, 0.5])
            coord = np.array([[0.1] + [0] * (ndim - 1),
                              [1.1] + [0] * (ndim - 1),
                              [2.1] + [0] * (ndim - 1)])

            input = np.array([[0, 1.0, 0]] * batch)
            output = interp.gridding(
                input, [batch] + shape, width, kernel, coord)
            output_expected = np.array(
                [[0, 0.9, 0.1]] * batch).reshape([batch] + shape)
            np.testing.assert_allclose(output, output_expected)

    if config.cupy_enabled:

        import cupy as cp

        def test_lin_interpolate(self):

            lin_interpolate = cp.ElementwiseKernel('raw S kernel, S x', 'S y',
                                              'y = lin_interpolate(&kernel[0], kernel.size(), x)',
                                              preamble=interp.lin_interpolate_cuda)

            kernel = cp.array([0.0, 2.0])
            x = cp.array([0.5])
            cp.testing.assert_allclose(lin_interpolate(kernel, x), 2.0)

        def test_interpolate_cuda(self):

            batch = 2
            for ndim in [1, 2, 3]:
                for dtype in [np.float, np.complex64, np.complex]:
                    shape = [3] + [1] * (ndim - 1)
                    width = 2.0
                    kernel = cp.array([1.0, 0.5])
                    coord = cp.array([[0.1] + [0] * (ndim - 1),
                                      [1.1] + [0] * (ndim - 1),
                                      [2.1] + [0] * (ndim - 1)])

                    input = cp.array([[0, 1.0, 0]] * batch,
                                     dtype=dtype).reshape([batch] + shape)
                    output = interp.interpolate(input, width, kernel, coord)
                    output_expected = cp.array(
                        [[0.1, 0.9, 0]] * batch, dtype=dtype)
                    cp.testing.assert_allclose(
                        output, output_expected, atol=1e-7)

        def test_gridding_cuda(self):

            batch = 2
            for ndim in [1, 2, 3]:
                for dtype in [np.float, np.complex64, np.complex]:
                    shape = [3] + [1] * (ndim - 1)
                    width = 2.0
                    kernel = cp.array([1.0, 0.5])
                    coord = cp.array([[0.1] + [0] * (ndim - 1),
                                      [1.1] + [0] * (ndim - 1),
                                      [2.1] + [0] * (ndim - 1)])

                    input = cp.array([[0, 1.0, 0]] * batch, dtype=dtype)
                    output = interp.gridding(
                        input, [batch] + shape, width, kernel, coord)
                    output_expected = cp.array(
                        [[0, 0.9, 0.1]] * batch, dtype=dtype).reshape([batch] + shape)
                    cp.testing.assert_allclose(
                        output, output_expected, atol=1e-7)
