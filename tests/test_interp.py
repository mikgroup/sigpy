import unittest
import numpy as np
from sigpy import interp, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestInterp(unittest.TestCase):

    def test_interpolate_spline(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        batch = 2
        for xp in xps:
            for ndim in [1, 2, 3]:
                for dtype in [np.float32, np.complex64]:
                    with self.subTest(ndim=ndim, xp=xp, dtype=dtype):
                        shape = [3] + [1] * (ndim - 1)
                        coord = xp.array([[0.1] + [0] * (ndim - 1),
                                          [1.1] + [0] * (ndim - 1),
                                          [2.1] + [0] * (ndim - 1)])

                        input = xp.array([[0, 1.0, 0]] * batch, dtype=dtype)
                        input = input.reshape([batch] + shape)
                        output = interp.interpolate(input, coord)
                        output_expected = xp.array([[0.1, 0.9, 0]] * batch)
                        xp.testing.assert_allclose(output, output_expected,
                                                   atol=1e-7)

    def test_gridding_spline(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        batch = 2
        for xp in xps:
            for ndim in [1, 2, 3]:
                for dtype in [np.float32, np.complex64]:
                    with self.subTest(ndim=ndim, xp=xp, dtype=dtype):
                        shape = [3] + [1] * (ndim - 1)
                        coord = xp.array([[0.1] + [0] * (ndim - 1),
                                          [1.1] + [0] * (ndim - 1),
                                          [2.1] + [0] * (ndim - 1)])

                        input = xp.array([[0, 1.0, 0]] * batch, dtype=dtype)
                        output = interp.gridding(input, coord, [batch] + shape)
                        output_expected = xp.array(
                            [[0, 0.9, 0.1]] * batch).reshape([batch] + shape)
                        xp.testing.assert_allclose(output, output_expected,
                                                   atol=1e-7)

    if config.cupy_enabled:
        def test_interpolate_cpu_gpu(self):
            for ndim in [1, 2, 3]:
                for dtype in [np.float32, np.complex64]:
                    with self.subTest(ndim=ndim, dtype=dtype):
                        shape = [2, 20] + [1] * (ndim - 1)
                        coord = np.random.random([10, ndim])

                        input = np.random.random(shape).astype(dtype=dtype)
                        output_cpu = interp.interpolate(
                            input, coord, kernel="kaiser_bessel")
                        output_gpu = interp.interpolate(
                            cp.array(input), cp.array(coord),
                            kernel="kaiser_bessel").get()
                        np.testing.assert_allclose(
                            output_cpu, output_gpu, atol=1e-5)

        def test_gridding_cpu_gpu(self):
            for ndim in [1, 2, 3]:
                for dtype in [np.float32, np.complex64]:
                    with self.subTest(ndim=ndim, dtype=dtype):
                        shape = [2, 20] + [1] * (ndim - 1)
                        coord = np.random.random([10, ndim])

                        input = np.random.random(
                            [2, 10] + [1] * (ndim - 1)).astype(dtype=dtype)
                        output_cpu = interp.gridding(
                            input, coord, shape, kernel="kaiser_bessel")
                        output_gpu = interp.gridding(
                            cp.array(input), cp.array(coord), shape,
                            kernel="kaiser_bessel").get()
                        np.testing.assert_allclose(
                            output_cpu, output_gpu, atol=1e-5)
