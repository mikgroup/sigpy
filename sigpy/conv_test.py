import unittest
import numpy as np
import numpy.testing as npt
from scipy.signal import correlate, convolve
from sigpy import conv, util, config

if config.cudnn_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestConv(unittest.TestCase):

    def test_convolve(self):

        for mode in ['full', 'valid']:
            x1 = np.array([0, 1, 0], np.complex)
            x2 = np.array([1, 1, 1], np.complex)
            y = conv.convolve(x1, x2, mode=mode)
            npt.assert_allclose(y, convolve(x1, x2, mode=mode), atol=1e-10)

            x1 = np.array([[0, 1, 0]], np.complex)
            x2 = np.array([[1, 1, 1]] * 2, np.complex)
            y = conv.convolve(x1, x2, mode=mode)
            npt.assert_allclose(y, np.tile(convolve(x1[0], x2[0], mode=mode), [2, 1]),
                                atol=1e-10)

            x1 = util.randn([3, 4, 5], dtype=np.float)
            x2 = util.randn([2, 3, 4], dtype=np.float)
            y = conv.convolve(x1, x2, mode=mode)
            npt.assert_allclose(y, convolve(x1, x2, mode=mode), atol=1e-10)

            x1 = util.randn([3, 4, 5])
            x2 = util.randn([2, 3, 4])
            y = conv.convolve(x1, x2, mode=mode)
            npt.assert_allclose(y, convolve(x1, x2, mode=mode), atol=1e-10)

    def test_correlate(self):

        for mode in ['full', 'valid']:
            x1 = np.array([0, 1, 0], np.complex)
            x2 = np.array([1, 1, 1], np.complex)
            y = conv.correlate(x1, x2, mode=mode)
            npt.assert_allclose(y, correlate(x1, x2, mode=mode), atol=1e-10)

            x1 = util.randn([3, 4, 5], dtype=np.float)
            x2 = util.randn([2, 3, 4], dtype=np.float)
            y = conv.correlate(x1, x2, mode=mode)
            npt.assert_allclose(y, correlate(x1, x2, mode=mode), atol=1e-10)

            x1 = util.randn([3, 4, 5])
            x2 = util.randn([2, 3, 4])
            y = conv.correlate(x1, x2, mode=mode)
            npt.assert_allclose(y, correlate(x1, x2, mode=mode), atol=1e-10)

    if config.cudnn_enabled:

        def test_cudnn_convolve(self):

            for dtype in [np.float, np.complex]:
                for mode in ['valid', 'full']:

                    x = util.randn([2, 1, 3, 4], dtype=dtype, device=0)
                    W = util.randn([5, 1, 2, 3], dtype=dtype, device=0)

                    y = conv.convolve(x, W.reshape([5, 2, 3]), mode=mode)
                    cp.testing.assert_allclose(y, conv.cudnn_convolve(x, W, mode=mode),
                                               atol=1e-5, rtol=1e-5)

        def test_cudnn_convolve_backward_data(self):

            for dtype in [np.float, np.complex]:
                for mode in ['valid', 'full']:

                    x_shape = [2, 1, 3, 4]
                    W = util.randn([5, 1, 2, 3], dtype=dtype, device=0)
                    if mode == 'full':
                        y = util.randn([2, 5, 4, 6], dtype=dtype, device=0)
                    else:
                        y = util.randn([2, 5, 2, 2], dtype=dtype, device=0)

                    x = conv.cudnn_convolve_backward_data(W, y, mode=mode)

                    npt.assert_allclose(x_shape, x.shape)

        def test_cudnn_convolve_backward_filter(self):

            for dtype in [np.float, np.complex]:
                for mode in ['valid', 'full']:

                    x = util.randn([2, 1, 3, 4], dtype=dtype, device=0)
                    W_shape = [5, 1, 2, 3]
                    if mode == 'full':
                        y = util.randn([2, 5, 4, 6], dtype=dtype, device=0)
                    else:
                        y = util.randn([2, 5, 2, 2], dtype=dtype, device=0)

                    W = conv.cudnn_convolve_backward_filter(x, y, mode=mode)

                    npt.assert_allclose(W_shape, W.shape)
