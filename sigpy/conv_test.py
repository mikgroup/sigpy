import unittest
import numpy as np
import numpy.testing as npt
from sigpy import backend, conv, util, config

if config.cudnn_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestConv(unittest.TestCase):

    def test_convolve_valid(self):
        mode = 'valid'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))
            
        for device in devices:
            xp = device.xp
            with device:
                for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    x = util.dirac([1, 3], device=device, dtype=dtype)
                    W = xp.ones([1, 3], dtype=dtype)
                    y = backend.to_device(conv.convolve(x, W, mode=mode), backend.cpu_device)
                    npt.assert_allclose(y, [[1]], atol=1e-5)

                    x = util.dirac([1, 3], device=device, dtype=dtype)
                    W = xp.ones([1, 2], dtype=dtype)
                    y = backend.to_device(conv.convolve(x, W, mode=mode), backend.cpu_device)
                    npt.assert_allclose(y, [[1, 1]], atol=1e-5)

                    x = util.dirac([1, 3], device=device, dtype=dtype)
                    W = xp.ones([2, 1, 3], dtype=dtype)
                    y = backend.to_device(conv.convolve(x, W, mode=mode,
                                                     output_multi_channel=True), backend.cpu_device)
                    npt.assert_allclose(y, [[[1]],
                                            [[1]]], atol=1e-5)

    def test_convolve_full(self):
        mode = 'full'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))
            
        for device in devices:
            xp = device.xp
            with device:
                for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    x = util.dirac([1, 3], device=device, dtype=dtype)
                    W = xp.ones([1, 3], dtype=dtype)
                    y = backend.to_device(conv.convolve(x, W, mode=mode), backend.cpu_device)
                    npt.assert_allclose(y, [[0, 1, 1, 1, 0]], atol=1e-5)

                    x = util.dirac([1, 3], device=device, dtype=dtype)
                    W = xp.ones([1, 2], dtype=dtype)
                    y = backend.to_device(conv.convolve(x, W, mode=mode), backend.cpu_device)
                    npt.assert_allclose(y, [[0, 1, 1, 0]], atol=1e-5)

                    x = util.dirac([1, 3], device=device, dtype=dtype)
                    W = xp.ones([2, 1, 3], dtype=dtype)
                    y = backend.to_device(conv.convolve(x, W, mode=mode,
                                                     output_multi_channel=True), backend.cpu_device)
                    npt.assert_allclose(y, [[[0, 1, 1, 1, 0]],
                                            [[0, 1, 1, 1, 0]]], atol=1e-5)

    def test_convolve_adjoint_input_valid(self):
        mode = 'valid'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))
            
        for device in devices:
            xp = device.xp
            with device:
                for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    y = xp.ones([1, 1], dtype=dtype)
                    W = xp.ones([1, 3], dtype=dtype)
                    x = backend.to_device(conv.convolve_adjoint_input(W, y, mode=mode), backend.cpu_device)
                    npt.assert_allclose(x, [[1, 1, 1]], atol=1e-5)

                    y = xp.ones([1, 2], dtype=dtype)
                    W = xp.ones([1, 2], dtype=dtype)
                    x = backend.to_device(conv.convolve_adjoint_input(W, y, mode=mode), backend.cpu_device)
                    npt.assert_allclose(x, [[1, 2, 1]], atol=1e-5)

                    y = xp.ones([2, 1, 1], dtype=dtype)
                    W = xp.ones([2, 1, 3], dtype=dtype)
                    x = backend.to_device(conv.convolve_adjoint_input(W, y, mode=mode,
                                                                   output_multi_channel=True), backend.cpu_device)
                    npt.assert_allclose(x, [[2, 2, 2]], atol=1e-5)

    def test_convolve_adjoint_input_full(self):
        mode = 'full'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))
            
        for device in devices:
            xp = device.xp
            with device:
                for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    y = xp.ones([1, 5], dtype=dtype)
                    W = xp.ones([1, 3], dtype=dtype)
                    x = backend.to_device(conv.convolve_adjoint_input(W, y, mode=mode), backend.cpu_device)
                    npt.assert_allclose(x, [[3, 3, 3]], atol=1e-5)

                    y = xp.ones([1, 4], dtype=dtype)
                    W = xp.ones([1, 2], dtype=dtype)
                    x = backend.to_device(conv.convolve_adjoint_input(W, y, mode=mode),
                                       backend.cpu_device)
                    npt.assert_allclose(x, [[2, 2, 2]], atol=1e-5)

                    y = xp.ones([2, 1, 5], dtype=dtype)
                    W = xp.ones([2, 1, 3], dtype=dtype)
                    x = backend.to_device(conv.convolve_adjoint_input(W, y, mode=mode,
                                                                   output_multi_channel=True),
                                       backend.cpu_device)
                    npt.assert_allclose(x, [[6, 6, 6]], atol=1e-5)

    def test_convolve_adjoint_filter_valid(self):
        mode = 'valid'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        ndim = 2
        for device in devices:
            xp = device.xp
            with device:
                for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    x = xp.ones([1, 3], dtype=dtype)
                    y = xp.ones([1, 1], dtype=dtype)
                    W = backend.to_device(conv.convolve_adjoint_filter(x, y, ndim, mode=mode),
                                       backend.cpu_device)
                    npt.assert_allclose(W, [[1, 1, 1]], atol=1e-5)

                    x = xp.ones([1, 3], dtype=dtype)
                    y = xp.ones([1, 2], dtype=dtype)
                    W = backend.to_device(conv.convolve_adjoint_filter(x, y, ndim, mode=mode),
                                       backend.cpu_device)
                    npt.assert_allclose(W, [[2, 2]], atol=1e-5)

                    x = xp.ones([1, 1, 3], dtype=dtype)
                    y = xp.ones([2, 1, 1], dtype=dtype)
                    W = backend.to_device(conv.convolve_adjoint_filter(x, y, ndim, mode=mode,
                                                                    output_multi_channel=True),
                                       backend.cpu_device)
                    npt.assert_allclose(W, [[[1, 1, 1]],
                                            [[1, 1, 1]]], atol=1e-5)

    def test_convolve_adjoint_filter_full(self):
        mode = 'full'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        ndim = 2
        for device in devices:
            xp = device.xp
            with device:
                for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    x = xp.ones([1, 3], dtype=dtype)
                    y = xp.ones([1, 5], dtype=dtype)
                    W = backend.to_device(conv.convolve_adjoint_filter(x, y, ndim, mode=mode),
                                       backend.cpu_device)
                    npt.assert_allclose(W, [[3, 3, 3]], atol=1e-5)

                    x = xp.ones([1, 3], dtype=dtype)
                    y = xp.ones([1, 4], dtype=dtype)
                    W = backend.to_device(conv.convolve_adjoint_filter(x, y, ndim, mode=mode),
                                       backend.cpu_device)
                    npt.assert_allclose(W, [[3, 3]], atol=1e-5)

                    x = xp.ones([1, 1, 3], dtype=dtype)
                    y = xp.ones([2, 1, 5], dtype=dtype)
                    W = backend.to_device(conv.convolve_adjoint_filter(x, y, ndim, mode=mode,
                                                                    output_multi_channel=True),
                                       backend.cpu_device)
                    npt.assert_allclose(W, [[[3, 3, 3]],
                                            [[3, 3, 3]]], atol=1e-5)
