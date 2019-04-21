import unittest
import numpy as np
import numpy.testing as npt
from sigpy import backend, conv, util, config


if __name__ == '__main__':
    unittest.main()

dtypes = [np.float32, np.float64,
          np.complex64, np.complex128]


class TestConv(unittest.TestCase):

    def test_convolve_valid(self):
        mode = 'valid'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for device in devices:
            xp = device.xp
            with device:
                for dtype in dtypes:
                    with self.subTest(dtype=dtype, device=device):
                        data = util.dirac([1, 3], device=device, dtype=dtype)
                        filt = xp.ones([1, 3], dtype=dtype)
                        output = backend.to_device(conv.convolve(
                            data, filt, mode=mode))
                        npt.assert_allclose(output, [[1]], atol=1e-5)

                        data = util.dirac([1, 3], device=device, dtype=dtype)
                        filt = xp.ones([1, 2], dtype=dtype)
                        output = backend.to_device(conv.convolve(
                            data, filt, mode=mode))
                        npt.assert_allclose(output, [[1, 1]], atol=1e-5)

                        data = util.dirac([1, 1, 3], device=device,
                                          dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        output = backend.to_device(
                            conv.convolve(data, filt,
                                          mode=mode,
                                          multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(output, [[[1]],
                                                     [[1]]], atol=1e-5)

                        data = util.dirac([1, 1, 3], device=device,
                                          dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        strides = [1, 2]
                        output = backend.to_device(
                            conv.convolve(data, filt,
                                          mode=mode, strides=strides,
                                          multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(output, [[[1]],
                                                     [[1]]], atol=1e-5)

    def test_convolve_full(self):
        mode = 'full'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))
            dtypes = [np.float32, np.float64,
                      np.complex64, np.complex128]

        for device in devices:
            xp = device.xp
            with device:
                for dtype in dtypes:
                    with self.subTest(dtype=dtype, device=device):
                        data = util.dirac([1, 3], device=device, dtype=dtype)
                        filt = xp.ones([1, 3], dtype=dtype)
                        output = backend.to_device(conv.convolve(
                            data, filt, mode=mode))
                        npt.assert_allclose(output, [[0, 1, 1, 1, 0]],
                                            atol=1e-5)

                        data = util.dirac([1, 3], device=device, dtype=dtype)
                        filt = xp.ones([1, 2], dtype=dtype)
                        output = backend.to_device(conv.convolve(
                            data, filt, mode=mode))
                        npt.assert_allclose(output, [[0, 1, 1, 0]], atol=1e-5)

                        data = util.dirac([1, 1, 3], device=device,
                                          dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        output = backend.to_device(
                            conv.convolve(data, filt,
                                          mode=mode,
                                          multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(output, [[[0, 1, 1, 1, 0]],
                                                     [[0, 1, 1, 1, 0]]],
                                            atol=1e-5)

                        data = util.dirac([1, 1, 3], device=device,
                                          dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        strides = [1, 2]
                        output = backend.to_device(
                            conv.convolve(data, filt,
                                          mode=mode,
                                          strides=strides,
                                          multi_channel=True))
                        npt.assert_allclose(output, [[[0, 1, 0]],
                                                     [[0, 1, 0]]], atol=1e-5)

    def test_convolve_adjoint_data_valid(self):
        mode = 'valid'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for device in devices:
            xp = device.xp
            with device:
                for dtype in dtypes:
                    with self.subTest(dtype=dtype, device=device):
                        output = xp.ones([1, 1], dtype=dtype)
                        filt = xp.ones([1, 3], dtype=dtype)
                        data_shape = [1, 3]
                        data = backend.to_device(conv.convolve_adjoint_data(
                            output, filt, data_shape, mode=mode))
                        npt.assert_allclose(data, [[1, 1, 1]], atol=1e-5)

                        output = xp.ones([1, 2], dtype=dtype)
                        filt = xp.ones([1, 2], dtype=dtype)
                        data_shape = [1, 3]
                        data = backend.to_device(conv.convolve_adjoint_data(
                            output, filt, data_shape, mode=mode))
                        npt.assert_allclose(data, [[1, 2, 1]], atol=1e-5)

                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 3]
                        data = backend.to_device(
                            conv.convolve_adjoint_data(
                                output, filt, data_shape,
                                mode=mode,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(data, [[[2, 2, 2]]], atol=1e-5)

                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 4]
                        strides = [1, 2]
                        data = backend.to_device(
                            conv.convolve_adjoint_data(
                                output, filt, data_shape,
                                mode=mode, strides=strides,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(data, [[[2, 2, 2, 0]]], atol=1e-5)

    def test_convolve_adjoint_data_full(self):
        mode = 'full'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for device in devices:
            xp = device.xp
            with device:
                for dtype in dtypes:
                    with self.subTest(dtype=dtype, device=device):
                        output = xp.ones([1, 5], dtype=dtype)
                        filt = xp.ones([1, 3], dtype=dtype)
                        data_shape = [1, 3]
                        data = backend.to_device(conv.convolve_adjoint_data(
                            output, filt, data_shape, mode=mode))
                        npt.assert_allclose(data, [[3, 3, 3]], atol=1e-5)

                        output = xp.ones([1, 4], dtype=dtype)
                        filt = xp.ones([1, 2], dtype=dtype)
                        data_shape = [1, 3]
                        data = backend.to_device(
                            conv.convolve_adjoint_data(
                                output, filt, data_shape, mode=mode))
                        npt.assert_allclose(data, [[2, 2, 2]], atol=1e-5)

                        output = xp.ones([2, 1, 5], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 3]
                        data = backend.to_device(
                            conv.convolve_adjoint_data(
                                output, filt, data_shape,
                                mode=mode,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(data, [[[6, 6, 6]]], atol=1e-5)

                        output = xp.ones([2, 1, 5], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 8]
                        strides = [1, 2]
                        data = backend.to_device(
                            conv.convolve_adjoint_data(
                                output, filt, data_shape,
                                mode=mode, strides=strides,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(data, [[[4, 2, 4, 2, 4, 2, 4, 2]]],
                                            atol=1e-5)

    def test_convolve_adjoint_filter_valid(self):
        mode = 'valid'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for device in devices:
            xp = device.xp
            with device:
                for dtype in dtypes:
                    with self.subTest(dtype=dtype, device=device):
                        data = xp.ones([1, 3], dtype=dtype)
                        output = xp.ones([1, 1], dtype=dtype)
                        filt_shape = [1, 3]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape, mode=mode))
                        npt.assert_allclose(filt, [[1, 1, 1]], atol=1e-5)

                        data = xp.ones([1, 3], dtype=dtype)
                        output = xp.ones([1, 2], dtype=dtype)
                        filt_shape = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape, mode=mode))
                        npt.assert_allclose(filt, [[2, 2]], atol=1e-5)

                        data = xp.ones([1, 1, 3], dtype=dtype)
                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape,
                                mode=mode,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(filt, [[[[1, 1, 1]]],
                                                   [[[1, 1, 1]]]], atol=1e-5)

                        data = xp.ones([1, 1, 4], dtype=dtype)
                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        strides = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape,
                                mode=mode, strides=strides,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(filt, [[[[1, 1, 1]]],
                                                   [[[1, 1, 1]]]], atol=1e-5)

    def test_convolve_adjoint_filter_full(self):
        mode = 'full'
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for device in devices:
            xp = device.xp
            with device:
                for dtype in dtypes:
                    with self.subTest(dtype=dtype, device=device):
                        data = xp.ones([1, 3], dtype=dtype)
                        output = xp.ones([1, 5], dtype=dtype)
                        filt_shape = [1, 3]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape, mode=mode))
                        npt.assert_allclose(filt, [[3, 3, 3]], atol=1e-5)

                        data = xp.ones([1, 3], dtype=dtype)
                        output = xp.ones([1, 4], dtype=dtype)
                        filt_shape = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape, mode=mode))
                        npt.assert_allclose(filt, [[3, 3]], atol=1e-5)

                        data = xp.ones([1, 1, 3], dtype=dtype)
                        output = xp.ones([2, 1, 5], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape,
                                mode=mode,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(filt, [[[[3, 3, 3]]],
                                                   [[[3, 3, 3]]]], atol=1e-5)

                        data = xp.ones([1, 1, 3], dtype=dtype)
                        output = xp.ones([2, 1, 3], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        strides = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_adjoint_filter(
                                output, data, filt_shape,
                                mode=mode, strides=strides,
                                multi_channel=True),
                            backend.cpu_device)
                        npt.assert_allclose(filt, [[[[2, 1, 2]]],
                                                   [[[2, 1, 2]]]], atol=1e-5)
