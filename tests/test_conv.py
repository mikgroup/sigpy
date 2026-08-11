import unittest

import numpy as np
import numpy.testing as npt

from sigpy import backend, config, conv, util

if __name__ == "__main__":
    unittest.main()

dtypes = [np.float32, np.float64, np.complex64, np.complex128]


class TestConv(unittest.TestCase):
    def test_convolve_valid(self):
        mode = "valid"
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for D in [1, 2, 3]:
            for device in devices:
                xp = device.xp
                with device:
                    for dtype in dtypes:
                        with self.subTest(D=D, dtype=dtype, device=device):
                            data = util.dirac(
                                [3] + [1] * (D - 1), device=device, dtype=dtype
                            )
                            filt = xp.ones([3] + [1] * (D - 1), dtype=dtype)
                            output = backend.to_device(
                                conv.convolve(data, filt, mode=mode)
                            )
                            npt.assert_allclose(
                                output, np.ones([1] * D), atol=1e-5
                            )

                            data = util.dirac(
                                [3] + [1] * (D - 1), device=device, dtype=dtype
                            )
                            filt = xp.ones([2] + [1] * (D - 1), dtype=dtype)
                            output = backend.to_device(
                                conv.convolve(data, filt, mode=mode)
                            )
                            npt.assert_allclose(
                                output, np.ones([2] + [1] * (D - 1)), atol=1e-5
                            )

                            data = util.dirac(
                                [1, 3] + [1] * (D - 1),
                                device=device,
                                dtype=dtype,
                            )
                            filt = xp.ones(
                                [2, 1, 3] + [1] * (D - 1), dtype=dtype
                            )
                            output = backend.to_device(
                                conv.convolve(
                                    data, filt, mode=mode, multi_channel=True
                                ),
                                backend.cpu_device,
                            )
                            npt.assert_allclose(
                                output,
                                np.ones([2, 1] + [1] * (D - 1)),
                                atol=1e-5,
                            )

                            data = util.dirac(
                                [1, 3] + [1] * (D - 1),
                                device=device,
                                dtype=dtype,
                            )
                            filt = xp.ones(
                                [2, 1, 3] + [1] * (D - 1), dtype=dtype
                            )
                            strides = [2] + [1] * (D - 1)
                            output = backend.to_device(
                                conv.convolve(
                                    data,
                                    filt,
                                    mode=mode,
                                    strides=strides,
                                    multi_channel=True,
                                ),
                                backend.cpu_device,
                            )
                            npt.assert_allclose(
                                output,
                                np.ones([2, 1] + [1] * (D - 1)),
                                atol=1e-5,
                            )

    def test_convolve_full(self):
        mode = "full"
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for D in [1, 2, 3]:
            for device in devices:
                xp = device.xp
                with device:
                    for dtype in dtypes:
                        with self.subTest(D=D, dtype=dtype, device=device):
                            data = util.dirac(
                                [3] + [1] * (D - 1), device=device, dtype=dtype
                            )
                            filt = xp.ones([3] + [1] * (D - 1), dtype=dtype)
                            output = backend.to_device(
                                conv.convolve(data, filt, mode=mode)
                            )
                            npt.assert_allclose(
                                output,
                                np.array([0, 1, 1, 1, 0]).reshape(
                                    [5] + [1] * (D - 1)
                                ),
                                atol=1e-5,
                            )

                            data = util.dirac(
                                [3] + [1] * (D - 1), device=device, dtype=dtype
                            )
                            filt = xp.ones([2] + [1] * (D - 1), dtype=dtype)
                            output = backend.to_device(
                                conv.convolve(data, filt, mode=mode)
                            )
                            npt.assert_allclose(
                                output,
                                np.array([0, 1, 1, 0]).reshape(
                                    [4] + [1] * (D - 1)
                                ),
                                atol=1e-5,
                            )

                            data = util.dirac(
                                [1, 3] + [1] * (D - 1),
                                device=device,
                                dtype=dtype,
                            )
                            filt = xp.ones(
                                [2, 1, 3] + [1] * (D - 1), dtype=dtype
                            )
                            output = backend.to_device(
                                conv.convolve(
                                    data, filt, mode=mode, multi_channel=True
                                ),
                                backend.cpu_device,
                            )
                            npt.assert_allclose(
                                output,
                                np.array(
                                    [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]
                                ).reshape([2, 5] + [1] * (D - 1)),
                                atol=1e-5,
                            )

                            data = util.dirac(
                                [1, 3] + [1] * (D - 1),
                                device=device,
                                dtype=dtype,
                            )
                            filt = xp.ones(
                                [2, 1, 3] + [1] * (D - 1), dtype=dtype
                            )
                            strides = [2] + [1] * (D - 1)
                            output = backend.to_device(
                                conv.convolve(
                                    data,
                                    filt,
                                    mode=mode,
                                    strides=strides,
                                    multi_channel=True,
                                )
                            )
                            npt.assert_allclose(
                                output,
                                np.array([[0, 1, 0], [0, 1, 0]]).reshape(
                                    [2, 3] + [1] * (D - 1)
                                ),
                                atol=1e-5,
                            )

    def test_convolve_data_adjoint_valid(self):
        mode = "valid"
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
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output, filt, data_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(data, [[1, 1, 1]], atol=1e-5)

                        output = xp.ones([1, 2], dtype=dtype)
                        filt = xp.ones([1, 2], dtype=dtype)
                        data_shape = [1, 3]
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output, filt, data_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(data, [[1, 2, 1]], atol=1e-5)

                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 3]
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output,
                                filt,
                                data_shape,
                                mode=mode,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(data, [[[2, 2, 2]]], atol=1e-5)

                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 4]
                        strides = [1, 2]
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output,
                                filt,
                                data_shape,
                                mode=mode,
                                strides=strides,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(data, [[[2, 2, 2, 0]]], atol=1e-5)

    def test_convolve_data_adjoint_full(self):
        mode = "full"
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
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output, filt, data_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(data, [[3, 3, 3]], atol=1e-5)

                        output = xp.ones([1, 4], dtype=dtype)
                        filt = xp.ones([1, 2], dtype=dtype)
                        data_shape = [1, 3]
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output, filt, data_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(data, [[2, 2, 2]], atol=1e-5)

                        output = xp.ones([2, 1, 5], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 3]
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output,
                                filt,
                                data_shape,
                                mode=mode,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(data, [[[6, 6, 6]]], atol=1e-5)

                        output = xp.ones([2, 1, 5], dtype=dtype)
                        filt = xp.ones([2, 1, 1, 3], dtype=dtype)
                        data_shape = [1, 1, 8]
                        strides = [1, 2]
                        data = backend.to_device(
                            conv.convolve_data_adjoint(
                                output,
                                filt,
                                data_shape,
                                mode=mode,
                                strides=strides,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(
                            data, [[[4, 2, 4, 2, 4, 2, 4, 2]]], atol=1e-5
                        )

    def test_convolve_filter_adjoint_valid(self):
        mode = "valid"
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
                            conv.convolve_filter_adjoint(
                                output, data, filt_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(filt, [[1, 1, 1]], atol=1e-5)

                        data = xp.ones([1, 3], dtype=dtype)
                        output = xp.ones([1, 2], dtype=dtype)
                        filt_shape = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_filter_adjoint(
                                output, data, filt_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(filt, [[2, 2]], atol=1e-5)

                        data = xp.ones([1, 1, 3], dtype=dtype)
                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        filt = backend.to_device(
                            conv.convolve_filter_adjoint(
                                output,
                                data,
                                filt_shape,
                                mode=mode,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(
                            filt, [[[[1, 1, 1]]], [[[1, 1, 1]]]], atol=1e-5
                        )

                        data = xp.ones([1, 1, 4], dtype=dtype)
                        output = xp.ones([2, 1, 1], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        strides = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_filter_adjoint(
                                output,
                                data,
                                filt_shape,
                                mode=mode,
                                strides=strides,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(
                            filt, [[[[1, 1, 1]]], [[[1, 1, 1]]]], atol=1e-5
                        )

    def test_convolve_filter_adjoint_full(self):
        mode = "full"
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
                            conv.convolve_filter_adjoint(
                                output, data, filt_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(filt, [[3, 3, 3]], atol=1e-5)

                        data = xp.ones([1, 3], dtype=dtype)
                        output = xp.ones([1, 4], dtype=dtype)
                        filt_shape = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_filter_adjoint(
                                output, data, filt_shape, mode=mode
                            )
                        )
                        npt.assert_allclose(filt, [[3, 3]], atol=1e-5)

                        data = xp.ones([1, 1, 3], dtype=dtype)
                        output = xp.ones([2, 1, 5], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        filt = backend.to_device(
                            conv.convolve_filter_adjoint(
                                output,
                                data,
                                filt_shape,
                                mode=mode,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(
                            filt, [[[[3, 3, 3]]], [[[3, 3, 3]]]], atol=1e-5
                        )

                        data = xp.ones([1, 1, 3], dtype=dtype)
                        output = xp.ones([2, 1, 3], dtype=dtype)
                        filt_shape = [2, 1, 1, 3]
                        strides = [1, 2]
                        filt = backend.to_device(
                            conv.convolve_filter_adjoint(
                                output,
                                data,
                                filt_shape,
                                mode=mode,
                                strides=strides,
                                multi_channel=True,
                            ),
                            backend.cpu_device,
                        )
                        npt.assert_allclose(
                            filt, [[[[2, 1, 2]]], [[[2, 1, 2]]]], atol=1e-5
                        )

    @unittest.skipIf(not config.cupy_enabled, "cupy not installed")
    def test_convolve_gpu_matches_cpu(self):
        # The GPU results must match the CPU results bit-for-bit in intent,
        # whether the GPU path goes through cuDNN or the FFT fallback.
        np.random.seed(0)
        device = backend.Device(0)
        for D in [1, 2, 3]:
            for mode in ["full", "valid"]:
                for strides in [None, [2] * D]:
                    # Equal sizes make valid mode degenerate to a single
                    # output sample, which is worth covering separately.
                    for m, n in [([5] * D, [3] * D), ([4] * D, [4] * D)]:
                        for dtype in dtypes:
                            for multi_channel in [False, True]:
                                with self.subTest(
                                    D=D,
                                    mode=mode,
                                    strides=strides,
                                    m=m,
                                    n=n,
                                    dtype=dtype,
                                    multi_channel=multi_channel,
                                ):
                                    self._check_gpu_matches_cpu(
                                        device,
                                        D,
                                        m,
                                        n,
                                        mode,
                                        strides,
                                        dtype,
                                        multi_channel,
                                    )

    def _check_gpu_matches_cpu(
        self, device, D, m, n, mode, strides, dtype, multi_channel
    ):
        if multi_channel:
            c_i, c_o = 2, 3
            data_shape = [2, c_i] + m
            filt_shape = [c_o, c_i] + n
        else:
            data_shape = [2] + m
            filt_shape = n

        # Generate on CPU and copy, so both devices see identical inputs.
        data = util.randn(data_shape, dtype=dtype)
        filt = util.randn(filt_shape, dtype=dtype)
        data_gpu = backend.to_device(data, device)
        filt_gpu = backend.to_device(filt, device)

        kwargs = {
            "mode": mode,
            "strides": strides,
            "multi_channel": multi_channel,
        }

        output = conv.convolve(data, filt, **kwargs)
        self._assert_close(
            conv.convolve(data_gpu, filt_gpu, **kwargs), output, dtype
        )

        output_gpu = backend.to_device(output, device)
        self._assert_close(
            conv.convolve_data_adjoint(
                output_gpu, filt_gpu, data_shape, **kwargs
            ),
            conv.convolve_data_adjoint(output, filt, data_shape, **kwargs),
            dtype,
        )
        self._assert_close(
            conv.convolve_filter_adjoint(
                output_gpu, data_gpu, filt_shape, **kwargs
            ),
            conv.convolve_filter_adjoint(output, data, filt_shape, **kwargs),
            dtype,
        )

    def _assert_close(self, actual, desired, dtype):
        # Without cuDNN the GPU convolves via FFT while scipy picks the
        # direct method for these sizes. The two agree only to roundoff,
        # which scales with the magnitude of the inputs, so compare the
        # normalized error rather than an element-wise absolute tolerance.
        actual = backend.to_device(actual)
        desired = backend.to_device(desired)
        self.assertEqual(actual.shape, desired.shape)
        self.assertEqual(actual.dtype, desired.dtype)
        err = np.linalg.norm(actual - desired) / np.linalg.norm(desired)
        self.assertLess(err, 1e3 * np.finfo(dtype).eps)

    def test_convolve_adjoint_consistency(self):
        # <convolve(data, filt), output> must equal both
        # <data, convolve_data_adjoint(output, filt)> and
        # <filt, convolve_filter_adjoint(output, data)>.
        np.random.seed(0)
        devices = [backend.cpu_device]
        if config.cupy_enabled:
            devices.append(backend.Device(0))

        for device in devices:
            xp = device.xp
            for D in [1, 2, 3]:
                for mode in ["full", "valid"]:
                    for strides in [None, [2] * D]:
                        for dtype in dtypes:
                            with self.subTest(
                                D=D,
                                mode=mode,
                                strides=strides,
                                dtype=dtype,
                                device=device,
                            ):
                                c_i, c_o = 2, 3
                                data_shape = [4, c_i] + [6] * D
                                filt_shape = [c_o, c_i] + [3] * D
                                data = util.randn(
                                    data_shape, dtype=dtype, device=device
                                )
                                filt = util.randn(
                                    filt_shape, dtype=dtype, device=device
                                )
                                kwargs = {
                                    "mode": mode,
                                    "strides": strides,
                                    "multi_channel": True,
                                }
                                with device:
                                    output = conv.convolve(
                                        data, filt, **kwargs
                                    )
                                    shape = output.shape
                                    adj_output = util.randn(
                                        shape, dtype=dtype, device=device
                                    )
                                    lhs = xp.vdot(adj_output, output)
                                    rhs_data = xp.vdot(
                                        conv.convolve_data_adjoint(
                                            adj_output,
                                            filt,
                                            data_shape,
                                            **kwargs,
                                        ),
                                        data,
                                    )
                                    rhs_filt = xp.vdot(
                                        conv.convolve_filter_adjoint(
                                            adj_output,
                                            data,
                                            filt_shape,
                                            **kwargs,
                                        ),
                                        filt,
                                    )

                                atol = (
                                    1e-3
                                    if np.dtype(dtype).itemsize <= 8
                                    else 1e-9
                                )
                                npt.assert_allclose(
                                    backend.to_device(rhs_data),
                                    backend.to_device(lhs),
                                    atol=atol,
                                    rtol=1e-4,
                                )
                                npt.assert_allclose(
                                    backend.to_device(rhs_filt),
                                    backend.to_device(lhs),
                                    atol=atol,
                                    rtol=1e-4,
                                )
