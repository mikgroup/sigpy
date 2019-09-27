import unittest
import pickle
import numpy as np
import numpy.testing as npt
from sigpy import backend, linop, util, config

if __name__ == '__main__':
    unittest.main()

dtypes = [np.float32, np.float64, np.complex64, np.complex128]
devices = [backend.cpu_device]
if config.cupy_enabled:
    devices.append(backend.Device(0))


class TestLinop(unittest.TestCase):

    def check_linop_unitary(self, A,
                            device=backend.cpu_device, dtype=np.float):
        device = backend.Device(device)
        x = util.randn(A.ishape, dtype=dtype, device=device)
        xp = device.xp
        with device:
            xp.testing.assert_allclose(A.H * A * x, x,
                                       atol=1e-5, rtol=1e-5)

    def check_linop_linear(self, A,
                           device=backend.cpu_device, dtype=np.float):
        device = backend.Device(device)
        a = util.randn([1], dtype=dtype, device=device)
        x = util.randn(A.ishape, dtype=dtype, device=device)
        y = util.randn(A.ishape, dtype=dtype, device=device)

        xp = device.xp
        with device:
            xp.testing.assert_allclose(A(a * x + y),
                                       a * A(x) + A(y),
                                       atol=1e-5, rtol=1e-5)

    def check_linop_adjoint(self, A,
                            device=backend.cpu_device, dtype=np.float):
        device = backend.Device(device)
        x = util.randn(A.ishape, dtype=dtype, device=device)
        y = util.randn(A.oshape, dtype=dtype, device=device)

        xp = device.xp
        with device:
            lhs = xp.vdot(A * x, y)
            rhs = xp.vdot(A.H.H * x, y)
            xp.testing.assert_allclose(lhs, rhs,
                                       atol=1e-5, rtol=1e-5)

            rhs = xp.vdot(x, A.H * y)
            xp.testing.assert_allclose(lhs, rhs,
                                       atol=1e-5, rtol=1e-5)

    def check_linop_pickleable(self, A):
        with self.subTest(A=A):
            assert A.__repr__() == pickle.loads(pickle.dumps(A)).__repr__()

    def test_Identity(self):
        shape = [5]
        A = linop.Identity(shape)
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_unitary(A)
        self.check_linop_pickleable(A)

    def test_ToDevice(self):
        shape = [5]
        odevice = backend.cpu_device
        idevice = backend.cpu_device
        A = linop.ToDevice(shape, odevice, idevice)
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Conj(self):
        shape = [5]
        I = linop.Identity(shape)
        A = linop.Conj(I)
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Add(self):
        shape = [5]
        I = linop.Identity(shape)
        A = linop.Add([I, I])
        x = util.randn(shape)

        npt.assert_allclose(A(x), 2 * x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Compose(self):
        shape = [5]
        I = linop.Identity(shape)
        A = linop.Compose([I, I])
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Hstack(self):
        shape = [5]
        I = linop.Identity(shape)
        x1 = util.randn(shape)
        x2 = util.randn(shape)
        x = util.vec([x1, x2])

        A = linop.Hstack([I, I])
        npt.assert_allclose(A(x), x1 + x2)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        shape = [5, 3]
        I = linop.Identity(shape)
        x1 = util.randn(shape)
        x2 = util.randn(shape)
        x = np.concatenate([x1, x2], axis=1)

        A = linop.Hstack([I, I], axis=1)
        npt.assert_allclose(A(x), x1 + x2)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Vstack(self):
        shape = [5]
        I = linop.Identity(shape)
        x = util.randn(shape)

        A = linop.Vstack([I, I])
        npt.assert_allclose(A(x), util.vec([x, x]))
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        shape = [5, 3]
        I = linop.Identity(shape)
        x = util.randn(shape)

        A = linop.Vstack([I, I], axis=1)
        npt.assert_allclose(A(x), np.concatenate([x, x], axis=1))
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Diag(self):
        shape = [5]
        I = linop.Identity(shape)
        x = util.randn([10])

        A = linop.Diag([I, I])
        npt.assert_allclose(A(x), x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        shape = [5, 3]
        I = linop.Identity(shape)
        x = util.randn([5, 6])

        A = linop.Diag([I, I], axis=1)
        npt.assert_allclose(A(x), x)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_FFT(self):
        for ndim in [1, 2, 3]:
            for n in [3, 4, 5, 6]:
                ishape = [n] * ndim
                A = linop.FFT(ishape)
                self.check_linop_linear(A)
                self.check_linop_adjoint(A)
                self.check_linop_unitary(A)
                self.check_linop_pickleable(A)

    def test_NUFFT(self):
        for ndim in [1, 2, 3]:
            for n in [2, 3, 4, 5, 6]:
                ishape = [3] * ndim
                coord = np.random.random([10, ndim])

                A = linop.NUFFT(ishape, coord)
                self.check_linop_linear(A)
                self.check_linop_adjoint(A)
                self.check_linop_pickleable(A)

    def test_MatMul(self):
        mshape = (5, 4, 2)
        ishape = (5, 2, 3)
        A = linop.MatMul(ishape, util.randn(mshape))
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

    def test_RightMatMul(self):
        ishape = (5, 4, 2)
        mshape = (5, 2, 3)
        A = linop.RightMatMul(ishape, util.randn(mshape))
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

    def test_Multiply(self):
        # Test scalar
        ishape = [2]
        mult = 1.1

        A = linop.Multiply(ishape, mult)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        x = np.array([1.0, 2.0], np.complex)
        y = np.array([1.1, 2.2], np.complex)
        npt.assert_allclose(A * x, y)

        # Test simple
        ishape = [2]
        mult = np.array([1.0, 2.0])

        A = linop.Multiply(ishape, mult)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        x = np.array([1.0, 2.0], np.complex)
        y = np.array([1.0, 4.0], np.complex)
        npt.assert_allclose(A * x, y)

        # Test broadcasting
        ishape = [2]
        mult = np.array([[1.0, 2.0],
                         [3.0, 4.0]])

        A = linop.Multiply(ishape, mult)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        x = np.array([1.0, 2.0], np.complex)
        y = np.array([[1.0, 4.0],
                      [3.0, 8.0]], np.complex)
        npt.assert_allclose(A * x, y)

    def test_Resize(self):
        ishape = [3]
        oshape = [5]

        A = linop.Resize(oshape, ishape)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        A = linop.Resize(oshape, ishape, oshift=[1])
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        ishape = [5]
        oshape = [3]

        A = linop.Resize(oshape, ishape)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        A = linop.Resize(oshape, ishape, ishift=[1])
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Downsample(self):
        ishape = [5]
        factors = [2]

        A = linop.Downsample(ishape, factors)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        A = linop.Downsample(ishape, factors, shift=[1])
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Upsample(self):
        oshape = [5]
        factors = [2]

        A = linop.Downsample(oshape, factors)
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

        A = linop.Downsample(oshape, factors, shift=[1])
        self.check_linop_linear(A)
        self.check_linop_adjoint(A)
        self.check_linop_pickleable(A)

    def test_Interpolate(self):

        # Test linear interpolation.
        width = 2.0
        ishape = [5]
        coord = np.array([[0.5], [1.5], [2.5]])
        kernel = [1.0, 0.5]

        A = linop.Interpolate(ishape, coord, width, kernel)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

        npt.assert_allclose([0.5, 1.5, 1.5], A * x)

        # Test no batch
        ishape = [2, 2]
        coord = np.array([[0, 0],
                          [1, 0],
                          [1.5, 0]]) / 4.0

        A = linop.Interpolate(ishape, coord, width, kernel)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        # Test batch
        ishape = [2, 2, 2]

        A = linop.Interpolate(ishape, coord, width, kernel)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

    def test_Wavelet(self):

        shape = [16]
        A = linop.Wavelet(shape, level=1, wave_name='haar')
        self.check_linop_adjoint(A)
        self.check_linop_unitary(A)
        self.check_linop_pickleable(A)

        shape = [129]
        A = linop.Wavelet(shape)
        self.check_linop_adjoint(A)
        self.check_linop_unitary(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        shape = [17]
        A = linop.Wavelet(shape, level=1)
        self.check_linop_adjoint(A)
        self.check_linop_unitary(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

    def test_Circshift(self):

        shape = [8]
        shift = [4]
        A = linop.Circshift(shape, shift)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_unitary(A)
        self.check_linop_pickleable(A)

    def test_FiniteDifference(self):
        shape = [8]
        A = linop.FiniteDifference(shape)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

    def test_Transpose(self):
        shape = [3, 4]
        A = linop.Transpose(shape)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_unitary(A)
        self.check_linop_pickleable(A)

        x = util.randn(shape)
        npt.assert_allclose(A(x), np.transpose(x))

    def test_Sum(self):
        shape = [2, 3, 4, 2]
        axes = [1, 3]
        A = linop.Sum(shape, axes)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

    def test_ArrayToBlocks(self):
        ishape = [4]
        blk_shape = [2]
        blk_strides = [2]

        A = linop.ArrayToBlocks(ishape, blk_shape, blk_strides)
        self.check_linop_adjoint(A)
        self.check_linop_linear(A)
        self.check_linop_pickleable(A)

        x = np.array([1, 2, 3, 4], np.complex)

        npt.assert_allclose(A(x), [[1, 2],
                                   [3, 4]])

    def test_ConvolveData(self):
        for device in devices:
            for dtype in dtypes:
                for mode in ['full', 'valid']:
                    with self.subTest(mode=mode, dtype=dtype, device=device):
                        data_shape = [3, 4]
                        filt = util.randn([2, 3], dtype=dtype)
                        A = linop.ConvolveData(data_shape, filt, mode=mode)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        data_shape = [4, 3, 4]
                        filt = util.randn([1, 4, 2, 3], dtype=dtype)
                        A = linop.ConvolveData(
                            data_shape, filt, mode=mode, multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        data_shape = [1, 3, 4]
                        filt = util.randn([4, 1, 2, 3], dtype=dtype)
                        A = linop.ConvolveData(
                            data_shape, filt, mode=mode, multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        data_shape = [2, 3, 4]
                        filt = util.randn([4, 2, 2, 3], dtype=dtype)
                        A = linop.ConvolveData(
                            data_shape, filt,
                            mode=mode,
                            multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        data_shape = [2, 3, 4]
                        filt = util.randn([4, 2, 2, 3], dtype=dtype)
                        strides = [2, 2]
                        A = linop.ConvolveData(
                            data_shape, filt,
                            mode=mode, strides=strides,
                            multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

    def test_ConvolveFilter(self):
        for device in devices:
            for dtype in dtypes:
                for mode in ['full', 'valid']:
                    with self.subTest(mode=mode, dtype=dtype, device=device):
                        filt_shape = [2, 3]
                        data = util.randn([3, 4], dtype=dtype)
                        A = linop.ConvolveFilter(filt_shape, data, mode=mode)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        filt_shape = [1, 4, 2, 3]
                        data = util.randn([4, 3, 4], dtype=dtype)
                        A = linop.ConvolveFilter(
                            filt_shape, data, mode=mode, multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        filt_shape = [4, 1, 2, 3]
                        data = util.randn([1, 3, 4], dtype=dtype)
                        A = linop.ConvolveFilter(
                            filt_shape, data, mode=mode, multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        filt_shape = [4, 2, 2, 3]
                        data = util.randn([2, 3, 4], dtype=dtype)
                        A = linop.ConvolveFilter(
                            filt_shape, data,
                            mode=mode,
                            multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)

                        filt_shape = [4, 2, 2, 3]
                        strides = [2, 2]
                        data = util.randn([2, 3, 4], dtype=dtype)
                        A = linop.ConvolveFilter(
                            filt_shape, data,
                            mode=mode, strides=strides,
                            multi_channel=True)
                        self.check_linop_linear(A, dtype=dtype, device=device)
                        self.check_linop_adjoint(A, dtype=dtype, device=device)
                        self.check_linop_pickleable(A)
