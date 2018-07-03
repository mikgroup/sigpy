import unittest
import pickle
import numpy as np
import numpy.testing as npt
from sigpy import linop, util, config

if __name__ == '__main__':
    unittest.main()


def check_linop_unitary(A, dtype=np.complex, device=util.cpu_device):

    device = util.Device(device)
    x = util.randn(A.ishape, dtype=dtype, device=device)

    xp = device.xp
    with device:
        xp.testing.assert_allclose(A.H * A * x, x, atol=1e-5, rtol=1e-5)


def check_linop_linear(A, dtype=np.complex, device=util.cpu_device):

    device = util.Device(device)
    a = util.randn([1], dtype=dtype, device=device)
    x = util.randn(A.ishape, dtype=dtype, device=device)
    y = util.randn(A.ishape, dtype=dtype, device=device)

    xp = device.xp
    with device:
        xp.testing.assert_allclose(A(a * x + y),
                                   a * A(x) + A(y), atol=1e-5, rtol=1e-5)


def check_linop_adjoint(A, dtype=np.complex, device=util.cpu_device):

    device = util.Device(device)
    x = util.randn(A.ishape, dtype=dtype, device=device)
    y = util.randn(A.oshape, dtype=dtype, device=device)

    xp = device.xp
    with device:
        lhs = xp.vdot(A * x, y)
        rhs = xp.vdot(x, A.H * y)

        xp.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


def check_linop_pickleable(A):

    assert A.__repr__() == pickle.loads(pickle.dumps(A)).__repr__()


class TestLinop(unittest.TestCase):

    def test_Identity(self):

        shape = [5]
        device = util.cpu_device
        A = linop.Identity(shape)
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_unitary(A)
        check_linop_pickleable(A)

    def test_Move(self):

        shape = [5]
        odevice = util.cpu_device
        idevice = util.cpu_device
        A = linop.Move(shape, odevice, idevice)
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Conj(self):

        shape = [5]
        device = util.cpu_device
        I = linop.Identity(shape)
        A = linop.Conj(I)
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Add(self):

        shape = [5]
        device = util.cpu_device
        I = linop.Identity(shape)
        A = linop.Add([I, I])
        x = util.randn(shape)

        npt.assert_allclose(A(x), 2 * x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Compose(self):

        shape = [5]
        device = util.cpu_device
        I = linop.Identity(shape)
        A = linop.Compose([I, I])
        x = util.randn(shape)

        npt.assert_allclose(A(x), x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Hstack(self):

        shape = [5]
        device = util.cpu_device
        I = linop.Identity(shape)
        x1 = util.randn(shape)
        x2 = util.randn(shape)
        x = util.vec([x1, x2])

        A = linop.Hstack([I, I])
        npt.assert_allclose(A(x), x1 + x2)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        shape = [5, 3]
        device = util.cpu_device
        I = linop.Identity(shape)
        x1 = util.randn(shape)
        x2 = util.randn(shape)
        x = np.concatenate([x1, x2], axis=1)

        A = linop.Hstack([I, I], axis=1)
        npt.assert_allclose(A(x), x1 + x2)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Vstack(self):

        shape = [5]
        device = util.cpu_device
        I = linop.Identity(shape)
        x = util.randn(shape)

        A = linop.Vstack([I, I])
        npt.assert_allclose(A(x), util.vec([x, x]))
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        shape = [5, 3]
        device = util.cpu_device
        I = linop.Identity(shape)
        x = util.randn(shape)

        A = linop.Vstack([I, I], axis=1)
        npt.assert_allclose(A(x), np.concatenate([x, x], axis=1))
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Diag(self):

        shape = [5]
        device = util.cpu_device
        I = linop.Identity(shape)
        x = util.randn([10])

        A = linop.Diag([I, I])
        npt.assert_allclose(A(x), x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        shape = [5, 3]
        device = util.cpu_device
        I = linop.Identity(shape)
        x = util.randn([5, 6])

        A = linop.Diag([I, I], axis=1)
        npt.assert_allclose(A(x), x)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_FFT(self):

        for ndim in [1, 2, 3]:
            ishape = [3] * ndim
            A = linop.FFT(ishape)
            check_linop_linear(A)
            check_linop_adjoint(A)
            check_linop_unitary(A)
            check_linop_pickleable(A)

    def test_NUFFT(self):

        for ndim in [1, 2, 3]:
            ishape = [3] * ndim
            coord = np.random.random([10, ndim])

            A = linop.NUFFT(ishape, coord)
            check_linop_linear(A)
            check_linop_adjoint(A)
            check_linop_pickleable(A)

    def test_MatMul(self):

        mshape = (5, 4, 2)
        ishape = (5, 2, 3)
        A = linop.MatMul(ishape, util.randn(mshape))
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

    def test_RightMatMul(self):

        ishape = (5, 4, 2)
        mshape = (5, 2, 3)
        A = linop.RightMatMul(ishape, util.randn(mshape))
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

    def test_Multiply(self):

        # Test scalar
        ishape = [2]
        mult = 1.1

        A = linop.Multiply(ishape, mult)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        x = np.array([1.0, 2.0], np.complex)
        y = np.array([1.1, 2.2], np.complex)
        npt.assert_allclose(A * x, y)

        # Test simple
        oshape = [2]
        ishape = [2]
        mult = np.array([1.0, 2.0])

        A = linop.Multiply(ishape, mult)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        x = np.array([1.0, 2.0], np.complex)
        y = np.array([1.0, 4.0], np.complex)
        npt.assert_allclose(A * x, y)

        # Test broadcasting
        oshape = [2, 2]
        ishape = [2]
        mult = np.array([[1.0, 2.0],
                         [3.0, 4.0]])

        A = linop.Multiply(ishape, mult)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        x = np.array([1.0, 2.0], np.complex)
        y = np.array([[1.0, 4.0],
                      [3.0, 8.0]], np.complex)
        npt.assert_allclose(A * x, y)

    def test_Resize(self):
        ishape = [3]
        oshape = [5]

        A = linop.Resize(oshape, ishape)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Resize(oshape, ishape, oshift=[1])
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        ishape = [5]
        oshape = [3]

        A = linop.Resize(oshape, ishape)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Resize(oshape, ishape, ishift=[1])
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Downsample(self):
        ishape = [5]
        factors = [2]

        A = linop.Downsample(ishape, factors)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Downsample(ishape, factors, shift=[1])
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Upsample(self):
        oshape = [5]
        factors = [2]

        A = linop.Downsample(oshape, factors)
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Downsample(oshape, factors, shift=[1])
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Interp(self):

        # Test linear interpolation.
        width = 2.0
        ishape = [5]
        coord = np.array([[0.5], [1.5], [2.5]])
        table = [1.0, 0.5]

        A = linop.Interp(ishape, coord, width, table)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

        npt.assert_allclose([0.5, 1.5, 1.5], A * x)

        # Test no batch
        ishape = [2, 2]
        coord = np.array([[0, 0],
                          [1, 0],
                          [1.5, 0]]) / 4.0

        A = linop.Interp(ishape, coord, width, table)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        # Test batch
        ishape = [2, 2, 2]

        A = linop.Interp(ishape, coord, width, table)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

    def test_Wavelet(self):

        shape = [16]
        A = linop.Wavelet(shape, level=1, wave_name='haar')
        check_linop_adjoint(A)
        check_linop_unitary(A)
        check_linop_pickleable(A)

        shape = [129]
        A = linop.Wavelet(shape)
        check_linop_adjoint(A)
        check_linop_unitary(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        shape = [17]
        A = linop.Wavelet(shape, level=1)
        check_linop_adjoint(A)
        check_linop_unitary(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

    def test_Circshift(self):

        shape = [8]
        shift = [4]
        A = linop.Circshift(shape, shift)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_unitary(A)
        check_linop_pickleable(A)

    def test_Gradient(self):

        shape = [8]
        A = linop.Gradient(shape)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

    def test_Transpose(self):

        shape = [3, 4]
        A = linop.Transpose(shape)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_unitary(A)
        check_linop_pickleable(A)

        x = util.randn(shape)
        npt.assert_allclose(A(x), np.transpose(x))

    def test_Sum(self):

        shape = [2, 3, 4, 2]
        axes = [1, 3]
        A = linop.Sum(shape, axes)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

    def test_TensorToBlocks(self):

        ishape = [4]
        bshape = [2]

        A = linop.TensorToBlocks(ishape, bshape)
        check_linop_adjoint(A)
        check_linop_linear(A)
        check_linop_pickleable(A)

        x = np.array([1, 2, 3, 4], np.complex)

        npt.assert_allclose(A(x), [[1, 2],
                                   [3, 4]])

    def test_Convolve(self):
        ishape = [3, 4]
        filt = util.randn([2, 3])

        A = linop.Convolve(ishape, filt, mode='full')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Convolve(ishape, filt, mode='valid')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        ishape = [2, 3]
        filt = util.randn([3, 4])

        A = linop.Convolve(ishape, filt, mode='full')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Convolve(ishape, filt, mode='valid')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    def test_Correlate(self):
        ishape = [3, 4]
        filt = util.randn([2, 3])

        A = linop.Correlate(ishape, filt, mode='full')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Correlate(ishape, filt, mode='valid')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        ishape = [2, 3]
        filt = util.randn([3, 4])

        A = linop.Correlate(ishape, filt, mode='full')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

        A = linop.Correlate(ishape, filt, mode='valid')
        check_linop_linear(A)
        check_linop_adjoint(A)
        check_linop_pickleable(A)

    if config.cudnn_enabled:
        def test_CudnnConvolveData(self):
            x_shape = [2, 1, 3, 4]
            W = util.randn([5, 1, 2, 3], device=0)

            A = linop.CudnnConvolveData(x_shape, W, mode='full')
            check_linop_linear(A, device=0)
            check_linop_adjoint(A, device=0)
            check_linop_pickleable(A)

            A = linop.CudnnConvolveData(x_shape, W, mode='valid')
            check_linop_linear(A, device=0)
            check_linop_adjoint(A, device=0)
            check_linop_pickleable(A)

        def test_CudnnConvolveFilter(self):
            x = util.randn([2, 1, 3, 4], device=0)
            W_shape = [5, 1, 2, 3]

            A = linop.CudnnConvolveFilter(W_shape, x, mode='full')
            check_linop_linear(A, device=0)
            check_linop_adjoint(A, device=0)
            check_linop_pickleable(A)

            A = linop.CudnnConvolveFilter(W_shape, x, mode='valid')
            check_linop_linear(A, device=0)
            check_linop_adjoint(A, device=0)
            check_linop_pickleable(A)
