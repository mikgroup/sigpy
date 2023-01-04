import unittest
import numpy.testing as npt
from sigpy import backend, config, fourier, linop, util

from sigpy.mri import nlop

if __name__ == '__main__':
    unittest.main()

devices = [backend.cpu_device]
if config.cupy_enabled:
    devices.append(backend.Device(0))


class TestNlop(unittest.TestCase):

    def test_nlinv_model(self):
        for device in devices:
            xp = device.xp

            I = util.randn((1, 3, 3), dtype=complex, device=device)
            C = util.randn((4, 3, 3), dtype=complex, device=device)

            A = nlop.Nlinv(I.shape, C.shape, W_coil=False)

            x = device.xp.ones(A.ishape, dtype=complex)
            x[0, :, :] = I
            x[1:, :, :] = C

            F = linop.FFT(C.shape, axes=(-2, -1))

            # test forward
            y1 = F(I*C)
            y2 = A.forward(x)

            npt.assert_allclose(backend.to_device(y2),
                                backend.to_device(y1),
                                err_msg='forward operator!')

            # test derivative
            dx = util.randn(x.shape, dtype=complex, device=device)

            dy1 = F(dx[0, :, :] * C + I * dx[1:, :, :])
            dy2 = A.derivative(x, dx)

            npt.assert_allclose(backend.to_device(dy2),
                                backend.to_device(dy1),
                                err_msg='derivative operator!')

            # test adjoint
            dx1 = xp.zeros(x.shape, dtype=complex)

            dI1 = xp.sum(xp.conj(C) * F.H(dy1), axis=0)
            dC1 = xp.conj(I) * F.H(dy1)

            dx1[0, :, :] = dI1
            dx1[1:, :, :] = dC1

            dx2 = A.adjoint(x, dy2)

            npt.assert_allclose(backend.to_device(dx2),
                                backend.to_device(dx1),
                                err_msg='adjoint operator!')
