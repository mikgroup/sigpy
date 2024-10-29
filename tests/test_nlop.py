import unittest
import numpy as np
import numpy.testing as npt
from sigpy import backend, config, nlop, util, linop

import torch
from torch.autograd import Variable

if __name__ == '__main__':
    unittest.main()

devices = [backend.cpu_device]
if config.cupy_enabled:
    devices.append(backend.Device(0))


class TestNlop(unittest.TestCase):

    def check_nlop_derivative(self, A,
                              device=backend.cpu_device,
                              dtype=float):
        device = backend.Device(device)

        scale = 1e-8
        x = util.randn(A.ishape, dtype=dtype, device=device)
        h = util.randn(A.ishape, dtype=dtype, device=device)

        with device:
            dy1 = (A.forward(x + scale * h) - A.forward(x)) / scale
            dy2 = A.derivative(x, h)

            npt.assert_allclose(backend.to_device(dy1),
                                backend.to_device(dy2),
                                atol=1e-5,
                                err_msg=A.repr_str + ' derivative operator!')

    def check_nlop_adjoint(self, A, device=backend.cpu_device, dtype=float):
        device = backend.Device(device)
        x = util.randn(A.ishape, dtype=dtype, device=device)

        dx = util.randn(A.ishape, dtype=dtype, device=device)
        dy = util.randn(A.oshape, dtype=dtype, device=device)

        xp = device.xp
        with device:
            lhs = xp.vdot(A.derivative(x, dx), dy)
            rhs = xp.vdot(dx, A.adjoint(x, dy))

            npt.assert_allclose(backend.to_device(lhs),
                                backend.to_device(rhs),
                                atol=1e-5,
                                err_msg=A.repr_str + ' adjoint operator!')

    def test_Exponential(self):
        for device in devices:
            xp = device.xp
            list_bias = [True, False, True, False]
            list_coef = [1, 1, 6, 6]
            for bias, num_coef in zip(list_bias, list_coef):
                num_param = num_coef + 2 if bias is True else num_coef + 1
                I = util.randn((num_param, 1, 3, 3),
                               dtype=complex, device=device)
                image_shape = I.shape[1:]

                num_time = 7
                tvec = util.randn((num_time, num_coef),
                                  dtype=float, device=device)

                A = nlop.Exponential(I.shape, tvec, bias=bias)

                # test forward
                y2 = A.forward(I)
                y1 = xp.zeros_like(y2)
                offset = 1 if bias else 0
                a = I[offset, ...]
                R = I[offset+1:, ...]
                Rr = xp.reshape(R, (R.shape[0], -1))
                y1 = xp.exp(xp.matmul(tvec, Rr))
                y1 = a * xp.reshape(y1, [num_time] + list(image_shape))
                if bias is True:
                    y1 += I[0, ...]

                npt.assert_allclose(backend.to_device(y2),
                                    backend.to_device(y1),
                                    err_msg=A.repr_str + ' forward operator!')

                # test derivative
                self.check_nlop_derivative(A, device=device, dtype=I.dtype)

                # test adjoint
                self.check_nlop_adjoint(A, device=device, dtype=I.dtype)

    def test_Exponential_torch(self):
        b0 = util.randn((1, 1, 3, 3), dtype=float)
        D = util.randn((6, 1, 3, 3), dtype=float)
        I = np.concatenate((b0, D))

        tvec = util.randn((16, 6), dtype=float)

        A = nlop.Exponential(I.shape, tvec, bias=False)

        dy = np.ones(A.oshape, dtype=float)
        dx = A.adjoint(I, dy)

        # PyTorch
        b0_torch = Variable(torch.tensor(b0), requires_grad=True)
        D_torch = Variable(torch.tensor(D), requires_grad=True)
        t_torch = Variable(torch.tensor(tvec), requires_grad=False)

        Dr_torch = torch.reshape(D_torch, (D_torch.shape[0], -1))
        y2 = torch.exp(torch.matmul(t_torch, Dr_torch))
        y2 = torch.reshape(y2, A.oshape)
        y2 = (y2 * b0_torch).sum()

        y2.backward()

        db0 = b0_torch.grad.numpy()
        dD = D_torch.grad.numpy()

        dx_torch = np.concatenate((db0, dD))

        npt.assert_allclose(dx, dx_torch, err_msg='Derivative mismatch!')

    def test_Compose(self):
        tvec = util.randn((15, 6), dtype=float)
        x = util.randn((7, 1, 3, 3), dtype=complex)

        E = nlop.Exponential(x.shape, tvec, bias=False)

        smat = util.randn((8, 3, 3), dtype=complex)
        S = linop.Multiply(E.oshape, smat)

        A = S * E

        y1 = smat * E(x)
        y2 = A(x)

        npt.assert_allclose(y2, y1,
                            err_msg=A.repr_str + ' forward operator!')

        dx = util.randn(x.shape, dtype=x.dtype)

        dy1 = smat * E.derivative(x, dx)
        dy2 = A.derivative(x, dx)

        npt.assert_allclose(dy2, dy1,
                            err_msg=A.repr_str + ' derivative operator!')

        dy = util.randn(A.oshape, dtype=x.dtype)

        dx1 = E.adjoint(x, S.H * dy)
        dx2 = A.adjoint(x, dy)

        npt.assert_allclose(dx2, dx1,
                            err_msg=A.repr_str + ' adjoint operator!')
