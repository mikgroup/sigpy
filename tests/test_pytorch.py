import unittest
import numpy as np
import numpy.testing as npt
from sigpy import backend, config, linop, pytorch

if config.pytorch_enabled:
    import torch
    if __name__ == '__main__':
        unittest.main()

    devices = [backend.cpu_device]
    if config.cupy_enabled:
        devices.append(backend.Device(0))

    class TestPytorch(unittest.TestCase):

        def test_to_pytorch(self):
            for dtype in [np.float32, np.float64]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        xp = device.xp
                        array = xp.array([1, 2, 3], dtype=dtype)
                        tensor = pytorch.to_pytorch(array)
                        array[0] = 0
                        torch.testing.assert_allclose(
                            tensor, torch.tensor([0, 2, 3],
                                                 dtype=tensor.dtype,
                                                 device=tensor.device))

        def test_to_pytorch_complex(self):
            for dtype in [np.complex64, np.complex128]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        xp = device.xp
                        array = xp.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=dtype)
                        tensor = pytorch.to_pytorch(array)
                        array[0] = 0
                        torch.testing.assert_allclose(
                            tensor, torch.tensor([0, 2 + 2j, 3 + 3j],
                                                 dtype=tensor.dtype,
                                                 device=tensor.device))

        def test_from_pytorch(self):
            for dtype in [torch.float32, torch.float64]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        if device == backend.cpu_device:
                            torch_device = torch.device('cpu')
                        else:
                            torch_device = torch.device('cuda:0')

                        tensor = torch.tensor([1, 2, 3],
                                              dtype=dtype, device=torch_device)
                        array = pytorch.from_pytorch(tensor)
                        array[0] = 0
                        np.testing.assert_allclose(tensor.cpu().numpy(),
                                                   [0, 2, 3])

        def test_from_pytorch_complex(self):
            for dtype in [torch.complex64, torch.complex128]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        if device == backend.cpu_device:
                            torch_device = torch.device('cpu')
                        else:
                            torch_device = torch.device('cuda:0')

                        tensor = torch.tensor([1+1j, 2+2j, 3+3j],
                                              dtype=dtype, device=torch_device)
                        tensor[0] = 0
                        array = pytorch.from_pytorch(tensor)
                        xp = device.xp
                        xp.testing.assert_array_equal(array,
                                                      [0, 2 + 2j, 3 + 3j])

        def test_to_pytorch_function(self):
            A = linop.Resize([5], [3])
            x = np.array([1, 2, 3], np.float32)
            y = np.ones([5])

            with self.subTest('forward'):
                f = pytorch.to_pytorch_function(A).apply
                x_torch = pytorch.to_pytorch(x)
                npt.assert_allclose(f(x_torch).detach().numpy(),
                                    A(x))

            with self.subTest('adjoint'):
                y_torch = pytorch.to_pytorch(y)
                loss = (f(x_torch) - y_torch).pow(2).sum() / 2
                loss.backward()
                npt.assert_allclose(x_torch.grad.detach().numpy(),
                                    A.H(A(x) - y))

        def test_to_pytorch_function_complex(self):
            for device in devices:

                if device == backend.cpu_device:
                    torch_device = torch.device('cpu')
                else:
                    torch_device = torch.device('cuda:0')

                A = linop.FFT([3])
                A_torch = pytorch.to_pytorch_function(A)

                x = device.xp.array([1 + 1j, 2 + 2j, 3 + 3j], np.complex64)
                x_torch = torch.tensor([1 + 1j, 2 + 2j, 3 + 3j],
                                       dtype=torch.complex64,
                                       device=torch_device,
                                       requires_grad=True)

                y = device.xp.ones([3], np.complex64)
                y_torch = torch.ones([3],
                                     dtype=torch.complex64,
                                     device=torch_device)

                with self.subTest('forward'):
                    npt.assert_allclose(
                        A_torch.apply(x_torch).detach().cpu().numpy(),
                        backend.to_device(A(x), backend.cpu_device))

                with self.subTest('adjoint'):
                    loss = torch.abs(A_torch.apply(x_torch) - y_torch)\
                               .pow(2).sum() / 2
                    loss.backward()
                    npt.assert_allclose(
                        x_torch.grad.detach().cpu().numpy(),
                        backend.to_device(A.H(A(x) - y), backend.cpu_device))
