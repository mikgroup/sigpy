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
                        tensor[0] = 0
                        xp.testing.assert_allclose(array, [0, 2, 3])

        def test_to_pytorch_complex(self):
            for dtype in [np.complex64, np.complex128]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        xp = device.xp
                        array = xp.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=dtype)
                        tensor = pytorch.to_pytorch(array)
                        tensor[0, 0] = 0
                        xp.testing.assert_allclose(array, [1j, 2 + 2j, 3 + 3j])

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
            for dtype in [torch.float32, torch.float64]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        if device == backend.cpu_device:
                            torch_device = torch.device('cpu')
                        else:
                            torch_device = torch.device('cuda:0')

                        tensor = torch.tensor([[1, 1], [2, 2], [3, 3]],
                                              dtype=dtype, device=torch_device)
                        array = pytorch.from_pytorch(tensor, iscomplex=True)
                        xp = device.xp
                        xp.testing.assert_array_equal(array,
                                                      [1 + 1j, 2 + 2j, 3 + 3j])
                        array[0] -= 1
                        np.testing.assert_allclose(tensor.cpu().numpy(),
                                                   [[0, 1], [2, 2], [3, 3]])

        def test_to_pytorch_function(self):
            A = linop.Resize([5], [3])
            x = np.array([1, 2, 3], np.float)
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
            A = linop.FFT([3])
            x = np.array([1 + 1j, 2 + 2j, 3 + 3j], np.complex)
            y = np.ones([3], np.complex)

            with self.subTest('forward'):
                f = pytorch.to_pytorch_function(
                    A,
                    input_iscomplex=True,
                    output_iscomplex=True).apply
                x_torch = pytorch.to_pytorch(x)
                npt.assert_allclose(f(x_torch).detach().numpy().ravel(),
                                    A(x).view(np.float))

            with self.subTest('adjoint'):
                y_torch = pytorch.to_pytorch(y)
                loss = (f(x_torch) - y_torch).pow(2).sum() / 2
                loss.backward()
                npt.assert_allclose(x_torch.grad.detach().numpy().ravel(),
                                    A.H(A(x) - y).view(np.float))
