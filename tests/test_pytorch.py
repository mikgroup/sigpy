import unittest
import numpy as np
from sigpy import backend, config, pytorch


if config.pytorch_enabled:
    import torch

    if __name__ == '__main__':
        unittest.main()

    devices = [backend.cpu_device]
    if config.cupy_enabled:
        devices.append(backend.Device(0))

    class TestPytorch(unittest.TestCase):

        def test_to_torch(self):
            for dtype in [np.float32, np.float64]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        xp = device.xp
                        array = xp.array([1, 2, 3], dtype=dtype)
                        tensor = pytorch.to_pytorch(array)
                        tensor[0] = 0
                        xp.testing.assert_allclose(array, [0, 2, 3])

        def test_to_torch_complex(self):
            for dtype in [np.complex64, np.complex128]:
                for device in devices:
                    with self.subTest(device=device, dtype=dtype):
                        xp = device.xp
                        array = xp.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=dtype)
                        tensor = pytorch.to_pytorch(array)
                        tensor[0, 0] = 0
                        xp.testing.assert_allclose(array, [1j, 2 + 2j, 3 + 3j])

        def test_from_torch(self):
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

        def test_from_torch_complex(self):
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
                        array[0] -= 1
                        np.testing.assert_allclose(tensor.cpu().numpy(),
                                                   [[0, 1], [2, 2], [3, 3]])
