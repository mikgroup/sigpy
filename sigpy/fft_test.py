import pickle
import unittest
import numpy as np
import numpy.testing as npt
from sigpy import fft, util

if __name__ == '__main__':
    unittest.main()


class TestFft(unittest.TestCase):

    def test_fft(self):
        input = np.array([0, 1, 0], dtype=np.complex)
        npt.assert_allclose(fft.fft(input),
                            np.ones(3) / 3**0.5, atol=1e-5)

        input = np.array([1, 1, 1], dtype=np.complex)
        npt.assert_allclose(fft.fft(input),
                            [0, 3**0.5, 0], atol=1e-5)

        input = util.randn([4, 5, 6])
        npt.assert_allclose(fft.fft(input),
                            np.fft.fftshift(np.fft.fftn(
                                np.fft.ifftshift(input), norm='ortho')),
                            atol=1e-5)

        input = np.array([0, 1, 0], dtype=np.complex)
        npt.assert_allclose(fft.fft(input, oshape=[5]),
                            np.ones(5) / 5**0.5, atol=1e-5)

    def test_fft_dtype(self):

        for dtype in [np.complex64, np.complex128]:
            input = np.array([0, 1, 0], dtype=dtype)
            output = fft.fft(input)

            assert output.dtype == dtype

    def test_ifft(self):
        input = np.array([0, 1, 0], dtype=np.complex)
        npt.assert_allclose(fft.ifft(input),
                            np.ones(3) / 3 ** 0.5, atol=1e-5)

        input = np.array([1, 1, 1], dtype=np.complex)
        npt.assert_allclose(fft.ifft(input),
                            [0, 3**0.5, 0], atol=1e-5)

        input = util.randn([4, 5, 6])
        npt.assert_allclose(fft.ifft(input),
                            np.fft.fftshift(np.fft.ifftn(
                                np.fft.ifftshift(input), norm='ortho')),
                            atol=1e-5)

        input = np.array([0, 1, 0], dtype=np.complex)
        npt.assert_allclose(fft.ifft(input, oshape=[5]),
                            np.ones(5) / 5**0.5, atol=1e-5)
