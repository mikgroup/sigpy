import unittest
import numpy as np
import numpy.testing as npt
from sigpy import wavelet

if __name__ == '__main__':
    unittest.main()


class TestWavelet(unittest.TestCase):

    def test_fwt(self):
        n = 8
        input = np.zeros(n, dtype=np.float)
        input[0] = 1
        npt.assert_allclose(wavelet.fwt(input, level=1, wave_name='haar'),
                            [1 / 2**0.5, 0, 0, 0, 1 / 2**0.5, 0, 0, 0])

    def test_fwt_iwt(self):
        for n in range(5, 11):
            input = np.zeros(n, dtype=np.float)
            input[0] = 1
            _, coeff_slices = wavelet.get_wavelet_shape([n])
            npt.assert_allclose(
                wavelet.iwt(wavelet.fwt(input), [n], coeff_slices), input)
