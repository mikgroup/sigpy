import unittest
import numpy as np
import numpy.testing as npt
from sigpy import wavelet

if __name__ == '__main__':
    unittest.main()


class TestWavelet(unittest.TestCase):

    def test_fwt_iwt(self):
        for n in [5, 6, 7, 8, 9, 10]:
            input = np.arange(n, dtype=np.float)
            npt.assert_allclose(wavelet.iwt(wavelet.fwt(input), [n]), input)
