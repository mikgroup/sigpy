import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import espirit, sim

if __name__ == '__main__':
    unittest.main()


class TestEspirit(unittest.TestCase):

    def test_espirit_calib(self):
        mps_shape = [8, 16, 16]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec = espirit.espirit_calib(ksp)

        np.testing.assert_allclose(np.abs(mps), np.abs(mps_rec),
                                   rtol=0.2, atol=0.2)
