import unittest
import numpy as np
import sigpy as sp

from sigpy.mri import espirit, sim

if __name__ == '__main__':
    unittest.main()


class TestEspirit(unittest.TestCase):

    def test_espirit_maps(self):
        mps_shape = [8, 32, 32]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec = espirit.espirit_maps(ksp)

        np.testing.assert_allclose(np.abs(mps)[:, 8:24, 8:24],
                                   np.abs(mps_rec[:, 8:24, 8:24]),
                                   rtol=1e-3, atol=1e-3)
