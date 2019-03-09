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

    def test_espirit_maps_eig(self):
        mps_shape = [8, 32, 32]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec, eig_val = espirit.espirit_maps(ksp,
                                                output_eigenvalue=True)

        np.testing.assert_allclose(eig_val, 1, rtol=0.01, atol=0.01)
