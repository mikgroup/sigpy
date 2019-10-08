import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import dcf, samp

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def shepp_logan_setup(self):
        img_shape = [16, 16]
        coord_shape = [int(16 * np.pi), 16, 2]

        img = sp.shepp_logan(img_shape)
        coord = samp.radial(coord_shape, img_shape)
        ksp = sp.nufft(img, coord)
        return img, coord, ksp

    def test_shepp_logan_dcf(self):
        img, coord, ksp = self.shepp_logan_setup()
        pm_dcf = dcf.pipe_menon_dcf(coord, show_pbar=False)
        img_dcf = sp.nufft_adjoint(ksp * pm_dcf, coord, oshape=img.shape)
        img_dcf /= np.abs(img_dcf).max()
        npt.assert_allclose(img, img_dcf, atol=1, rtol=1e-1)
