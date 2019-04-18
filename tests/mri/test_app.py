import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import app, sim

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def shepp_logan_setup(self):
        img_shape = [6, 6]
        mps_shape = [4, 6, 6]

        img = sp.shepp_logan(img_shape)
        mps = sim.birdcage_maps(mps_shape)

        mask = np.zeros(img_shape)
        mask[:, ::2] = 1

        ksp = mask * sp.fft(mps * img, axes=[-2, -1])
        return img, mps, ksp

    def test_shepp_logan_SenseRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        lamda = 0

        img_rec = app.SenseRecon(
            ksp, mps, lamda, alg_name='ConjugateGradient',
            show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

        img_rec = app.SenseRecon(
            ksp, mps, lamda, alg_name='GradientMethod', show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

        img_rec = app.SenseRecon(
            ksp,
            mps,
            lamda,
            alg_name='PrimalDualHybridGradient',
            max_iter=1000,
            show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    if sp.config.mpi4py_enabled:
        def test_shepp_logan_SenseRecon_with_comm(self):
            img, mps, ksp = self.shepp_logan_setup()
            lamda = 0
            comm = sp.Communicator()
            ksp = ksp[comm.rank::comm.size]
            mps = mps[comm.rank::comm.size]

            img_rec = app.SenseRecon(
                ksp, mps, lamda, comm=comm, alg_name='ConjugateGradient',
                show_pbar=False).run()
            npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

            img_rec = app.SenseRecon(
                ksp, mps, lamda, alg_name='GradientMethod',
                show_pbar=False).run()
            npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

            img_rec = app.SenseRecon(
                ksp,
                mps,
                lamda,
                alg_name='PrimalDualHybridGradient',
                max_iter=1000,
                show_pbar=False).run()
            npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    def test_shepp_logan_SenseConstrainedRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        std = 0

        img_rec = app.SenseConstrainedRecon(ksp, mps, std,
                                            show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    def test_shepp_logan_L1WaveletRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        lamda = 0

        img_rec = app.L1WaveletRecon(
            ksp, mps, lamda, alg_name='GradientMethod',
            show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

        img_rec = app.L1WaveletRecon(
            ksp,
            mps,
            lamda,
            alg_name='PrimalDualHybridGradient',
            max_iter=1000, show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    def test_shepp_logan_L1WaveletConstrainedRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        std = 0

        img_rec = app.L1WaveletConstrainedRecon(
            ksp, mps, std, max_iter=1000, show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    def test_shepp_logan_TotalVariationRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        lamda = 0
        img_rec = app.TotalVariationRecon(ksp, mps, lamda, max_iter=1000,
                                          show_pbar=False).run()

        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    def test_shepp_logan_TotalVariationConstrainedRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        std = 0

        img_rec = app.TotalVariationConstrainedRecon(
            ksp, mps, std, max_iter=2000, show_pbar=False).run()
        npt.assert_allclose(img, img_rec, atol=1e-3, rtol=1e-3)

    def test_ones_JsenseRecon(self):
        img_shape = [6, 6]
        mps_shape = [4, 6, 6]

        img = np.ones(img_shape, dtype=np.complex)
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps * img, axes=[-2, -1])

        _app = app.JsenseRecon(ksp, mps_ker_width=6, ksp_calib_width=6,
                               show_pbar=False)
        mps_rec = _app.run()

        npt.assert_allclose(mps, mps_rec, atol=1e-3, rtol=1e-3)

    def test_espirit_maps(self):
        mps_shape = [8, 32, 32]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec = app.EspiritCalib(ksp, show_pbar=False).run()

        np.testing.assert_allclose(np.abs(mps)[:, 8:24, 8:24],
                                   np.abs(mps_rec[:, 8:24, 8:24]),
                                   rtol=1e-3, atol=1e-3)

    def test_espirit_maps_eig(self):
        mps_shape = [8, 32, 32]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec, eig_val = app.EspiritCalib(
            ksp, output_eigenvalue=True, show_pbar=False).run()

        np.testing.assert_allclose(eig_val, 1, rtol=0.01, atol=0.01)
