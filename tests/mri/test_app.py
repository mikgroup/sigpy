import unittest
import numpy as np
import sigpy as sp
import scipy.ndimage.filters as filt
import numpy.testing as npt

from sigpy.mri import app, sim
from sigpy.mri import linop

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

        for solver in ['ConjugateGradient',
                       'GradientMethod',
                       'PrimalDualHybridGradient',
                       'ADMM']:
            with self.subTest(solver=solver):
                img_rec = app.SenseRecon(
                    ksp, mps, lamda, solver=solver,
                    show_pbar=False).run()
                npt.assert_allclose(img, img_rec, atol=1e-2, rtol=1e-2)

    if sp.config.mpi4py_enabled:
        def test_shepp_logan_SenseRecon_with_comm(self):
            img, mps, ksp = self.shepp_logan_setup()
            lamda = 0
            comm = sp.Communicator()
            ksp = ksp[comm.rank::comm.size]
            mps = mps[comm.rank::comm.size]

            for solver in ['ConjugateGradient',
                           'GradientMethod',
                           'PrimalDualHybridGradient',
                           'ADMM']:
                with self.subTest(solver=solver):
                    img_rec = app.SenseRecon(
                        ksp, mps, lamda, comm=comm, solver=solver,
                        show_pbar=False).run()
                    npt.assert_allclose(img, img_rec, atol=1e-2, rtol=1e-2)

    def test_shepp_logan_L1WaveletRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        lamda = 0

        for solver in ['GradientMethod',
                       'PrimalDualHybridGradient',
                       'ADMM']:
            with self.subTest(solver=solver):
                img_rec = app.L1WaveletRecon(
                    ksp, mps, lamda, solver=solver,
                    show_pbar=False).run()
                npt.assert_allclose(img, img_rec, atol=1e-2, rtol=1e-2)

    def test_shepp_logan_TotalVariationRecon(self):
        img, mps, ksp = self.shepp_logan_setup()
        lamda = 0
        for solver in ['PrimalDualHybridGradient',
                       'ADMM']:
            with self.subTest(solver=solver):
                img_rec = app.TotalVariationRecon(
                    ksp, mps, lamda,
                    solver=solver, max_iter=1000, show_pbar=False).run()

                npt.assert_allclose(img, img_rec, atol=1e-2, rtol=1e-2)

    def test_ones_JsenseRecon(self):
        img_shape = [6, 6]
        mps_shape = [4, 6, 6]

        img = np.ones(img_shape, dtype=np.complex)
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps * img, axes=[-2, -1])

        _app = app.JsenseRecon(ksp, mps_ker_width=6, ksp_calib_width=6,
                               show_pbar=False)
        mps_rec = _app.run()

        npt.assert_allclose(mps, mps_rec, atol=1e-2, rtol=1e-2)

    def test_espirit_maps(self):
        mps_shape = [8, 32, 32]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec = app.EspiritCalib(ksp, show_pbar=False).run()

        np.testing.assert_allclose(np.abs(mps)[:, 8:24, 8:24],
                                   np.abs(mps_rec[:, 8:24, 8:24]),
                                   rtol=1e-2, atol=1e-2)

    def test_espirit_maps_eig(self):
        mps_shape = [8, 32, 32]
        mps = sim.birdcage_maps(mps_shape)
        ksp = sp.fft(mps, axes=[-1, -2])
        mps_rec, eig_val = app.EspiritCalib(
            ksp, output_eigenvalue=True, show_pbar=False).run()

        np.testing.assert_allclose(eig_val, 1, rtol=0.01, atol=0.01)

    def test_radial_SmallTipSpatialDomain(self):
        img_shape = [32, 32]
        sens_shape = [8, 32, 32]

        # target - slightly blurred circle
        x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
                        -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
        circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
        target = np.zeros(img_shape)
        target[circle] = 1
        target = filt.gaussian_filter(target, 1)
        target = target.astype(np.complex)

        sens = sp.mri.sim.birdcage_maps(sens_shape)

        traj = sp.mri.radial((sens_shape[1], sens_shape[1], 2),
                             img_shape, golden=True, dtype=np.float)

        # reshape to be Nt*2 trajectory
        traj = np.reshape(traj, [traj.shape[0] * traj.shape[1], 2])

        A = linop.Sense(sens, coord=traj,
                        weights=None, ishape=img_shape).H

        pulses = app.SpatialPtxPulses(target, sens, coord=traj, lamda=0.01,
                                      max_iter=1000, tol=1e-3,
                                      show_pbar=False).run()

        npt.assert_array_almost_equal(A * pulses, target, 1e-3)
