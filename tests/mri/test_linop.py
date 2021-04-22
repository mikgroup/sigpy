import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import linop

if __name__ == '__main__':
    unittest.main()


def check_linop_adjoint(A, dtype=np.float, device=sp.cpu_device):

    device = sp.Device(device)
    x = sp.randn(A.ishape, dtype=dtype, device=device)
    y = sp.randn(A.oshape, dtype=dtype, device=device)

    xp = device.xp
    with device:
        lhs = xp.vdot(A * x, y)
        rhs = xp.vdot(x, A.H * y)

        xp.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


class TestLinop(unittest.TestCase):

    def test_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        A = linop.Sense(mps)

        check_linop_adjoint(A, dtype=np.complex)

        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]),
                            A * img)

    def test_sense_model_batch(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        A = linop.Sense(mps, coil_batch_size=1)
        check_linop_adjoint(A, dtype=np.complex)
        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]),
                            A * img)

    def test_noncart_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)
        coord = coord.astype(np.float)

        A = linop.Sense(mps, coord=coord)
        check_linop_adjoint(A, dtype=np.complex)
        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]).ravel(),
                            (A * img).ravel(), atol=0.1, rtol=0.1)

    def test_sense_tseg_off_res_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)
        coord = coord.astype(np.float)

        d = np.sqrt(x * x + y * y)
        sigma, mu, a = 2, 0.25, 400
        b0 = a * np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        tseg = {"b0": b0, "dt": 4e-6, "lseg": 1, "n_bins": 10}

        F = sp.linop.NUFFT(mps_shape, coord)
        b, ct = sp.mri.util.tseg_off_res_b_ct(b0=b0, bins=10, lseg=1, dt=4e-6,
                                              T=coord.shape[0] * 4e-6)
        B1 = sp.linop.Multiply(F.oshape, b.T)
        Ct1 = sp.linop.Multiply(img_shape, ct.reshape(img_shape))
        S = sp.linop.Multiply(img_shape, mps)

        A = linop.Sense(mps, coord=coord, tseg=tseg)

        check_linop_adjoint(A, dtype=np.complex)
        npt.assert_allclose(B1 * F * S * Ct1 * img, A * img)

    def test_noncart_sense_model_batch(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)
        coord = coord.astype(np.float)

        A = linop.Sense(mps, coord=coord, coil_batch_size=1)
        check_linop_adjoint(A, dtype=np.complex)
        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]).ravel(),
                            (A * img).ravel(), atol=0.1, rtol=0.1)

    if sp.config.mpi4py_enabled:
        def test_sense_model_with_comm(self):
            img_shape = [16, 16]
            mps_shape = [8, 16, 16]
            comm = sp.Communicator()

            img = sp.randn(img_shape, dtype=np.complex)
            mps = sp.randn(mps_shape, dtype=np.complex)
            comm.allreduce(img)
            comm.allreduce(mps)
            ksp = sp.fft(img * mps, axes=[-1, -2])

            A = linop.Sense(mps[comm.rank::comm.size], comm=comm)

            npt.assert_allclose(A.H(ksp[comm.rank::comm.size]), np.sum(
                sp.ifft(ksp, axes=[-1, -2]) * mps.conjugate(), 0))
