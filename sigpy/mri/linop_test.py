import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import linop
from sigpy.linop_test import check_linop_adjoint

if __name__ == '__main__':
    unittest.main()


class TestLinop(unittest.TestCase):

    def test_sense_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        mask = np.zeros(img_shape)
        mask[::2, ::2] = 1.0

        A = linop.Sense(mps)

        check_linop_adjoint(A, dtype=np.complex)

        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]),
                            A * img)

    def test_sense_model_batch(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        mask = np.zeros(img_shape, dtype=np.complex)
        mask[::2, ::2] = 1.0

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

        A = linop.Sense(mps, coord=coord)
        check_linop_adjoint(A, dtype=np.complex)
        npt.assert_allclose(sp.fft(img * mps, axes=[-1, -2]).ravel(),
                            (A * img).ravel(), atol=0.1, rtol=0.1)

    def test_noncart_sense_model_batch(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        img = sp.randn(img_shape, dtype=np.complex)
        mps = sp.randn(mps_shape, dtype=np.complex)

        y, x = np.mgrid[:16, :16]
        coord = np.stack([np.ravel(y - 8), np.ravel(x - 8)], axis=1)

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
            
            npt.assert_allclose(A.H(ksp[comm.rank::comm.size]),
                                np.sum(sp.ifft(ksp, axes=[-1, -2]) * mps.conjugate(), 0))
