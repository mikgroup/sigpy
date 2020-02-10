import unittest
import numpy as np
import sigpy as sp
import sigpy.mri.rf as rf

from sigpy.mri.rf import linop

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

    def test_spatial_explicit_model(self):
        img_shape = [16, 16]
        mps_shape = [8, 16, 16]

        fov = 22.4
        N = 32
        gts = 6.4e-6
        gslew = 9000 / 100
        amp = 3
        dt = 4e-6

        g, k, t, s, dens = rf.spiral_varden(fov, N, gts, gslew, amp, 75, 75, 2)

        mps = sp.randn(mps_shape, dtype=np.complex)

        A = linop.PtxSpatialExplicit(mps, k, dt, img_shape)

        check_linop_adjoint(A, dtype=np.complex)
