import unittest
import numpy as np
import numpy.testing as npt

from sigpy import util
from sigpy.mri import epi

if __name__ == '__main__':
    unittest.main()


class TestEpi(unittest.TestCase):

    def test_get_B(self):
        b_0 = np.zeros((1, 1))
        b_1 = np.ones((6, 1))

        b = np.concatenate((b_0, b_1))
        g = np.array([[ 0         , 0         , 0          ],
                      [ 1         , 0         , 0          ],
                      [ 0         , 1         , 0          ],
                      [ 0         , 0         , 1          ],
                      [ 1 / 2**0.5, 1 / 2**0.5, 0          ],
                      [ 1 / 2**0.5, 0         , 1 / 2**0.5 ],
                      [ 0         , 1 / 2**0.5, 1 / 2**0.5 ]])

        B1 = epi.get_B(b, g)
        #                 xx,  xy,  yy,xz,yz,zz
        B2 = np.array([[ 0  , 0  , 0  , 0, 0, 0 ],
                       [ 1  , 0  , 0  , 0, 0, 0 ],
                       [ 0  , 0  , 1  , 0, 0, 0 ],
                       [ 0  , 0  , 0  , 0, 0, 1 ],
                       [ 0.5, 1  , 0.5, 0, 0, 0 ],
                       [ 0.5, 0  , 0  , 1, 0, 0.5 ],
                       [ 0  , 0  , 0.5, 0, 1, 0.5 ]]) * -1.

        npt.assert_allclose(B1, B2, atol=1e-5, rtol=1e-5)

    def test_get_D(self):
        ndif = 81
        nlin = 3
        ncol = 3

        b_0 = np.zeros((1, 1))
        b_100 = np.ones((20, 1)) * 100
        b_500 = np.ones((20, 1)) * 500
        b_1000 = np.ones((40, 1)) * 1000

        b = np.concatenate((b_0, b_100, b_500, b_1000))

        g0 = np.zeros((1, 3))
        g1 = np.random.normal(size=(ndif-1, 3), loc=0, scale=0.212)

        g = np.concatenate((g0, g1))

        B = epi.get_B(b, g)

        D = util.randn((6, nlin, ncol)) * 1e-3
        Dr = D.reshape(6, -1)

        S0 = np.abs(util.randn((nlin, ncol)))
        sig = S0 * np.exp(-np.matmul(B, Dr).reshape(ndif, nlin, ncol))

        Dinv = epi.get_D(B, sig, fit_only_tensor=True)

        npt.assert_allclose(Dinv, D, atol=1e-5, rtol=1e-5)

    def test_get_eig(self):
        #                xx,   xy,   yy,    xz,   yz,   zz
        D = np.array([[2.00,  0.00, 1.00, 0.00, 0.00, 0.50],
                      [1.75, -0.43, 1.25, 0.00, 0.00, 0.50],
                      [1.50, -0.50, 1.50, 0.00, 0.00, 0.50],
                      [1.00,  0.00, 2.00, 0.00, 0.00, 0.50]])

        eigvals_exp = np.array([2.0, 1.0, 0.5])

        for n in range(0, D.shape[0]):
            Dn = np.reshape(D[n, :], (6, 1, 1))
            eigvals, _ = epi.get_eig(Dn)
            npt.assert_allclose(np.squeeze(eigvals), eigvals_exp,
                                atol=3e-3, rtol=3e-3)

    # def test_comp_dipy(self):
    #     ndif = 115
    #     nlin = 3
    #     ncol = 3

    #     b_0 = np.zeros((1, 1))
    #     b_1000 = np.ones((20, 1)) * 1000
    #     b_2000 = np.ones((30, 1)) * 2000
    #     b_3000 = np.ones((64, 1)) * 3000

    #     b = np.concatenate((b_0, b_1000, b_2000, b_3000))

    #     g0 = np.zeros((1, 3))
    #     g1 = np.random.normal(size=(ndif-1, 3), loc=0, scale=0.212)
    #     gsum = np.sum(g1**2, axis=1)**0.5
    #     g1 = g1 / gsum[:, np.newaxis]

    #     g = np.concatenate((g0, g1))

    #     B = epi.get_B(b, g)

    #     D = util.randn((6, nlin, ncol)) * 1e-3
    #     Dr = D.reshape(6, -1)

    #     S0 = np.abs(util.randn((nlin, ncol)))
    #     sig = S0 * np.exp(-np.matmul(B, Dr).reshape(ndif, nlin, ncol))

    #     Di = epi.get_D(B, sig)
    #     evals, evecs = epi.get_eig(Di, B=B)
    #     FA = epi.get_FA(evals)


    #     from dipy.core.gradients import gradient_table

    #     gtab = gradient_table(b.flatten(), g, atol=0.1)

    #     import dipy.reconst.dti as dti
    #     from dipy.reconst.dti import fractional_anisotropy, color_fa
    #     tenmodel = dti.TensorModel(gtab)

    #     tenfit = tenmodel.fit(np.transpose(sig, axes=(1, 2, 0)))

    #     evals_dp = np.transpose(tenfit.evals, axes=(2, 0, 1))
    #     evecs_dp = np.transpose(tenfit.evecs, axes=(2, 3, 0, 1))
    #     FA_dp = fractional_anisotropy(tenfit.evals)


    #     evecs0_dp = np.transpose(tenfit.evecs[..., 0], axes=(2, 0, 1))

    #     npt.assert_allclose(evals, evals_dp, atol=3e-3, rtol=3e-3)
    #     npt.assert_allclose(evecs, evecs_dp, atol=3e-3, rtol=3e-3)
    #     npt.assert_allclose(FA, FA_dp, atol=3e-3, rtol=3e-3)
