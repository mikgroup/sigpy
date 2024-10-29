import unittest
import numpy as np
import numpy.testing as npt
from sigpy import app, util
from sigpy.mri import dce

if __name__ == '__main__':
    unittest.main()

class TestDCE(unittest.TestCase):

    def test_dce(self):

        # %% DCE Sample Time
        acq_time = 10 # total acquisition time (minute)
        temp_res = 12 # frames per minute

        frames = acq_time * temp_res + 1

        delay = 8     # baseline frames

        sample_time_0 = np.zeros([1, delay])
        sample_time_1 = (np.arange(1, frames-delay+1, 1) * (1/temp_res)).reshape((1, -1))

        sample_time = np.hstack((sample_time_0, sample_time_1))

        # %% DCE Parameters (K_trans, v_p)^T
        K_trans = np.array([0.0402, 0.2505]).reshape((1, -1))
        v_p = np.array([0.05, 0.06]).reshape((1, -1))

        param = []
        param.append(K_trans)
        param.append(v_p)

        param = np.array(param)
        param = param[:, None, None, :, :]

        # %% DCE Model
        DCE = dce.DCE(param.shape, sample_time)
        sig = DCE(param) + util.randn(DCE.oshape) * 1E-6

        # %% NLLS Solver
        x = np.ones_like(param, dtype=complex) * 0.1

        x = app.NonLinearLeastSquares(DCE, sig, x=x,
                                      lamda=1E-3, redu=3,
                                      gn_iter=8,
                                      inner_iter=100,
                                      show_pbar=False,
                                      verbose=True).run()

        npt.assert_allclose(x, param, rtol=1E-5, atol=1E-5)